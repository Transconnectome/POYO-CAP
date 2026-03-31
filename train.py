# -*- coding: utf-8 -*-



import os
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("WORLD_SIZE =", os.environ.get("WORLD_SIZE"),
      "RANK =", os.environ.get("RANK"),
      "LOCAL_RANK =", os.environ.get("LOCAL_RANK"))
print("torch.cuda.device_count() =", torch.cuda.device_count())
print("current_device =", torch.cuda.current_device())


import numpy as np
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric import Fabric

from lightning.pytorch.callbacks import Callback


fabric = Fabric(
    accelerator="gpu",
    devices="auto",
    plugins=[SLURMEnvironment()]
)
# print("RANK:", os.environ.get("RANK"))
# print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))
# print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"))
print("SLURM_PROCID:", os.environ.get("SLURM_PROCID"))
print("SLURM_LOCALID:", os.environ.get("SLURM_LOCALID"))

import pickle
import pandas as pd

old_unpickler = pickle.Unpickler  # Unfortunate hack to fix a bug in Lightning.
# https://github.com/Lightning-AI/lightning/issues/18152
# Will likely be fixed by 2.1.0.
import lightning
from lightning.pytorch.strategies import DDPStrategy

pickle.Unpickler = old_unpickler

from collections import OrderedDict
import copy
import logging

import hydra
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)

import sys
# Add the repo root to sys.path so that the kirby package can be found
# regardless of the working directory from which this script is launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Flags are absorbed by Hydra.
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_optimizer import Lamb
from kirby.optim import SparseLamb


from kirby.data import Dataset, collate
from kirby.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
    DistributedSamplerWrapper,
)
from kirby.taxonomy import decoder_registry
from kirby.transforms import Compose
from kirby.utils import seed_everything, train_wrapper
from kirby.models.capoyo import CaPOYOTokenizer
from kirby.utils.gradient_rescale import UnitEmbeddingGradientRescaling

# wandb_project = os.environ.get("WANDB_PROJECT")
# wandb_entity = os.environ.get("WANDB_ENTITY")

if os.getenv("RANK", "0") != "0":  # if not global rank 0
    os.environ["WANDB_DISABLED"] = "true"  # disable wandb for this process

class GradientChecker(Callback):
    def on_after_backward(self, trainer, pl_module):
        # check gradients only at a specific early step (e.g., step 10)
        if trainer.global_step == 10:
            print("\n--- Full Gradient Inspection (Global Step 10) ---")

            none_grad_params = []
            zero_grad_params = []

            # iterate over all model parameters
            for name, param in pl_module.model.named_parameters():
                # only target trainable parameters
                if param.requires_grad:
                    if param.grad is None:
                        none_grad_params.append(name)
                    elif torch.all(param.grad == 0):
                        zero_grad_params.append(name)

            # print summary
            if none_grad_params:
                print("CRITICAL: the following parameters have no gradient (None):")
                for name in none_grad_params:
                    print(f"  - {name}")

            if zero_grad_params:
                print("WARNING: the following parameters have all-zero gradients:")
                for name in zero_grad_params:
                    print(f"  - {name}")

            if not none_grad_params and not zero_grad_params:
                print("All trainable parameters have non-zero gradients.")

            print("--------------------------------------------------\n")


def get_object_size(obj):
    """Recursively measure object size."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.nbytes if hasattr(obj, 'nbytes') else obj.element_size() * obj.nelement()
    elif isinstance(obj, (list, tuple)):
        return sum(get_object_size(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(get_object_size(k) + get_object_size(v) for k, v in obj.items())
    else:
        return sys.getsizeof(obj)


def run_training(cfg: DictConfig):
    # Fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    # Higher speed on machines with tensor cores.
    torch.set_float32_matmul_precision("medium")

    log = logging.getLogger(__name__)

    # Device setup is managed by PyTorch Lightning.

    # make model
    model = hydra.utils.instantiate(
        cfg.model,
        decoder_specs=decoder_registry,
        _convert_="object",
    )

    # prepare tokenizer and transforms

    # The transform list is defined in the config file.
    # sequence_length = 1
    transforms = hydra.utils.instantiate(
        cfg.train_transforms, sequence_length=cfg.sequence_length
    )

    # build tokenizer
    tokenizer = CaPOYOTokenizer(
        model.unit_emb.tokenizer,
        model.session_emb.tokenizer,
        decoder_registry=decoder_registry,
        dim=model.dim_input,
        patch_size=model.patch_size,
        latent_step=cfg.get("latent_step", 1.0 / 8.0),
        num_latents_per_step=cfg.model.num_latents,
        batch_type=model.batch_type,
        use_cre_line_embedding=model.use_cre_line_embedding,
        use_depth_embedding=model.use_depth_embedding,
        use_spatial_embedding=model.use_spatial_embedding,
        use_roi_feat_embedding=model.use_roi_feat_embedding,
        use_cre_loss=model.use_cre_loss,
        use_cre_data=model.use_cre_data,
        use_random_masking_loss=model.use_random_masking_loss,
        no_decoding=model.no_decoding,
        task=model.task,
        unet_type=model.unet_type)

    transform = Compose([*transforms, tokenizer])

    eval_only = cfg.get("eval_only", False)

    if eval_only:
        # eval_only mode: use all data as test set
        test_tokenizer = copy.copy(tokenizer)
        test_tokenizer.eval = True
        test_dataset = Dataset(
            cfg.data_root,
            "all",  # merge train+valid+test
            include=OmegaConf.to_container(cfg.dataset),
            transform=test_tokenizer,
            pretrain=False,
            finetune=cfg.finetune,
            small_model=cfg.small_model,
            task=cfg.task,
            model_dim=model.dim,
            zscore_normalize=cfg.get("zscore_normalize", False),
        )
        test_dataset.disable_data_leakage_check()
        print(f'eval_only dataset length: {len(test_dataset)}')

    elif cfg.pretrain:
        pretrain_dataset = Dataset(
            cfg.data_root,
            "pretrain",
            include=OmegaConf.to_container(cfg.dataset),  # converts to native list[dicts]
            transform=transform,
            pretrain=True,
            finetune=cfg.finetune,
            small_model=cfg.small_model,
            task=cfg.task,
            model_dim=model.dim,
            zscore_normalize=cfg.get("zscore_normalize", False),
        )
        pretrain_dataset.disable_data_leakage_check()
    else:
        train_dataset = Dataset(
            cfg.data_root,
            "train",
            include=OmegaConf.to_container(cfg.dataset),  # converts to native list[dicts]
            transform=transform,
            pretrain=False,
            finetune=cfg.finetune,
            small_model=cfg.small_model,
            task=cfg.task,
            model_dim=model.dim,
            zscore_normalize=cfg.get("zscore_normalize", False),
        )
        train_dataset.disable_data_leakage_check()
        # In Lightning, testing only happens once, at the end of training. To get the
        # intended behavior, we need to specify a validation set instead.
        val_tokenizer = copy.copy(tokenizer)
        val_tokenizer.eval = True
        val_dataset = Dataset(
            cfg.data_root,
            "valid",
            include=OmegaConf.to_container(cfg.dataset),  # converts to native list[dicts]
            transform=val_tokenizer,
            pretrain=False,
            finetune=cfg.finetune,
            small_model=cfg.small_model,
            task=cfg.task,
            model_dim=model.dim,
            zscore_normalize=cfg.get("zscore_normalize", False),
        )

        print('valid dataset length', len(val_dataset))

        val_dataset.disable_data_leakage_check()

        # initialize test dataset
        test_tokenizer = copy.copy(tokenizer)
        test_tokenizer.eval = True
        test_dataset = Dataset(
            cfg.data_root,
            "test",
            include=OmegaConf.to_container(cfg.dataset),  # converts to native list[dicts]
            transform=test_tokenizer,
            pretrain=False,
            finetune=cfg.finetune,
            small_model=cfg.small_model,
            task=cfg.task,
            model_dim=model.dim,
            zscore_normalize=cfg.get("zscore_normalize", False),
        )
        print('test dataset length', len(test_dataset))

        test_dataset.disable_data_leakage_check()

    if eval_only:
        # eval_only: load checkpoint if provided, else run from scratch
        if cfg.ckpt_path is not None:
            model = load_model_from_ckpt(model, cfg.ckpt_path, finetune=True)
            log.info(f"[eval_only] Loaded model from {cfg.ckpt_path}")
        else:
            log.info("[eval_only] No checkpoint -> from scratch (random init)")

        # register unit/session vocabulary
        if cfg.finetune:
            try:
                model.unit_emb.extend_vocab(test_dataset.unit_ids, exist_ok=False)
            except ValueError as err:
                print(err)
            model.unit_emb.subset_vocab(test_dataset.unit_ids)

            try:
                model.session_emb.extend_vocab(test_dataset.session_ids, exist_ok=False)
            except ValueError as err:
                print(err)
            model.session_emb.subset_vocab(test_dataset.session_ids)
        else:
            model.unit_emb.initialize_vocab(test_dataset.unit_ids)
            model.session_emb.initialize_vocab(test_dataset.session_ids)

        # freeze entire model
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        log.info("[eval_only] All parameters frozen.")

    elif cfg.finetune:
        print('finetuning')
        if cfg.ckpt_path is not None:
            model = load_model_from_ckpt(model, cfg.ckpt_path, finetune=True)
            log.info(f"Loaded model state dict from {cfg.ckpt_path}")
        ## in case of finetuning
        # assert (
        #     cfg.ckpt_path is not None
        # ), "Missing `ckpt_path`. Checkpoint is required finetuning."



        # Optionally freeze parameters for Unit Identification
        if cfg.freeze_perceiver_until_epoch != 0:
            model.freeze_middle()
            log.info(f"Froze perceiver")


        if cfg.freeze_encoder == 'former':
            model.freeze_encoder_former()
            log.info(f"Froze Former Part of Encoder")

        elif cfg.freeze_encoder == 'middle':
            model.freeze_encoder_middle()
            log.info(f"Froze Middle Part of Encoder")

        elif cfg.freeze_encoder == 'latter':
            model.freeze_encoder_latter()
            log.info(f"Froze Latter Part of Encoder")

        elif cfg.freeze_encoder == 'whole':
            model.freeze_encoder()
            log.info(f"Froze Whole Encoder")

        # Register new units and sessions, and delete old ones
        try:
            model.unit_emb.extend_vocab(train_dataset.unit_ids, exist_ok=False)
        except ValueError as err:
            print(err)
        model.unit_emb.subset_vocab(train_dataset.unit_ids)

        try:
            model.session_emb.extend_vocab(train_dataset.session_ids, exist_ok=False)
        except ValueError as err:
            print(err)
        model.session_emb.subset_vocab(train_dataset.session_ids)
        log.info(f'Registered new units and sessions, and delete old ones')
    else:
        # pretrain: register units and sessions
        model.unit_emb.initialize_vocab(pretrain_dataset.unit_ids)
        model.session_emb.initialize_vocab(pretrain_dataset.session_ids)


    # sampler and dataloader
    if eval_only:
        # eval_only: create only the test loader
        if cfg.task in ['movie_decoding_one', 'movie_decoding_three', 'scene_decoding', 'stimulus', 'stimulus_binary']:
            test_sampler = DistributedSamplerWrapper(
                RandomFixedWindowSampler(
                    interval_dict=test_dataset.get_sampling_intervals(),
                    window_length=cfg.sequence_length,
                    generator=torch.Generator().manual_seed(cfg.seed + 1),
                )
            )
        else:
            test_sampler = DistributedSamplerWrapper(
                SequentialFixedWindowSampler(
                    interval_dict=test_dataset.get_sampling_intervals(),
                    window_length=cfg.sequence_length,
                    step=cfg.sequence_length / 2,
                )
            )

        test_loader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            collate_fn=collate,
            batch_size=cfg.get("eval_batch_size", cfg.batch_size),
            num_workers=4,
        )
        log.info(f"[eval_only] Test on {len(test_sampler)} samples (all data)")

    elif cfg.pretrain:
        pretrain_sampler = RandomFixedWindowSampler(
            interval_dict=pretrain_dataset.get_sampling_intervals(),
            window_length=cfg.sequence_length,
            generator=torch.Generator().manual_seed(cfg.seed + 1),
        )

        pretrain_loader = DataLoader(
            pretrain_dataset,
            sampler=pretrain_sampler,
            collate_fn=collate,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            # For debugging. we allow the user to set num_workers to 0.
            persistent_workers=True if cfg.num_workers > 0 else False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )

        ## pretrain loader size ##
        total_dataloader_size = 0
        for batch in pretrain_loader:
            dataloader_per_batch_size = get_object_size(batch)
            total_dataloader_size += dataloader_per_batch_size

        log.info(f"Pre-training on {total_dataloader_size/(1024*1024)} MB")
        log.info(f"Pre-training on {len(pretrain_sampler)} samples")
        log.info(f"Pre-training on {len(pretrain_dataset.unit_ids)} units")
        log.info(f"Pre-training on {len(pretrain_dataset.session_ids)} sessions")

    else:
        train_sampler = RandomFixedWindowSampler(
                    interval_dict=train_dataset.get_sampling_intervals(),
                    window_length=cfg.sequence_length,
                    generator=torch.Generator().manual_seed(cfg.seed + 1),
                )


        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            collate_fn=collate,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            # For debugging. we allow the user to set num_workers to 0.
            persistent_workers=True if cfg.num_workers > 0 else False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )


        log.info(f"Training on {len(train_sampler)} samples")
        log.info(f"Training on {len(train_dataset.unit_ids)} units")
        log.info(f"Training on {len(train_dataset.session_ids)} sessions")

        if cfg.task in ['movie_decoding_one', 'movie_decoding_three', 'scene_decoding', 'stimulus', 'stimulus_binary']:
            val_sampler = DistributedSamplerWrapper(
                RandomFixedWindowSampler(
                    interval_dict=val_dataset.get_sampling_intervals(),
                    window_length=cfg.sequence_length,
                    generator=torch.Generator().manual_seed(cfg.seed + 1),
                )
            )

        else:
            val_sampler = DistributedSamplerWrapper(
                SequentialFixedWindowSampler(
                    interval_dict=val_dataset.get_sampling_intervals(),
                    window_length=cfg.sequence_length,
                    step=cfg.sequence_length / 2,
                )
            )

        val_loader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            collate_fn=collate,
            batch_size=cfg.batch_size,
            num_workers=4
        )
        # val_loader = DataLoader(
        #     val_dataset,
        #     sampler=val_sampler,
        #     collate_fn=collate,
        #     batch_size=cfg.batch_size,
        #     num_workers=cfg.num_workers,
        #     drop_last=True,
        #     pin_memory=True,
        #     # For debugging. we allow the user to set num_workers to 0.
        #     persistent_workers=True if cfg.num_workers > 0 else False,
        #     prefetch_factor=2 if cfg.num_workers > 0 else None,
        # )

        log.info(f"Validation on {len(val_sampler)} samples")

        # Test sampler and dataloader
        if cfg.task in ['movie_decoding_one', 'movie_decoding_three', 'scene_decoding']:
            test_sampler = DistributedSamplerWrapper(
                RandomFixedWindowSampler(
                    interval_dict=test_dataset.get_sampling_intervals(),
                    window_length=cfg.sequence_length,
                    generator=torch.Generator().manual_seed(cfg.seed + 1),
                )
            )
        else:
            test_sampler = DistributedSamplerWrapper(
                SequentialFixedWindowSampler(
                    interval_dict=test_dataset.get_sampling_intervals(),
                    window_length=cfg.sequence_length,
                    step=cfg.sequence_length / 2,
                )
            )

        test_loader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            collate_fn=collate,
            batch_size=cfg.get(
                "eval_batch_size", cfg.batch_size
            ),  # Default to training batch size, but allow override in config.
            num_workers=4
        )

        # test_loader = DataLoader(
        #     test_dataset,
        #     sampler=test_sampler,
        #     collate_fn=collate,
        #     batch_size=cfg.batch_size,
        #     num_workers=cfg.num_workers,
        #     drop_last=True,
        #     pin_memory=True,
        #     # For debugging. we allow the user to set num_workers to 0.
        #     persistent_workers=True if cfg.num_workers > 0 else False,
        #     prefetch_factor=2 if cfg.num_workers > 0 else None,
        # )

        log.info(f"Test on {len(test_sampler)} samples")

    if eval_only:
        # ── eval_only mode: skip optimizer/scheduler, go directly to test ──
        # dummy optimizer/scheduler (required by TrainWrapper)
        dummy_optimizer = torch.optim.SGD(
            [torch.nn.Parameter(torch.zeros(1))], lr=1e-8
        )
        dummy_scheduler = torch.optim.lr_scheduler.StepLR(dummy_optimizer, step_size=1)

        wrapper = train_wrapper.TrainWrapper(
            model=model,
            optimizer=dummy_optimizer,
            scheduler=dummy_scheduler,
            use_cre_loss=model.use_cre_loss,
            use_random_masking_loss=model.use_random_masking_loss,
            no_decoding=model.no_decoding,
            task=cfg.task
        )

        tb = lightning.pytorch.loggers.tensorboard.TensorBoardLogger(
            save_dir=cfg.log_dir,
        )
        wandb = lightning.pytorch.loggers.WandbLogger(
            name=cfg.name,
            project=cfg.get("wandb_project", None),
            entity=cfg.get("wandb_entity", None),
            log_model=cfg.get("wandb_log_model", False),
            save_dir=cfg.log_dir,
        )

        callbacks = [
            ModelSummary(max_depth=2),
            train_wrapper.CustomValidator(
                test_loader,
                use_random_masking_loss=model.use_random_masking_loss,
                task=cfg.task,
                name=cfg.name,
                on_test=True,
            ),
        ]

        trainer = lightning.Trainer(
            logger=[tb, wandb],
            default_root_dir=cfg.log_dir,
            max_epochs=1,
            log_every_n_steps=1,
            strategy=("ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto"),
            callbacks=callbacks,
            num_sanity_val_steps=0,
            precision=cfg.precision,
            accelerator="gpu",
            devices=cfg.gpus,
            num_nodes=cfg.nodes,
        )

        for logger in trainer.loggers:
            logger.log_hyperparams(OmegaConf.to_container(cfg))

        log.info("[eval_only] Skipping training -> starting test directly.")
        trainer.test(wrapper, [0])

    else:
        # ── standard train/finetune/pretrain path ──
        # No need to explicitly use DDP with the model, lightning does this for us.
        max_lr = cfg.base_lr * cfg.batch_size

        if cfg.epochs > 0 and cfg.steps == 0:
            epochs = cfg.epochs
        elif cfg.steps > 0 and cfg.epochs == 0:
            epochs = cfg.steps // len(train_loader) + 1
        else:
            raise ValueError("Must specify either epochs or steps")

        print(f"Epochs: {epochs}")

        if cfg.finetune:
            # Optionally freeze parameters for Unit Identification
            if cfg.freeze_perceiver_until_epoch != 0:
                model.freeze_perceiver()

        unit_emb_lr_factor = cfg.get("unit_emb_lr_factor", 1.0)
        unit_emb_params = model.unit_emb.parameters()
        # session_emb_params = model.session_emb.parameters()
        special_emb_params = list(unit_emb_params)  # + list(session_emb_params)

        remaining_params = [
            p
            for n, p in model.named_parameters()
            if "unit_emb" not in n  # and "session_emb" not in n
        ]

        param_groups = [
            {
                "params": special_emb_params,
                "lr": max_lr * unit_emb_lr_factor,
                "weight_decay": cfg.weight_decay,
                "sparse": True,
            },
            {
                "params": remaining_params,
                "lr": max_lr,
                "weight_decay": cfg.weight_decay,
                "sparse": False,
            },
        ]

        print("\n--- Optimizer Parameter Check ---")
        all_param_names = {name for name, param in model.named_parameters() if param.requires_grad}

        optimized_param_ids = set()
        for group in param_groups:
            for param in group['params']:
                optimized_param_ids.add(id(param))

        optimized_param_names = set()
        for name, param in model.named_parameters():
            if id(param) in optimized_param_ids:
                optimized_param_names.add(name)

        missing_params = all_param_names - optimized_param_names

        if missing_params:
            print("Missing from optimizer:")
            for name in sorted(list(missing_params)):
                print(f"  - {name}")
        else:
            print("All trainable params are in optimizer.")
        print("----------------------------------------\n")

        if cfg.get("use_sparse_lamb", False):
            optimizer = SparseLamb(param_groups)
        else:
            optimizer = Lamb(param_groups)

        if cfg.pretrain:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[max_lr * unit_emb_lr_factor, max_lr],
                epochs=epochs,
                steps_per_epoch=len(pretrain_loader),
                pct_start=cfg.pct_start,
                anneal_strategy="cos",
                div_factor=1,
            )

        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[max_lr * unit_emb_lr_factor, max_lr],
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=cfg.pct_start,
                anneal_strategy="cos",
                div_factor=1,
            )

        # Now we create the model wrapper.
        wrapper = train_wrapper.TrainWrapper(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            use_cre_loss=model.use_cre_loss,
            use_random_masking_loss=model.use_random_masking_loss,
            no_decoding=model.no_decoding,
            task=cfg.task
        )

        tb = lightning.pytorch.loggers.tensorboard.TensorBoardLogger(
            save_dir=cfg.log_dir,
        )


        wandb = lightning.pytorch.loggers.WandbLogger(
            name=cfg.name,
            #project=wandb_project,
            project=cfg.get("wandb_project", None),
            entity=cfg.get("wandb_entity", None),
            log_model=cfg.get("wandb_log_model", False),
            save_dir=cfg.log_dir,
        )

        print(f"Wandb ID: {wandb.version}")

        if cfg.pretrain:
            model_ckpt_callback = ModelCheckpoint(
                dirpath=os.path.join(cfg.log_dir, f"lightning_logs/{cfg.name}"),
                save_last=True,
                verbose=True,
                monitor="train_loss",
                mode="min",
                save_on_train_epoch_end=True,
                every_n_epochs=cfg.eval_epochs,
            )

            callbacks = [
                ModelSummary(max_depth=2),
                model_ckpt_callback,
                LearningRateMonitor(logging_interval="step"),
            ]
        else:
            model_ckpt_callback = ModelCheckpoint(
                dirpath=os.path.join(cfg.log_dir, f"lightning_logs/{cfg.name}"),
                save_last=True,
                verbose=True,
                monitor="average_val_metric",
                mode="max",
                save_on_train_epoch_end=False,
                every_n_epochs=cfg.eval_epochs,
            )

            callbacks = [
                ModelSummary(max_depth=2),
                model_ckpt_callback,
                GradientChecker(),
                train_wrapper.CustomValidator(val_loader, use_random_masking_loss=model.use_random_masking_loss, task=cfg.task, name=cfg.name),
                train_wrapper.CustomValidator(test_loader, use_random_masking_loss=model.use_random_masking_loss, task=cfg.task, name=cfg.name, on_test=True),
                LearningRateMonitor(logging_interval="step"),
            ]

        if cfg.finetune:
            if cfg.freeze_perceiver_until_epoch > 0:
                callbacks.append(
                    train_wrapper.UnfreezeAtEpoch(cfg.freeze_perceiver_until_epoch)
                )

        if cfg.get("gradient_rescale", False):
            callbacks.append(UnitEmbeddingGradientRescaling(train_dataset))

        trainer = lightning.Trainer(
            logger=[tb, wandb],
            default_root_dir=cfg.log_dir,
            check_val_every_n_epoch=cfg.eval_epochs,
            max_epochs=epochs,
            log_every_n_steps=1,
            strategy=("ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto"),
            callbacks=callbacks,
            num_sanity_val_steps=0,
            precision=cfg.precision,
            reload_dataloaders_every_n_epochs=2000,
            accelerator="gpu",
            devices=cfg.gpus,
            num_nodes=cfg.nodes,
            gradient_clip_val=10.0
        )

        log.info(
            f"Local rank/node rank/world size/num nodes: {trainer.local_rank}/{trainer.node_rank}/{trainer.world_size}/trainer.num_nodes"
        )

        for logger in trainer.loggers:
            logger.log_hyperparams(OmegaConf.to_container(cfg))

        test_ckpt = cfg.get("test_ckpt", None)
        test_only = test_ckpt is not None
        if not test_only:
            # Training
            if cfg.pretrain:
                trainer.fit(
                    wrapper,
                    pretrain_loader,
                    [0],
                    ckpt_path=cfg.ckpt_path if cfg.continue_learning else None,
                )
            else:
                print("\n--- Model Freeze Status ---")
                frozen_params = []
                trainable_params = []

                for name, param in wrapper.model.named_parameters():
                    if param.requires_grad:
                        trainable_params.append(name)
                    else:
                        frozen_params.append(name)

                if frozen_params:
                    print("Frozen params (requires_grad=False):")
                    for name in frozen_params:
                        print(f"  - {name}")
                else:
                    print("All params are trainable (requires_grad=True).")
                print("-------------------------------------------\n")

                trainer.fit(
                    wrapper,
                    train_loader,
                    [0],
                    ckpt_path=cfg.ckpt_path if cfg.continue_learning else None,
                )

        # Testing
        log.info("Beginning Testing")

        if test_ckpt is None:
            test_ckpt = model_ckpt_callback.best_model_path
            assert len(test_ckpt) > 0, (
                "No best model has been checkpointed yet. "
                "Probably because the validator has not been run."
            )

        model = load_model_from_ckpt(model, test_ckpt)
        log.info(f"Loaded model state dict from {test_ckpt}")

        trainer.test(wrapper, [0])


def load_model_from_ckpt(model, ckpt_path, finetune=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.removeprefix("model.")
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict, strict=False)

    return model


# This loads the config file using Hydra, similar to Flags, but composable.
print("[ENV DEBUG]", {k: os.environ.get(k) for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]})

@hydra.main(
    version_base="1.3",
    config_path="./configs",
    config_name="train_openscope_calcium.yaml",
)
def main(cfg: DictConfig):
    # Train the whole thing.
    # This inner function is unnecessary, but I keep it here to maintain
    # a parallel to the original code.
    run_training(cfg)


if __name__ == "__main__":
    main()
