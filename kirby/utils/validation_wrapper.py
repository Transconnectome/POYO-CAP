"""A module that takes a long trial, chops it up into bite-sized pieces, processes it as
 usual, stitches it back together."""

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid
import logging
import os
from kirby.data.sampler import DistributedSamplerWrapper
from kirby.nn import compute_loss_or_metric
from kirby.taxonomy import Decoder, OutputType, Task
from rich import print as rprint


def move_to_gpu(d, pl_module):
    for k, v in d.items():
        if isinstance(v, dict):
            move_to_gpu(v, pl_module)
        elif isinstance(v, torch.Tensor):
            d[k] = v.to(pl_module.device)


class CustomValidator(Callback):
    def __init__(
        self,
        loader,
        on_test=False,  # True if we are testing, False if we are validating
        prefix=None,  # Prefix text for the metrics
        use_random_masking_loss = True,
        task = 'movie_decoding',
        name = 'experiment_name'
    ):
        super().__init__()
        self.loader = loader
        self.use_random_masking_loss = use_random_masking_loss
        self.task = task
        self.name = name

        self.on_test = on_test
        if prefix is None:
            self.prefix = "test" if on_test else "val"
        else:
            self.prefix = prefix

    def run(self, trainer, pl_module):
        session_timestamp = {}
        session_subtask_index = {}
        session_gt_output = {}
        session_pred_output = {}

        if isinstance(self.loader.sampler, DistributedSamplerWrapper):
            self.loader.sampler.set_params(trainer.world_size, trainer.local_rank)
        output_values_whole = []
        output_latents_whole = []
        pred_images_whole = []
        gt_images_whole = []
        gt_labels_whole = []
        session_ids_whole = []
        for batch in tqdm(
            self.loader,
            desc=f"{self.prefix} @ Epoch {trainer.current_epoch}",
            disable=(trainer.local_rank != 0),
        ):
            if batch is None:
                continue
            absolute_starts = batch.pop("absolute_start")  # (B,)
            session_ids = batch.pop("session_id")  # (B,)
            output_subtask_index = batch.pop("output_subtask_index")

            batch_format = None
            if "input_mask" in batch:
                batch_format = "padded"
            elif "input_seqlen" in batch:
                batch_format = "chained"
            else:
                raise ValueError("Invalid batch format.")

            # move to gpu dict of dicts
            move_to_gpu(batch, pl_module)

            # Autocast is explicitly set based on the precision specified by the user.
            # By default, torch autocasts to float16 for 16-bit inference.
            # This behavior is overridden to use bfloat16 if specified in trainer.precision.
            # If 16-bit inference is not enabled, autocast is not used.
            def get_autocast_args(trainer):
                if trainer.precision.startswith("bf16"):
                    return torch.bfloat16, True
                elif trainer.precision.startswith("16"):
                    return torch.float16, True
                else:
                    return None, False

            dtype, enabled = get_autocast_args(trainer)
            # forward pass
            with torch.cuda.amp.autocast(enabled=enabled, dtype=dtype):
                with torch.inference_mode():
                    # if self.use_random_masking_loss:
                    #     loss = pl_module.model(**batch)
                    # else:
                    #     if self.use_random_masking_loss:
                    #         loss = self.model(**data)
                    #     else:
                    output_values, output_latents, pred_output, loss, losses_taskwise = pl_module.model(**batch)

            
            if self.prefix == 'test':
                # always collect latents (regardless of task)
                output_latents_whole.append(output_latents.cpu())  # save GPU memory

                # stimulus task: collect label + session_id
                if self.task in ['stimulus', 'stimulus_binary']:
                    for sid in session_ids:
                        session_ids_whole.append(sid)
                    if 'FOOD' in output_values:
                        gt_labels_whole.append(output_values['FOOD'].cpu())
                    elif 'FOOD_BINARY' in output_values:
                        gt_labels_whole.append(output_values['FOOD_BINARY'].cpu())

                # Task-specific outputs
                if self.task in ['movie_decoding_one', 'movie_decoding_three', 'scene_decoding']:
                    recon_imgs = pred_output[0]  # (B, 1, 3, 32, 64)
                    gt_imgs = pred_output[1]
                    
                    pred_images_whole.append(recon_imgs.cpu())
                    gt_images_whole.append(gt_imgs.cpu())
                    
                    # Output values
                    if self.task == 'movie_decoding_one':
                        output_values_whole.append(output_values['NATURAL_MOVIE_ONE'].squeeze().cpu())
                    elif self.task == 'movie_decoding_three':
                        output_values_whole.append(output_values['NATURAL_MOVIE_THREE'].squeeze().cpu())
                    elif self.task == 'scene_decoding':
                        output_values_whole.append(output_values['NATURAL_SCENES'].squeeze().cpu())
        
            # log the val_loss
            pl_module.log_dict({f"{self.prefix}_loss": loss})

            # we need to get the timestamps, the ground truth values, the task ids as well
            # as the subtask ids. since the batch is padded and chained, this is a bit tricky
            # tldr: this extracts the ground truth in the same format as the model output
            if self.task in ['movie_decoding_one', 'movie_decoding_three', 'scene_decoding']:
                batch_size = pred_output[0].shape[0]
            else:
                batch_size = len(pred_output)
            # get gt_output and timestamps to be in the same format as pred_output
            timestamps = [{} for _ in range(batch_size)]
            subtask_index = [{} for _ in range(batch_size)]
            gt_output = [{} for _ in range(batch_size)]

            # collect ground truth
            for taskname, spec in pl_module.model.readout.decoder_specs.items():
                if taskname != 'CRE_LINE':
                    taskid = Decoder.from_string(taskname).value

                    # get the mask of tokens that belong to this task
                    mask = batch["output_decoder_index"] == taskid

                    if not torch.any(mask):
                        # there is not a single token for this task, so we skip
                        continue

                    # we need to distribute the outputs to their respective samples

                    if batch_format == "padded":
                        token_batch = torch.where(mask)[0]
                    elif batch_format == "chained":
                        token_batch = batch["output_batch_index"][mask]

                    batch_i, token_batch = torch.unique(token_batch, return_inverse=True)
                    for i in range(len(batch_i)):
                        timestamps[batch_i[i]][taskname] = (
                            batch["output_timestamps"][mask][token_batch == i]
                            + absolute_starts[batch_i[i]]
                        )
                        subtask_index[batch_i[i]][taskname] = output_subtask_index[
                            taskname
                        ][(token_batch == i).detach().cpu()]
                        gt_output[batch_i[i]][taskname] = batch["output_values"][taskname][
                            token_batch == i
                        ]

                    # register all of the data
                    if self.task in ['movie_decoding_one', 'movie_decoding_three', 'scene_decoding']:
                        recon_imgs = pred_output[0] # (batch, 1, 3, 32, 64)
                        gt_imgs = pred_output[1] # (batch, 1, 3, 32, 64)
                        # register all of the data
                        session_id = session_ids[0]
                        if session_id not in session_pred_output:
                            session_pred_output[session_id] = {}
                            session_gt_output[session_id] = {}

                            session_pred_output[session_id][
                                taskname
                            ] = recon_imgs.detach().cpu()
                            if 'movie_decoding' in self.task:
                                session_gt_output[session_id]['NATURAL_MOVIE_RECONSTRUCTION'] = (
                                    gt_imgs.detach().cpu()
                                )
                            elif self.task == 'scene_decoding':
                                session_gt_output[session_id]['NATURAL_SCENE_RECONSTRUCTION'] = (
                                    gt_imgs.detach().cpu()
                                )
                    else:
                        for i in range(batch_size):
                            session_id = session_ids[i]

                            if session_id not in session_pred_output:
                                session_pred_output[session_id] = {}
                                session_gt_output[session_id] = {}
                                session_timestamp[session_id] = {}
                                session_subtask_index[session_id] = {}

                            for taskname, pred_values in pred_output[i].items():
                                if taskname != 'CRE_LINE':
                                    if taskname not in session_pred_output[session_id]:
                                        session_pred_output[session_id][
                                            taskname
                                        ] = pred_values.detach().cpu()
                                        session_gt_output[session_id][taskname] = (
                                            gt_output[i][taskname].detach().cpu()
                                        )
                                        session_timestamp[session_id][taskname] = (
                                            timestamps[i][taskname].detach().cpu()
                                        )
                                        session_subtask_index[session_id][taskname] = (
                                            subtask_index[i][taskname].detach().cpu()
                                        )
                                    else:
                                        session_pred_output[session_id][taskname] = torch.cat(
                                            (
                                                session_pred_output[session_id][taskname],
                                                pred_values.detach().cpu(),
                                            )
                                        )
                                        session_gt_output[session_id][taskname] = torch.cat(
                                            (
                                                session_gt_output[session_id][taskname],
                                                gt_output[i][taskname].detach().cpu(),
                                            )
                                        )
                                        session_timestamp[session_id][taskname] = torch.cat(
                                            (
                                                session_timestamp[session_id][taskname],
                                                timestamps[i][taskname].detach().cpu(),
                                            )
                                        )
                                        session_subtask_index[session_id][taskname] = torch.cat(
                                            (
                                                session_subtask_index[session_id][taskname],
                                                subtask_index[i][taskname].detach().cpu(),
                                            )
                                        )
                                else:
                                    pass
                                        
                else:
                    pass
        


        def gather_concat_dict(obj):
            """All-gather and concatenate dictionary-of-dictionary-of-tensors objects"""
            gathered_objlist = [None] * trainer.world_size
            dist.all_gather_object(gathered_objlist, obj)

            # Concatenate all tensors in the dictionaries
            gathered_obj = defaultdict(lambda: defaultdict(list))
            for objlist in gathered_objlist:
                for outer_key, inner_dict in objlist.items():
                    for inner_key, tensor in inner_dict.items():
                        gathered_obj[outer_key][inner_key].append(tensor)

            # now actually concatenate the tensors in the innermost lists
            for outer_key, inner_dict in gathered_obj.items():
                for inner_key, tensor_list in inner_dict.items():
                    gathered_obj[outer_key][inner_key] = torch.cat(tensor_list, dim=0)

            dist.barrier()
            return gathered_obj

        # Gather
        if trainer.world_size > 1 and not self.use_random_masking_loss:
            if self.task in ['drifting_gratings', 'static_gratings', 'stimulus', 'stimulus_binary']:
                session_timestamp = gather_concat_dict(session_timestamp)
                session_subtask_index = gather_concat_dict(session_subtask_index)

            session_gt_output = gather_concat_dict(session_gt_output)
            session_pred_output = gather_concat_dict(session_pred_output)

        metrics = dict()
        for session_id in tqdm(
            session_gt_output,
            desc=f"Compiling metrics @ Epoch {trainer.current_epoch}",
            disable=(trainer.local_rank != 0),
        ):
            for taskname in session_gt_output[session_id]:
                # print('taskname is', taskname) # NATURAL_MOVIE_RECONSTRUCTION
                if taskname != 'CRE_LINE':
                    decoders = self.loader.dataset.session_info_dict[session_id]["config"][
                        "multitask_readout"
                    ]
                    if self.task in ['drifting_gratings', 'static_gratings', 'stimulus', 'stimulus_binary']:
                        decoder = None
                        for decoder_ in decoders:
                            if decoder_["decoder_id"] == taskname:
                                decoder = decoder_

                        assert decoder is not None, f"Decoder not found for {taskname}"
                        metrics_spec = decoder["metrics"]

                        for metric in metrics_spec:
                            gt = session_gt_output[session_id][taskname]
                            pred = session_pred_output[session_id][taskname]
                            timestamps = session_timestamp[session_id][taskname]
                            subtask_index = session_subtask_index[session_id][taskname]

                            metric_subtask = metric.get("subtask", None)
                            if metric_subtask is not None:
                                select_subtask_index = Task.from_string(metric_subtask).value
                                mask = subtask_index == select_subtask_index
                                gt = gt[mask]
                                pred = pred[mask]
                                timestamps = timestamps[mask]
                        # pool
                        output_type = pl_module.model.readout.decoder_specs[taskname].type

                        if output_type == OutputType.CONTINUOUS:
                            pred = avg_pool(timestamps, pred)
                            gt = avg_pool(timestamps, gt)
                        elif output_type in [
                            OutputType.BINARY,
                            OutputType.MULTINOMIAL,
                            OutputType.MULTILABEL,
                        ]:
                            gt = gt_pool(timestamps, gt)
                            pred = avg_pool(timestamps, pred)

                        # Resolve the appropriate loss function.
                        # metrics[
                        #     f"{self.prefix}_{session_id}_{str(taskname.lower())}_{metric['metric']}"
                        # ] = compute_loss_or_metric(
                        #     metric["metric"], output_type, pred, gt, 1.0, taskname
                        # ).to(pl_module.device) #.item()
                        metrics[
                            f"{self.prefix}_{session_id}_{str(taskname.lower())}_{metric['metric']}"
                        ] = compute_loss_or_metric(
                            metric["metric"], output_type, pred, gt, 1.0, taskname
                        ).item() # this was originally on GPU but moved to CPU unexpectedly


                    else:
                        if self.task == 'movie_decoding_one':
                            gt = session_gt_output[session_id]['NATURAL_MOVIE_RECONSTRUCTION']
                            pred = session_pred_output[session_id]['NATURAL_MOVIE_ONE']
                        elif self.task == 'movie_decoding_three':
                            gt = session_gt_output[session_id]['NATURAL_MOVIE_RECONSTRUCTION']
                            pred = session_pred_output[session_id]['NATURAL_MOVIE_THREE']
                        elif self.task == 'scene_decoding':
                            gt = session_gt_output[session_id]['NATURAL_SCENE_RECONSTRUCTION']
                            pred = session_pred_output[session_id]['NATURAL_SCENES']
                        
                        metrics[f"{self.prefix}_{session_id}_{str(taskname.lower())}_SSIM"] = compute_loss_or_metric(
                                    "SSIM", None, pred, gt.float(), 1.0, taskname
                                ).item()

                    
                else:
                    pass

                    
        # Add average of all metrics
        # TODO: Clean this up so we get average-metric per task-type

        metrics[f"average_{self.prefix}_metric"] = np.array(
                    list(metrics.values())
                ).mean()
        if self.prefix == 'test' and trainer.local_rank == 0:
            save_dir = f'/scratch/x3312a02/POYO-SSL/examples/capoyo/logs/lightning_logs/{self.name}'
            os.makedirs(save_dir, exist_ok=True)
            
            # save latents (always)
            if len(output_latents_whole) > 0:
                output_latents_tensor = torch.cat(output_latents_whole, dim=0)
                latent_path = f'{save_dir}/output_latents.pt'
                torch.save(output_latents_tensor, latent_path)
                print(f"💾 Saved latents: {output_latents_tensor.shape} -> {latent_path}")
            
            # stimulus task: save labels + session IDs
            if self.task in ['stimulus', 'stimulus_binary'] and len(gt_labels_whole) > 0:
                import pickle
                gt_labels_tensor = torch.cat(gt_labels_whole, dim=0)
                torch.save(gt_labels_tensor, f'{save_dir}/gt_labels.pt')
                with open(f'{save_dir}/session_ids.pkl', 'wb') as f:
                    pickle.dump(session_ids_whole, f)
                print(f"💾 Saved gt_labels: {gt_labels_tensor.shape}, session_ids: {len(session_ids_whole)}")

            # task-specific saves
            if self.task in ['movie_decoding_one', 'movie_decoding_three', 'scene_decoding']:
                if len(output_values_whole) > 0:
                    output_values_tensor = torch.cat(output_values_whole, dim=0)
                    torch.save(output_values_tensor, f'{save_dir}/output_values.pt')
                    print(f"💾 Saved output_values: {output_values_tensor.shape}")
                
                if len(pred_images_whole) > 0:
                    pred_images_tensor = torch.cat(pred_images_whole, dim=0)
                    gt_images_tensor = torch.cat(gt_images_whole, dim=0)
                    torch.save(pred_images_tensor, f'{save_dir}/test_pred.pt')
                    torch.save(gt_images_tensor, f'{save_dir}/test_true.pt')
                    print(f"💾 Saved images: pred={pred_images_tensor.shape}, gt={gt_images_tensor.shape}")

        
        # average_tensor = torch.stack(
        #                             list(metrics.values())
        #                         ).mean()
        
        # metrics[f"average_{self.prefix}_metric"] = average_tensor
        # np.array(list(metrics.values())).mean()

        print("\n--- 🕵️ Inspecting metrics dictionary before logging 🕵️ ---")
        is_problem_found = False
        for key, value in metrics.items():
            # if value is a tensor, check its device
            if isinstance(value, torch.Tensor):
                print(f"  - Key: {key}, Type: {type(value)}, Device: {value.device}")
                if value.device.type == 'cpu':
                    print(f"  - 🔴 FOUND IT! This tensor is on the CPU!")
                    is_problem_found = True
            # otherwise (Python/Numpy number etc. -> candidate to be converted to CPU tensor)
            else:
                print(f"  - Key: {key}, Type: {type(value)}, Value: {value}")
                print(f"  - 🔴 POTENTIAL PROBLEM! This is not a tensor. It will be converted to a CPU tensor.")
                is_problem_found = True

        if not is_problem_found:
            print("  - ✅ All tensors seem to be on the correct GPU device.")
        print("----------------------------------------------------------\n")

        pl_module.log_dict(metrics, on_epoch=True) #, sync_dist=True)
        logging.info(f"Logged {len(metrics)} {self.prefix} metrics.")

        metrics_data = []
        for metric_name, metric_value in metrics.items():
            metrics_data.append({"metric": metric_name, "value": metric_value})

        metrics_df = pd.DataFrame(metrics_data)
        if trainer.local_rank == 0:
            if self.task in ['movie_decoding_one', 'movie_decoding_three', 'scene_decoding']:
                if pl_module.tb is not None:
                    pl_module.tb.add_text(
                        f"{self.prefix}_metrics", metrics_df.to_markdown()
                    )
                if pl_module.wandb is not None:
                    pl_module.wandb.log(
                        {f"{self.prefix}_metrics": wandb.Table(dataframe=metrics_df)}
                    )
                    recon_imgs = recon_imgs.squeeze(1) / 255.0  # Remove unnecessary dimensions
                    recon_imgs = recon_imgs.cpu().detach()
                    grid = make_grid(recon_imgs, nrow=int(recon_imgs.shape[0]**0.5), normalize=False) # arrange 4 images in a 2x2 grid

                    gt_imgs = gt_imgs.squeeze(1) / 255.0  # Remove unnecessary dimensions
                    gt_imgs = gt_imgs.cpu().detach()
                    grid_gt = make_grid(gt_imgs, nrow=int(gt_imgs.shape[0]**0.5), normalize=False) # arrange 4 images in a 2x2 grid

                    pl_module.wandb.log({f"{self.prefix}_pred_images": wandb.Image(grid, caption="predicted images")})
                    pl_module.wandb.log({f"{self.prefix}_true_images": wandb.Image(grid_gt, caption="true images")})
                    if self.prefix == 'test':
                        output_latents_whole.append(output_latents)
                        pred_images_whole.append(recon_imgs)
                        gt_images_whole.append(gt_imgs)

                        if self.task == 'movie_decoding_one':
                            output_values_whole.append(output_values['NATURAL_MOVIE_ONE'].squeeze())
                        elif self.task == 'movie_decoding_three':
                            output_values_whole.append(output_values['NATURAL_MOVIE_THREE'].squeeze())
                        elif self.task == 'scene_decoding':
                            output_values_whole.append(output_values['NATURAL_SCENES'].squeeze())
                        
            
            elif self.task in ['drifting_gratings', 'static_gratings', 'stimulus', 'stimulus_binary']:
                if pl_module.tb is not None:
                    pl_module.tb.add_text(
                        f"{self.prefix}_metrics", metrics_df.to_markdown()
                    )
                if pl_module.wandb is not None:
                    if self.prefix == 'test' and output_latents_whole:
                        output_latents_whole.append(output_latents)
                        output_values_whole.append(output_values)
        '''
        if self.prefix == 'test':
            output_latents_tensor = torch.cat(output_latents_whole, dim=0)
            torch.save(output_latents_tensor, f'/scratch/x3312a02/POYO-SSL/examples/capoyo/logs/lightning_logs/{self.name}/output_latents.pt')
            if self.task in ['movie_decoding_one', 'movie_decoding_three', 'scene_decoding']:
                output_values_tensor = torch.cat(output_values_whole, dim=0)
                torch.save(output_values_tensor, f'/scratch/x3312a02/POYO-SSL/examples/capoyo/logs/lightning_logs/{self.name}/output_values.pt')

            if self.task in ['movie_decoding_one', 'movie_decoding_three', 'scene_decoding']:
                pred_images_tensor = torch.cat(pred_images_whole, dim=0)
                gt_images_tensor = torch.cat(gt_images_whole, dim=0) 
                torch.save(pred_images_tensor, f'/scratch/x3312a02/POYO-SSL/examples/capoyo/logs/lightning_logs/{self.name}/test_pred.pt')
                torch.save(gt_images_tensor, f'/scratch/x3312a02/POYO-SSL/examples/capoyo/logs/lightning_logs/{self.name}/test_true.pt')
        '''


        rprint(metrics_df)

        
        return metrics_df

    def on_validation_epoch_start(self, trainer, pl_module):
        if not self.on_test:
            return self.run(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        if self.on_test:
            return self.run(trainer, pl_module)


def avg_pool(timestamps: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    r"""This function performs pooling operations (mean or mode) on a tensor based on
    unique timestamps and the datatype of the values.

    Args:
        timestamps (torch.Tensor): A 1D tensor containing timestamps.
        values (torch.Tensor): A tensor of values that correspond to the timestamps. It
            expects a tensor of shape (N, ...), where N is the number of timestamps.

    Returns:
        torch.Tensor: A tensor with the pooled values for each unique timestamp. If the
          values are continuous, the function performs mean pooling, averaging the
          values for each unique timestamp. If the values are categorical (labels),
          the function returns the mode of the values for each unique timestamp.

    Note:
        For mean pooling, this function leverages `torch.scatter_add_` to efficiently
        aggregate values for each unique timestamp
    """
    # Find unique timestamps and their inverse indices
    unique_timestamps, indices = torch.unique(
        timestamps, return_inverse=True, sorted=True
    )

    # Prepare a tensor for summing values for each unique timestamp
    pooled_sum = torch.zeros(
        (len(unique_timestamps), *values.shape[1:]),
        device=values.device,
        dtype=values.dtype,
    )

    # Use mode for integers
    if values.dtype == torch.long:
        # NOT IDEAL, IT IS FASTER TO AVERAGE THE LOGITS THAN TO PERFORM A VOTE
        mode_values = torch.zeros_like(pooled_sum)
        for i, timestamp in enumerate(unique_timestamps):
            mask = timestamps == timestamp
            group_values = values[mask]
            mode, _ = torch.mode(group_values, dim=0)
            mode_values[i] = mode
        return mode_values

    # Count occurrences of each unique timestamp
    counts = torch.zeros(
        len(unique_timestamps), device=timestamps.device, dtype=values.dtype
    )
    counts = counts.scatter_add_(
        0, indices, torch.ones_like(indices, dtype=values.dtype)
    )
    # Accumulate values for each unique timestamp
    indices_expanded = indices.unsqueeze(-1).expand_as(values)
    pooled_sum.scatter_add_(0, indices_expanded, values)
    # Calculate the average
    epsilon = 1e-8  # small constant to prevent division by zero
    averages = torch.div(pooled_sum, counts.unsqueeze(-1) + epsilon)

    return averages


def gt_pool(timestamps: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    r"""Wrapper over `avg_pool` specifically for pooling ground truth categorical
    values.
    """
    return (
        torch.round(avg_pool(timestamps, values.float().view(-1, 1))).long().squeeze()
    )
