import time
import subprocess
import logging
from typing import Optional

import torch
import torch.nn as nn
from lightning import LightningModule
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.callbacks import Callback

from kirby.utils.validation_wrapper import CustomValidator

log = logging.getLogger(__name__)


class TrainWrapper(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        use_cre_loss = True,
        use_random_masking_loss = True,
        no_decoding = True,
        task = 'movie_decoding'
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["optimizer", "scheduler"])
        self.use_cre_loss = use_cre_loss
        self.use_random_masking_loss = use_random_masking_loss
        self.no_decoding = no_decoding
        self.task = task
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tb = None
        self.wandb = None

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

    def setup(self, stage=None):
        # Make specific loggers available.
        for logger in self.loggers:
            if isinstance(logger, pl_loggers.WandbLogger):
                self.wandb = logger.experiment
            elif isinstance(logger, pl_loggers.TensorBoardLogger):
                self.tb = logger.experiment

    def on_train_start(self):
        # Log the output of `cat /proc/meminfo` using a shell script.
        try:
            # Execute the command and capture its output
            result = subprocess.run(
                ["cat", "/proc/meminfo"],
                capture_output=True,
                text=True,
                check=True,
            )
            result = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
            result = ""

        # Log the output
        logging.info(f"Memory info: \n{result}")

    def on_train_epoch_start(self):
        self.epoch_time = time.time()

    def training_step(self, data, data_idx):
        

        if self.use_cre_loss:
            ## cre-line
            max_values, _ = torch.max(data['unit_cre_line'], dim=1)
            all_zeros = (max_values == 0)
            unique_cre_lines = torch.where(all_zeros, torch.zeros_like(max_values), max_values) # shape is (128,)
            data['output_values']['CRE_LINE'] = unique_cre_lines
        
        # output_latents, output, loss, taskwise_loss = self.model(**data)
        _, _, output, loss, taskwise_loss = self.model(**data)
        '''
        print('shape of output is:', len(output)) # 128
        # every single element has same key - "DRIFTING_GRATINGS"
        print('key of output[127] is:', output[127].keys()) # "DRIFTING_GRATINGS"
        print('value of output[127] is:', output[127]['DRIFTING_GRATINGS']) # eight-length vector like:
        # tensor([[-0.0908,  0.1493,  0.1451,  0.0615,  0.3907,  0.0552,  0.0028, -0.1028]
        print('key of data is:', data.keys())
        # dict_keys(['unit_index', 'timestamps', 'patches', 'token_type', 'input_mask', 'latent_index', 'latent_timestamps', 'unit_spatial_emb', 'unit_feats', 'unit_cre_line', 'session_index', 'output_timestamps', 'output_decoder_index', 'output_values', 'output_weights'])
        '''        
        # don't use this output for cre-line loss
        # output_latents : torch.Size([128, 1, 64])
        # print(data['unit_cre_line'].shape) # torch.Size([128, 6848])
        # print('unit cre line is: ', data['unit_cre_line']) # 0, 1, ..., 12 (integer) - it is cre number
        # each 128 batch has its cre-line.
        # Compute the mean and std of the output.
        
        if not self.no_decoding and self.task in ['drifting_gratings', 'static_gratings']:
            for name in data["output_values"].keys():
                output_predictions = torch.cat(
                    [pred[name] for pred in output if name in pred], dim=0
                )
                self.log(
                    f"predictions/mean_{name}", output_predictions.mean(), prog_bar=False
                )
                self.log(
                    f"predictions/std_{name}", output_predictions.std(), prog_bar=False
                )
                self.log(
                    f"targets/mean_{name}",
                    data["output_values"][name].to(torch.float).mean(),
                    prog_bar=False,
                )
                self.log(
                    f"targets/std_{name}",
                    data["output_values"][name].to(torch.float).std(),
                    prog_bar=False,
                )

            if "unit_index" in data:
                s = data["unit_index"].to(torch.float)
                self.log("inputs/mean_unit_index", s.mean(), prog_bar=False)
                self.log("inputs/std_unit_index", s.std(), prog_bar=False)

        self.log("train_loss", loss, prog_bar=True)
        self.log_dict({f"losses/{k}": v for k, v in taskwise_loss.items()})
        
            
        return {"loss": loss}

    def on_train_epoch_end(self):
        for tag, value in self.model.named_parameters():
            # self.log(f"weights/mean_{tag}", value.mean(), sync_dist=True)
            # self.log(f"weights/std_{tag}", value.std(), sync_dist=True)
            # if value.grad is not None:
            #     self.log(
            #         f"grads/mean_{tag}",
            #         value.grad.mean(),
            #         sync_dist=True,
            #     )

            self.log(f"weights/mean_{tag}", value.cpu().mean(), sync_dist=True)
            self.log(f"weights/std_{tag}", value.cpu().std(), sync_dist=True)
            if value.grad is not None:
                self.log(
                    f"grads/mean_{tag}",
                    value.grad.cpu().mean(),
                    sync_dist=True,
                )

        self.log("epoch_time", time.time() - self.epoch_time)

    def validation_step(self, data, data_idx):
        # Necessary to trick PyTorch Lightning into running the custom validator.
        pass

    def test_step(self, data, data_idx):
        # Necessary to trick PyTorch Lightning into running the custom validator.
        pass

class UnfreezeAtEpoch(Callback):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self._unfreeze_at_epoch:
            log.info(f"Reached epoch {trainer.current_epoch}, unfreezing entire model")
            for module in pl_module.model.children():
                for param in module.parameters():
                    param.requires_grad = True
