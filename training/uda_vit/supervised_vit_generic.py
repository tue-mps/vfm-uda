# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------


import io
import os
from logging import info
from typing import Optional, Tuple

import lightning
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.transforms import functional as FV
from torchvision.transforms import v2 as TV

from datasets.utils.mappings import get_label2name
from datasets.utils.util import colorize_mask, normalize
from models.warmup_and_linear_scheduler import WarmupAndLinearScheduler
from training.utils.inference_collection import slide_inference_pre_cropped
from training.utils.utils import get_full_names, get_param_group, process_parameters


class SupervisedVITGeneric(lightning.LightningModule):
    def __init__(
            self,
            batch_size: int,
            img_size: int,
            network: nn.Module,
            lr: float = 5e-5,
            lr_multiplier: float = 0.01,
            layerwise_lr_decay: float = 1.0,
            warmup_iters: int = 1500,
            weight_decay: float = 0.01,
            ignore_index: int = 255,
            ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.job_id = os.environ.get('SLURM_JOB_ID')
        self.batch_size = batch_size
        self.img_size = img_size
        self.lr = lr
        self.lr_multiplier = lr_multiplier
        self.layerwise_lr_decay = layerwise_lr_decay
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
        self.num_classes = network.num_classes
        self.warmup_iters = warmup_iters

        self.brightness_delta = 32 / 255.
        self.contrast_delta = 0.5
        self.saturation_delta = 0.5
        self.hue_delta = 18 / 360.
        self.color_jitter_probability = 0.8

        self.save_hyperparameters()

        #
        random_aug_weak = TV.Compose([
            TV.RandomApply([
                TV.ColorJitter(
                    brightness=self.brightness_delta,
                    contrast=self.contrast_delta,
                    saturation=self.saturation_delta,
                    hue=self.hue_delta)], p=0.9),
        ])
        self.random_aug_weak = TV.Lambda(lambda x: torch.stack([random_aug_weak(x_) for x_ in x]))

        self.network = network

        self.label2name = get_label2name()
        self.val_ds_names = ["cityscapes", "wilddash"]
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    validate_args=False,
                    ignore_index=ignore_index,
                    average=None,
                )
                for _ in range(len(self.val_ds_names))
            ]
        )

        self._load_ckpt(ckpt_path)
        self.automatic_optimization = False

    @torch.no_grad()
    def train_dataprep(self, batch):
        sourceds_image, sourceds_target, sourceds_ignore_mask = batch
        batch_size, _, H, W = sourceds_image.shape

        if torch.rand(1) < 0.5:
            sourceds_image = FV.hflip(sourceds_image)
            sourceds_target = FV.hflip(sourceds_target)

        sourceds_color_aug_image = self.random_aug_weak(sourceds_image.detach().clone().float() / 255.)

        return sourceds_image, sourceds_color_aug_image, sourceds_target

    def get_optimizers(self):
        opt = self.optimizers()
        opt.zero_grad()
        return opt

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
    ):
        opt = self.get_optimizers()

        ##
        # data prep
        ##
        sourceds_image, sourceds_color_aug_image, sourceds_target = self.train_dataprep(batch)

        ##
        # supervised training
        ##
        source_logits = self.network(normalize(sourceds_color_aug_image))
        loss_source = F.cross_entropy(source_logits, sourceds_target, ignore_index=self.ignore_index)
        self.manual_backward(loss_source)
        loss_source = loss_source.detach()

        opt.step()
        self.lr_schedulers().step()

        self.log("train_loss_source", loss_source, prog_bar=True)

        with torch.no_grad():
            accept_mask = sourceds_target != self.ignore_index
            sourceds_predicted_segmentation = torch.argmax(source_logits.detach(), dim=1)
            acc_source = (sourceds_predicted_segmentation == sourceds_target)
            acc_source = (acc_source * accept_mask).sum() / accept_mask.sum()
            self.log("train_acc_source", acc_source, prog_bar=True)

            if (self.global_step % 100) == 0:
                self._log_train(
                    sourceds_color_aug_image, sourceds_target,
                    sourceds_predicted_segmentation
                )

    @torch.no_grad()
    def _log_train(
            self,
            sourceds_image, sourceds_target,
            predicted_source_student,
    ):
        sourceds_image = (sourceds_image[0]).cpu().permute(1, 2, 0).float().numpy()
        color_sourceds_target = colorize_mask(sourceds_target[0].cpu().long().numpy())

        color_predicted_source_student = colorize_mask(predicted_source_student[0].cpu().long().numpy())

        fig, axes = plt.subplots(
            1, 3, figsize=(int(sourceds_image.shape[1] / 100 * 3),
                           int(sourceds_image.shape[0] / 100 * 1))
        )

        axes[0].imshow(sourceds_image)
        axes[0].axis("off")
        axes[1].imshow(color_sourceds_target)
        axes[1].axis("off")
        axes[2].imshow(color_predicted_source_student)
        axes[2].axis("off")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(
            buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black"
        )
        plt.close(fig)

        buf.seek(0)
        concatenated_image = Image.open(buf).convert('RGB')
        self.trainer.logger.experiment.log(  # type: ignore
            {
                f"train_debug": [
                    wandb.Image(concatenated_image)
                ]
            }
        )

    @torch.no_grad()
    def _log_pred(self, image, prediction, target, dataloader_idx, pred_idx, log_prefix):
        color_ground_truth_mask = colorize_mask(target)
        color_predicted_mask = colorize_mask(prediction)

        fig, axes = plt.subplots(
            1, 3, figsize=(int(target.shape[1] / 100 * 3), target.shape[0] / 100)
        )

        axes[0].imshow(image)
        axes[0].axis("off")
        axes[1].imshow(color_predicted_mask)
        axes[1].axis("off")
        axes[2].imshow(color_ground_truth_mask)
        axes[2].axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(
            buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black"
        )
        plt.close(fig)

        buf.seek(0)
        concatenated_image = Image.open(buf)
        ds_name = self.val_ds_names[dataloader_idx]
        self.trainer.logger.experiment.log(  # type: ignore
            {
                f"{log_prefix}_{ds_name}_pred_{pred_idx}": [
                    wandb.Image(concatenated_image)
                ]
            }
        )

    def eval_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int,
            log_prefix: str,
    ):
        b_image, b_crop, b_preds, b_count_mat, b_coords, b_target = batch

        all_segmentation = []
        for crop, preds, count_mat, coords, target in zip(b_crop, b_preds, b_count_mat, b_coords, b_target):
            crop = normalize(crop.float() / 255.0)
            segmentation = slide_inference_pre_cropped(self.network, crop, preds, count_mat, coords, self.num_classes)
            segmentation = torch.argmax(segmentation, dim=0)
            all_segmentation += [segmentation]

        for segmentation, target in zip(all_segmentation, b_target):
            self.metrics[dataloader_idx].update(
                segmentation.unsqueeze(0), target.unsqueeze(0)
            )

        if batch_idx == 0:
            for i, (img, segmentation, target) in enumerate(zip(b_image, all_segmentation, b_target)):

                if i < 4:  # limit images for viz
                    pred_idx = batch_idx * self.batch_size + i
                    self._log_pred(
                        img.cpu().permute(1, 2, 0).numpy(),
                        segmentation.cpu().numpy(),
                        target.cpu().numpy(),
                        dataloader_idx,
                        pred_idx,
                        log_prefix,
                    )

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        return self.eval_step(batch, batch_idx, dataloader_idx, "val")

    def test_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        return self.eval_step(batch, batch_idx, dataloader_idx, "test")

    def _on_eval_epoch_end(self, log_prefix):
        miou_per_dataset = []
        iou_per_dataset_per_class = []
        for metric_idx, metric in enumerate(self.metrics):
            iou_per_dataset_per_class.append(metric.compute())
            metric.reset()
            ds_name = self.val_ds_names[metric_idx]

            for iou_idx, iou in enumerate(iou_per_dataset_per_class[-1]):
                label_name = self.label2name[iou_idx]
                self.log(
                    f"{log_prefix}_{ds_name}_iou_{label_name}", iou, sync_dist=True
                )

            miou_per_dataset.append(float(iou_per_dataset_per_class[-1].mean()))
            self.log(
                f"{log_prefix}_{ds_name}_miou", miou_per_dataset[-1], sync_dist=True
            )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end("test")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"learning_rate/group_{i}", param_group["lr"], on_step=True)

    def configure_optimizers(self):
        current_params = {
            name
            for name, param in self.network.named_parameters()
            if param.requires_grad
        }

        lr = (
                self.lr
                * math.sqrt(self.batch_size * self.trainer.num_devices * self.trainer.num_nodes)
        )

        param_defs, current_params = process_parameters(
            self.network.param_defs_decoder, current_params
        )
        param_groups = [get_param_group(param_defs, lr)]

        lr *= self.lr_multiplier

        n_blocks = max(
            len(blocks) for _, blocks in self.network.param_defs_encoder_blocks
        )

        for i in range(n_blocks - 1, -1, -1):
            for block_name_prefix, blocks in self.network.param_defs_encoder_blocks:
                if i < len(blocks):
                    block_params = blocks[i].parameters()
                    block_param_names = get_full_names(
                        blocks[i], f"{block_name_prefix}.{i}"
                    )
                    current_params -= block_param_names
                    param_groups.append(get_param_group(block_params, lr))

            lr *= self.layerwise_lr_decay

        param_defs, current_params = process_parameters(
            self.network.param_defs_encoder_stems, current_params
        )
        param_groups.append(get_param_group(param_defs, lr))

        if current_params:
            raise ValueError(
                f"The following parameters are not included in the optimizer: {current_params}"
            )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        lr_scheduler = {
            "scheduler": WarmupAndLinearScheduler(
                optimizer,
                start_warmup_lr=1e-3,
                warmup_iters=self.warmup_iters,
                base_lr=1,
                final_lr=0,
                total_iters=self.trainer.max_steps,
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def _load_ckpt(self, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return

        ckpt_state: dict = torch.load(ckpt_path, map_location=self.device)

        if "state_dict" in ckpt_state:
            ckpt_state = ckpt_state["state_dict"]

        if "model" in ckpt_state:
            ckpt_state = ckpt_state["model"]

        model_state = self.state_dict()
        skipped_keys = []
        for k in ckpt_state:
            if (k in model_state) and (ckpt_state[k].shape == model_state[k].shape):
                model_state[k] = ckpt_state[k]
            else:
                skipped_keys.append(k)

        info(f"Skipped loading keys: {skipped_keys}")

        self.load_state_dict(model_state)
