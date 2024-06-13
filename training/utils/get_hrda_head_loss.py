# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------

import torch.nn.functional as F
from torchvision.transforms import functional as FV


def get_hrda_head_loss(
        fused_logit, higres_cropped_logit, label, crop_params, hr_loss_weight,
        label_weight=None, ignore_index=255
):
    if label_weight is None:
        label_weight = 1.
        label_weight_cropped = 1.
    else:
        label_weight_cropped = FV.crop(label_weight, *crop_params)

    loss_fused = F.cross_entropy(fused_logit, label, ignore_index=ignore_index, reduction='none')
    loss_fused = (loss_fused * label_weight).mean()

    loss_highres_crop = F.cross_entropy(
        higres_cropped_logit, FV.crop(label, *crop_params), ignore_index=ignore_index, reduction='none')
    loss_highres_crop = (loss_highres_crop * label_weight_cropped).mean()

    loss = loss_fused * (1. - hr_loss_weight) + loss_highres_crop * hr_loss_weight
    return loss
