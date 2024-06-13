# Obtained from: https://github.com/lhoyer/MIC/blob/master/seg/mmseg/models/utils/masking_transforms.py
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
from torch.nn import functional as F


class BlockMasking:

    def __init__(self, mask_block_size):
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs, mask_ratio):
        B, _, H, W = imgs.shape

        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > mask_ratio).float()
        input_mask = F.interpolate(input_mask, size=(H, W), mode="nearest")
        return input_mask

    @torch.no_grad()
    def __call__(self, imgs, mask_ratio):
        input_mask = self.generate_mask(imgs, mask_ratio)
        return imgs * input_mask
