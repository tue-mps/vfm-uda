# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------


import torch
from torch import nn


class TokenMasking(nn.Module):
    def __init__(self, mask_token):
        super().__init__()
        self.mask_token = mask_token

    def forward(self, x, mask_ratio):
        if self.training:
            masks = self.get_random_token_mask_idx(x, mask_ratio)
            mask_token = self.mask_token.to(x.dtype)[0]
            x[masks] = mask_token
            x = x.contiguous()
            return x
        else:
            return x

    def get_random_token_mask_idx(self, x: torch.Tensor, mask_ratio: float):
        B, L, C = x.shape
        token_mask = torch.rand((B, L), device=x.device)
        token_mask = mask_ratio > token_mask
        return token_mask
