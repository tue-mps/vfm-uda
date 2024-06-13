# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d

from models.token_masking import TokenMasking


class VITLightweightHead(nn.Module):
    def __init__(
            self,
            img_size: int,
            model_name: str,
            num_classes: int = None,
            patch_size: int = 14,
            align_corners: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.align_corners = align_corners

        self.vit_in_img_size = int((img_size * 1.1) // self.patch_size * self.patch_size)

        self.encoder = torch.hub.load('facebookresearch/dinov2', model_name)

        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(self.encoder.embed_dim,
                               self.encoder.embed_dim // 2,
                               kernel_size=2, stride=2, padding=0, bias=False
                               ),
            LayerNorm2d(self.encoder.embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.encoder.embed_dim // 2,
                               self.encoder.embed_dim // 4,
                               kernel_size=2, stride=2, padding=0, bias=False
                               ),
            LayerNorm2d(self.encoder.embed_dim // 4),
            nn.Conv2d(
                self.encoder.embed_dim // 4,
                self.encoder.embed_dim // 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.encoder.embed_dim // 4),
        )
        self.token_masking = TokenMasking(self.encoder.mask_token)

        out = nn.Conv2d(self.encoder.embed_dim // 4, num_classes, kernel_size=1, padding=0, bias=False)
        torch.nn.init.normal_(out.weight, 0, std=0.1)
        self.out = out

        self.param_defs_decoder = [
            ("out", self.out),
            ("upscale", self.upscale)
        ]

        self.param_defs_encoder_blocks = [
            ("encoder.blocks", self.encoder.blocks),
        ]

        self.param_defs_encoder_stems = [
            ("encoder.mask_token", self.encoder.mask_token),
            ("encoder.norm", self.encoder.norm),
            ("encoder.pos_embed", self.encoder.pos_embed),
            ("encoder.patch_embed.proj", self.encoder.patch_embed.proj),
            ("encoder.cls_token", self.encoder.cls_token)
            if hasattr(self.encoder, "cls_token")
            else (None, None),
        ]

        self.encoder_depth = len(self.encoder.blocks)

    def forward_features(self, img: torch.Tensor, mask_ratio: float):
        b, c, h, w = img.shape
        token_img_shape = (b, self.encoder.embed_dim, h // self.patch_size, w // self.patch_size)

        x_patch = self.encoder.patch_embed(img)
        if 0.0 < mask_ratio:
            x_patch = self.token_masking(x_patch, mask_ratio)
        x = torch.cat((self.encoder.cls_token.expand(x_patch.shape[0], -1, -1), x_patch), dim=1)
        x = x + self.encoder.interpolate_pos_encoding(x, w, h)
        x = x.contiguous()
        for i in range(self.encoder_depth):
            x = self.encoder.blocks[i](x)
        x = self.encoder.norm(x)
        x = self.token_to_image(x, token_img_shape)
        return x

    def forward(self, img, mask_ratio=0.0, return_features=False):
        b, c, h, w = img.shape
        assert h == w
        orig_img_size = [h, w]
        img = F.interpolate(
            img,
            size=(self.vit_in_img_size, self.vit_in_img_size),
            mode="bilinear", align_corners=self.align_corners
        )
        feats = self.forward_features(img, mask_ratio)
        logit = self.out(self.upscale(feats))
        logit = F.interpolate(logit, orig_img_size, mode="bilinear", align_corners=self.align_corners)
        if return_features:
            return logit, feats
        else:
            return logit

    @torch.no_grad()
    def inference(self, img):
        return self.forward(img)

    def token_to_image(self, x, shape, remove_class_token=True):
        if remove_class_token:
            x = x[:, 1:]
        x = x.permute(0, 2, 1)
        x = x.view(shape).contiguous()
        return x
