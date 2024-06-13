# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------


import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as TV
from torchvision.transforms import functional as FV

from models.daformer.aspp_head import ASPPWrapper
from models.daformer.conv_module import ConvModule
from models.token_masking import TokenMasking
from models.vit_variants.get_timm_vit import get_timm_vit


class VITDaFormerHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.channels = 256
        self.dropout_ratio = 0.1
        self.align_corners = False
        self.fuse_layer = ASPPWrapper(
            in_channels=self.channels,
            channels=self.channels,
            align_corners=self.align_corners,
            sep=True,
            dilations=(1, 6, 12, 18),
            pool=False,
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='LN')
        )
        self.conv_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)
        self.conv_seg.weight.data.normal_(mean=0, std=0.01)
        self.conv_seg.bias.data.zero_()
        self.dropout = nn.Dropout2d(self.dropout_ratio)

        self.upscale = nn.Sequential(
            nn.Conv2d(embed_dim, self.channels, kernel_size=1, padding=0, bias=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=self.align_corners)
        )

    def forward(self, x):
        x = self.upscale(x)
        x = self.fuse_layer(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x


class VITDaFormerAttnHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.channels = 256
        self.dropout_ratio = 0.1
        self.align_corners = False
        self.fuse_layer = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            padding=0,
            kernel_size=1,
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.conv_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(self.dropout_ratio)

        self.upscale = nn.Sequential(
            nn.Conv2d(embed_dim, self.channels, kernel_size=1, padding=0, bias=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=self.align_corners)
        )

    def forward(self, x):
        x = self.upscale(x)
        x = self.fuse_layer(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x


class VITHRDAExactTimm(nn.Module):
    def __init__(
            self,
            img_size: int,
            model_name: str,
            num_classes: int = None,
            patch_size: int = 14,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.vit_in_img_size = int(math.ceil(float(img_size) / self.patch_size) * self.patch_size)
        assert self.vit_in_img_size % 2 == 0, "self.vit_in_img_size: {}".format(self.vit_in_img_size)
        self.lr_scale = 0.5
        self.hr_crop_size = int(self.vit_in_img_size * self.lr_scale)
        self.align_corners = False

        self.encoder = get_timm_vit(model_name, self.hr_crop_size, patch_size)
        self.seg_head = VITDaFormerHead(self.encoder.embed_dim, num_classes)
        self.attn_head = VITDaFormerAttnHead(self.encoder.embed_dim, num_classes)

        if hasattr(self.encoder, "mask_token"):
            self.mask_token = self.encoder.mask_token
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, self.encoder.embed_dim))

        self.token_masking = TokenMasking(self.mask_token)

        self.param_defs_decoder = [
            ("seg_head", self.seg_head),
            ("attn_head", self.attn_head),
        ]

        self.param_defs_encoder_blocks = [
            ("encoder.blocks", self.encoder.blocks),
        ]

        self.param_defs_encoder_stems = [
            ("mask_token", self.mask_token),
            ("encoder.norm", self.encoder.norm),
            ("encoder.pos_embed", self.encoder.pos_embed),
            ("encoder.patch_embed.proj", self.encoder.patch_embed.proj),
            ("encoder.cls_token", self.encoder.cls_token)
            if hasattr(self.encoder, "cls_token")
            else (None, None),
        ]

        self.encoder_depth = len(self.encoder.blocks)

    def forward_encoder(self, img: torch.Tensor, mask_ratio: float = 0.0):
        b, c, h, w = img.shape
        token_img_shape = (b, self.encoder.embed_dim, h // self.patch_size, w // self.patch_size)

        x_patch = self.encoder.patch_embed(img)
        if 0.0 < mask_ratio:
            x_patch = self.token_masking(x_patch, mask_ratio)
        x = self.encoder._pos_embed(x_patch)
        x = self.encoder.norm_pre(x)
        x = x.contiguous()
        for i in range(self.encoder_depth):
            x = self.encoder.blocks[i](x)
        x = self.encoder.norm(x)
        x = self.token_to_image(x, token_img_shape)
        return x

    def token_to_image(self, x, shape, remove_class_token=True):
        if remove_class_token:
            x = x[:, 1:]
        x = x.permute(0, 2, 1)
        x = x.view(shape).contiguous()
        return x

    def forward(self, img, mask_ratio: float = 0.0):
        bs, c, h_orig, w_orig = img.shape
        orig_scaled_res = (int(h_orig * self.lr_scale), int(w_orig * self.lr_scale))

        # get cropping params
        crop_params = TV.RandomCrop.get_params(img, (orig_scaled_res[0], orig_scaled_res[1]))
        cropy1 = crop_params[0]
        cropx1 = crop_params[1]
        cropy2 = crop_params[0] + orig_scaled_res[0]
        cropx2 = crop_params[1] + orig_scaled_res[1]

        # lowres, whole image
        img_low = F.interpolate(img, size=(self.hr_crop_size, self.hr_crop_size), mode="bilinear",
                                align_corners=self.align_corners)
        embed_low = self.forward_encoder(img_low, mask_ratio)
        seg_low = self.seg_head(embed_low)
        attn = torch.sigmoid(self.attn_head(embed_low))

        # highres, cropped image
        img_high = FV.crop(img, *crop_params)
        img_high = F.interpolate(img_high, size=(self.hr_crop_size, self.hr_crop_size), mode="bilinear",
                                 align_corners=self.align_corners)
        seg_high = self.seg_head(self.forward_encoder(img_high, mask_ratio))

        # get orig size
        # dinov2 has a 14 patch size, so we need to do all this resizing tricks...
        seg_low = F.interpolate(seg_low, size=(h_orig, w_orig), mode="bilinear", align_corners=self.align_corners)
        attn = F.interpolate(attn, size=(h_orig, w_orig), mode="bilinear", align_corners=self.align_corners)
        seg_high = F.interpolate(seg_high, size=orig_scaled_res, mode="bilinear", align_corners=self.align_corners)
        embed_low = F.interpolate(embed_low,
                                  size=(int(h_orig // 8), int(w_orig // 8)), mode="bilinear", align_corners=False)

        # mask
        mask = torch.zeros_like(attn)
        mask[:, :, cropy1:cropy2, cropx1:cropx2] = 1
        attn = attn * mask

        # get padded seg_high
        seg_high_padded = torch.zeros_like(seg_low)
        seg_high_padded[:, :, cropy1:cropy2, cropx1:cropx2] = seg_high

        fused_seg = attn * seg_high_padded + (1 - attn) * seg_low
        return fused_seg, seg_high, crop_params, embed_low

    @torch.no_grad()
    def forward_features(self, img):
        bs, c, h_orig, w_orig = img.shape
        # lowres, whole image
        img_low = F.interpolate(img, size=(self.hr_crop_size, self.hr_crop_size), mode="bilinear",
                                align_corners=self.align_corners)
        embed_low = self.forward_encoder(img_low)
        embed_low = F.interpolate(embed_low,
                                  size=(int(h_orig // 8), int(w_orig // 8)), mode="bilinear", align_corners=False)

        # we don't need higres and only need the lowres features as published in MIC
        return embed_low

    @torch.no_grad()
    def inference(self, img):
        bs, c, h_orig, w_orig = img.shape

        img_low = F.interpolate(
            img, (self.hr_crop_size, self.hr_crop_size), mode="bilinear", align_corners=self.align_corners)
        img_high = F.interpolate(
            img, (self.vit_in_img_size, self.vit_in_img_size), mode="bilinear", align_corners=self.align_corners)

        embed_low = self.forward_encoder(img_low)
        seg_low = self.seg_head(embed_low)
        seg_high = self.get_slide_window_segmentation(img_high)
        attn = self.attn_head(embed_low)

        # get orig size
        seg_low = F.interpolate(seg_low, size=(h_orig, w_orig), mode="bilinear", align_corners=self.align_corners)
        attn = torch.sigmoid(
            F.interpolate(attn, size=(h_orig, w_orig), mode="bilinear", align_corners=self.align_corners))
        seg_high = F.interpolate(seg_high, size=(h_orig, w_orig), mode="bilinear", align_corners=self.align_corners)

        fused_seg = attn * seg_high + (1 - attn) * seg_low

        return fused_seg

    def get_slide_window_segmentation(self, img, mask_ratio: float = 0.0):
        batch_size, _, h_img, w_img = img.shape
        h_crop = self.hr_crop_size
        w_crop = self.hr_crop_size
        h_stride = h_crop // 2
        w_stride = w_crop // 2

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        logits = torch.zeros((batch_size, self.num_classes, h_img, w_img)).to(img).float()
        count_mat = torch.zeros((batch_size, 1, h_img, w_img)).to(img).float()
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_logits = self.seg_head(self.forward_encoder(crop_img, mask_ratio))
                crop_logits = F.interpolate(crop_logits, size=(h_crop, w_crop), mode="bilinear",
                                            align_corners=self.align_corners)
                logits += F.pad(crop_logits,
                                (int(x1), int(logits.shape[3] - x2), int(y1),
                                 int(logits.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        logits = logits / count_mat

        return logits
