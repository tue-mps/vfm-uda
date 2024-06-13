# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
from torch import nn

from models.daformer.aspp_head import ASPPWrapper
from models.daformer.mlp import MLP
from models.daformer.utils import resize


class DAFormerHead(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.in_index = [0, 1, 2, 3]
        self.in_channels = [64, 128, 320, 512]
        embed_dims = 256
        self.channels = 256
        self.dropout_ratio = 0.1
        self.num_classes = num_classes
        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.align_corners = False
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = dict(type='mlp', act_cfg=None, norm_cfg=None)
        embed_neck_cfg = dict(type='mlp', act_cfg=None, norm_cfg=None)

        fusion_cfg = dict(
            # _delete_=True,
            align_corners=self.align_corners,
            type='aspp',
            sep=True,
            dilations=(1, 6, 12, 18),
            pool=False,
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN', requires_grad=True))

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels, embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.conv_seg.weight.data.normal_(mean=0, std=0.01)
        self.conv_seg.bias.data.zero_()

        self.dropout = nn.Dropout2d(self.dropout_ratio)

    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous() \
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        feat = self.dropout(x)
        output = self.conv_seg(feat)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)

    else:
        raise NotImplementedError(type)
