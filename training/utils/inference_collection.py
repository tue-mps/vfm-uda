# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------


import torch
import torch.nn.functional as F
from torchvision.transforms import functional as FV


@torch.no_grad()
def multiscale_inference(img):
    b, c, h, w = img.shape
    orig_img_size = (h, w)
    ratios = [0.6, 1.0, 1.7, 2.3]
    img_sizes = [(int(orig_img_size[0] * r), int(orig_img_size[1] * r)) for r in ratios]

    segmentation = []
    for (h, w) in img_sizes:
        x1 = F.interpolate(img, size=(h, w), mode="bilinear")
        x2 = FV.hflip(x1)
        logit1 = F.interpolate(slide_inference(x1).detach(), size=orig_img_size, mode="bilinear")
        logit2 = F.interpolate(FV.hflip(slide_inference(x2).detach()), size=orig_img_size, mode="bilinear")
        pred1 = torch.softmax(logit1, dim=1).detach()
        pred2 = torch.softmax(logit2, dim=1).detach()
        segmentation += [pred1.unsqueeze(0)]
        segmentation += [pred2.unsqueeze(0)]

    segmentation = torch.cat(segmentation, dim=0).mean(0)
    return segmentation


@torch.no_grad()
def slide_inference(network, img, crop_size, num_classes):
    batch_size, _, h_img, w_img = img.shape
    h_crop = crop_size
    w_crop = crop_size
    h_stride = crop_size // 2
    w_stride = crop_size // 2

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = torch.zeros((batch_size, num_classes, h_img, w_img)).to(img).float()
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
            crop_seg_logit = network(crop_img)

            preds += F.pad(crop_seg_logit,
                           (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0
    seg_logits = preds / count_mat

    return seg_logits


@torch.no_grad()
def slide_inference_pre_cropped(network, crops, preds, count_mat, coords, num_classes):
    # we clip the classes only for the binning network
    crops = network(crops)[:, :num_classes, :, :]
    crops = torch.softmax(crops, dim=1)

    for (y1, y2, x1, x2), crp in zip(coords, crops):
        preds[:, y1:y2, x1:x2] += crp

    seg_logits = preds / count_mat.float()
    return seg_logits
