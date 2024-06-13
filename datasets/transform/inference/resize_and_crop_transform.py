from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F


class ResizeAndCropTransform:
    def __init__(
            self,
            img_size: (int, int),
            crop_size: (int, int),
            label_mapping: Optional[dict] = None,
    ):
        self.img_size = img_size
        self.crop_size = crop_size
        self.num_classes = 19

        if label_mapping is not None:
            self.label_mapping = torch.full((256,), 255, dtype=torch.int64)
            for from_id, to_id in label_mapping.items():
                if 0 <= from_id <= 255:
                    self.label_mapping[from_id] = to_id

    def __call__(self, image: Image.Image, target=None) -> (
            torch.Tensor, torch.Tensor, torch.Tensor, list, torch.Tensor):
        scale = self.img_size / min(image.size)
        size = [round(image.size[1] * scale), round(image.size[0] * scale)]

        image = F.resize(image, size)
        target = F.resize(target, size, T.InterpolationMode.NEAREST)

        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        if hasattr(self, "label_mapping"):
            target = self.label_mapping[target]

        _, h_img, w_img = image.shape

        h_crop = self.crop_size
        w_crop = self.crop_size
        h_stride = int(self.crop_size / 2)
        w_stride = int(self.crop_size / 2)

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        preds = torch.zeros((self.num_classes, h_img, w_img), dtype=torch.float)
        count_mat = torch.zeros((1, h_img, w_img), dtype=torch.uint8)

        all_coords = []
        all_crop = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                all_coords += [(y1, y2, x1, x2)]
                all_crop += [image[:, y1:y2, x1:x2].unsqueeze(0)]
                count_mat[:, y1:y2, x1:x2] += 1

        all_crop = torch.cat(all_crop, dim=0)
        assert (count_mat == 0).sum() == 0

        return image, all_crop, preds, count_mat, all_coords, target
