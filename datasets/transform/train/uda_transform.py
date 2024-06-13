from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TV
from torchvision.transforms import functional as FV


class UDATransform:
    def __init__(
            self,
            img_size: int,
            crop_size: int,
            min_scale: float = 0.9,
            max_scale: float = 1.1,
            cat_max_ratio=0.75,
            label_mapping: Optional[dict] = None,
            do_cropping: bool = True
    ):

        self.ignore_index = 255
        self.img_size = img_size
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.cat_max_ratio = cat_max_ratio
        self.pseudo_weight_ignore_top = 0.03
        self.pseudo_weight_ignore_bottom = 0.25
        self.pseudo_weight_ignore_left = 0.03
        self.pseudo_weight_ignore_right = 0.03
        self.do_cropping = do_cropping

        if label_mapping is not None:
            self.label_mapping = torch.full((256,), 255, dtype=torch.int64)
            for from_id, to_id in label_mapping.items():
                if 0 <= from_id <= 255:
                    self.label_mapping[from_id] = to_id

    def _torch_scaled_rand(self, min_val, max_val):
        return min_val + torch.rand(1).item() * (max_val - min_val)

    def __call__(self, image: Image.Image, target=None):
        h, w, c, = np.array(image).shape
        ignore_mask = np.zeros((h, w), dtype=np.uint8)
        ignore_mask[:int(self.pseudo_weight_ignore_top * h)] = self.ignore_index
        ignore_mask[-int(self.pseudo_weight_ignore_bottom * h):] = self.ignore_index
        ignore_mask[:, :int(self.pseudo_weight_ignore_left * w)] = self.ignore_index
        ignore_mask[:, -int(self.pseudo_weight_ignore_right * w):] = self.ignore_index
        ignore_mask = Image.fromarray(ignore_mask)

        scale = self.img_size / min(image.size) * self._torch_scaled_rand(
            self.min_scale, self.max_scale)

        size = [round(h * scale), round(w * scale)]
        image = FV.resize(image, size)
        if target is not None:
            target = FV.resize(target, size, TV.InterpolationMode.NEAREST)
        ignore_mask = FV.resize(ignore_mask, size, TV.InterpolationMode.NEAREST)
        image_tensor = FV.pil_to_tensor(image)

        if target is not None:
            target_tensor = torch.as_tensor(np.array(target), dtype=torch.int64)
        ignore_mask = torch.as_tensor(np.array(ignore_mask), dtype=torch.int64)

        if target is not None and hasattr(self, "label_mapping"):
            target_tensor = self.label_mapping[target_tensor]

        if not self.do_cropping:
            if target is not None:
                return image_tensor, target_tensor, ignore_mask
            else:
                return image_tensor, torch.ones_like(ignore_mask) * self.ignore_index, ignore_mask

        pad_w = max(self.crop_size - image_tensor.shape[2], 0)
        pad_h = max(self.crop_size - image_tensor.shape[1], 0)
        image_tensor = FV.pad(image_tensor, [0, 0, pad_w, pad_h])
        if target is not None:
            target_tensor = FV.pad(target_tensor, [0, 0, pad_w, pad_h], fill=self.ignore_index)
        ignore_mask = FV.pad(ignore_mask, [0, 0, pad_w, pad_h], fill=self.ignore_index)

        params = TV.RandomCrop.get_params(image_tensor, (self.crop_size, self.crop_size))

        if target is not None and self.cat_max_ratio < 1.0:
            for ci in range(10):
                temp = FV.crop(target_tensor, *params)
                labels, cnt = np.unique(temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if (len(cnt) > 1) and ((np.max(cnt) / np.sum(cnt)) < self.cat_max_ratio):
                    break
                params = TV.RandomCrop.get_params(image_tensor, (self.crop_size, self.crop_size))

        image_tensor = FV.crop(image_tensor, *params)
        if target is not None:
            target_tensor = FV.crop(target_tensor, *params)
        ignore_mask = FV.crop(ignore_mask, *params)

        if target is not None:
            return image_tensor, target_tensor, ignore_mask
        else:
            return image_tensor, torch.ones_like(ignore_mask) * self.ignore_index, ignore_mask
