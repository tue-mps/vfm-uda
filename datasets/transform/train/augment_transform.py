from typing import Any
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F


class AugmentTransform:
    def __init__(
            self,
            img_size: int,
            crop_size: int,
            training: bool,
            fine_tuning: bool = True,
            min_scale: float = 0.5,
            max_scale: float = 2.0,
            brightness_delta=32,
            contrast_low=0.5,
            contrast_high=1.5,
            saturation_low=0.5,
            saturation_high=1.5,
            hue_delta=18,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            cat_max_ratio=0.75,
            label_mapping: Optional[dict] = None,
    ):
        self.img_size = img_size
        self.crop_size = crop_size
        self.training = training
        self.fine_tuning = fine_tuning
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.brightness_factor = brightness_delta / 255.0
        self.contrast_low = contrast_low
        self.contrast_high = contrast_high
        self.saturation_low = saturation_low
        self.saturation_high = saturation_high
        self.hue_factor = hue_delta / 360.0
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = 255
        self.mean = mean
        self.std = std

        if label_mapping is not None:
            self.label_mapping = torch.full((256,), 255, dtype=torch.int64)
            for from_id, to_id in label_mapping.items():
                if 0 <= from_id <= 255:
                    self.label_mapping[from_id] = to_id

    def _torch_scaled_rand(self, min_val, max_val):
        return min_val + torch.rand(1).item() * (max_val - min_val)

    def _resize(self, image: Image.Image, target, scale):
        size = [round(image.size[1] * scale), round(image.size[0] * scale)]
        image = F.resize(image, size)  # type: ignore
        target = F.resize(target, size, T.InterpolationMode.NEAREST)

        return image, target

    def _flip(self, image: Any, target: Any):
        if torch.rand(1) < 0.5:
            image, target = F.hflip(image), F.hflip(target)

        return image, target

    def _brightness(self, image: Any):
        if torch.rand(1) < 0.5:
            image = F.adjust_brightness(
                image,
                self._torch_scaled_rand(1 - self.brightness_factor, 1 + self.brightness_factor)
            )

        return image

    def _contrast(self, image: Any):
        if torch.rand(1) < 0.5:
            image = F.adjust_contrast(
                image,
                self._torch_scaled_rand(self.contrast_low, self.contrast_high)
            )

        return image

    def _saturation_and_hue(self, image: Any):
        if torch.rand(1) < 0.5:
            image = F.adjust_saturation(
                image,
                self._torch_scaled_rand(self.saturation_low, self.saturation_high)
            )

        if torch.rand(1) < 0.5:
            image = F.adjust_hue(
                image,
                self._torch_scaled_rand(-self.hue_factor, self.hue_factor)
            )

        return image

    def _pad(self, image: torch.Tensor, target: torch.Tensor):
        pad_w = max(self.crop_size - image.shape[2], 0)
        pad_h = max(self.crop_size - image.shape[1], 0)

        image = F.pad(image, [0, 0, pad_w, pad_h])
        target = F.pad(
            target,
            [0, 0, pad_w, pad_h],
            fill=255 if self.fine_tuning else 0,
        )

        return image, target

    def _crop(self, image, target):
        params = T.RandomCrop.get_params(image, (self.crop_size, self.crop_size))

        if self.cat_max_ratio < 1.:
            for _ in range(10):
                temp = F.crop(target, *params)
                labels, cnt = np.unique(temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if (len(cnt) > 1) and ((np.max(cnt) / np.sum(cnt)) < self.cat_max_ratio):
                    break
                params = T.RandomCrop.get_params(image, (self.crop_size, self.crop_size))

        return F.crop(image, *params), F.crop(target, *params)

    def _color_augment_image(self, image):
        image = image.copy()
        image = self._brightness(image)
        if torch.rand(1) < 0.5:
            image = self._contrast(image)
            image = self._saturation_and_hue(image)
        else:
            image = self._saturation_and_hue(image)
            image = self._contrast(image)
        return image

    def __call__(self, image: Image.Image, target=None):
        if self.training:
            if self.fine_tuning:
                scale = (self.img_size / min(image.size)) * self._torch_scaled_rand(
                    self.min_scale, self.max_scale)

            else:
                scale = self.img_size / max(image.size)

            image, target = self._resize(image, target, scale)

            if self.fine_tuning:
                image, target = self._flip(image, target)
                image = self._color_augment_image(image)

        image_tensor = F.convert_image_dtype(F.pil_to_tensor(image), dtype=torch.float)
        image_tensor = F.normalize(image_tensor, mean=self.mean, std=self.std)

        if self.fine_tuning:
            target_tensor = torch.as_tensor(np.array(target), dtype=torch.int64)
        elif type(target) == torch.Tensor:
            target_tensor = target
        else:
            raise RuntimeError("Target is of unknown type")

        if hasattr(self, "label_mapping"):
            target_tensor = self.label_mapping[target_tensor]

        if self.training:
            image_tensor, target_tensor = self._pad(image_tensor, target_tensor)

            if self.fine_tuning:
                image_tensor, target_tensor = self._crop(image_tensor, target_tensor)

        return image_tensor, target_tensor
