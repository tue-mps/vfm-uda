# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------

import json
from os.path import join
from typing import Tuple, Union

import numpy as np
import torch
from torchvision import transforms as TV
from torchvision.transforms import functional as FV

from datasets.zip_dataset import ZipDataset


class RareClassSamplerDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            source: Union[ZipDataset, torch.utils.data.ConcatDataset],
            source_name: str,
            crop_size: int,
            rcs_class_temp: float = 0.05
    ):
        self.source = source
        self.crop_size = crop_size
        self.ignore_index = 255
        self.rcs_min_pixels = 3000
        self.rcs_class_temp = rcs_class_temp
        self.rcs_min_crop_ratio = crop_size * crop_size / (512.0 * 1024.0)
        self.cat_max_ratio = 0.75
        self.json_root_data_stat = join("datasets/rcs_stats", source_name)

        self.rcs_classes, self.rcs_classprob = self.get_rcs_class_probs(
            self.json_root_data_stat, self.rcs_class_temp)

        print("rcs_classes", self.rcs_classes)
        print("rcs_classprob", self.rcs_classprob)

        with open(join(self.json_root_data_stat, 'samples_with_class.json'), 'r') as of:
            samples_with_class_and_n = json.load(of)
        samples_with_class_and_n = {
            int(k): v
            for k, v in samples_with_class_and_n.items()
            if int(k) in self.rcs_classes
        }

        self.samples_with_class = {}
        for c in self.rcs_classes:
            self.samples_with_class[c] = []
            for file_index, pixels in samples_with_class_and_n[c]:
                if pixels > self.rcs_min_pixels:
                    self.samples_with_class[c].append(file_index)
            assert len(self.samples_with_class[c]) > 0
            # print("Rare class {} has {} valid samples.".format(c, len(self.samples_with_class[c])))

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get image with rare class
        chosen_class = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        source_index = np.random.choice(self.samples_with_class[chosen_class])
        image_tensor, target_tensor, ignore_mask = self.source[source_index]
        c, h, w = image_tensor.shape

        pad_w = max(self.crop_size - w, 0)
        pad_h = max(self.crop_size - h, 0)

        image_tensor = FV.pad(image_tensor, [0, 0, pad_w, pad_h])
        target_tensor = FV.pad(target_tensor, [0, 0, pad_w, pad_h], fill=self.ignore_index)
        ignore_mask = FV.pad(ignore_mask, [0, 0, pad_w, pad_h], fill=self.ignore_index)

        target_tensor = target_tensor.to(torch.uint8)

        for j in range(20):
            params = TV.RandomCrop.get_params(image_tensor, (self.crop_size, self.crop_size))
            crop_target_tensor = FV.crop(target_tensor, *params)
            n_class = torch.sum(crop_target_tensor == chosen_class)
            rcs_condition = n_class > (self.rcs_min_pixels * self.rcs_min_crop_ratio)

            labels, cnt = np.unique(crop_target_tensor, return_counts=True)
            cnt = cnt[labels != self.ignore_index]
            cat_max_condition = (len(cnt) > 1) and ((np.max(cnt) / np.sum(cnt)) < self.cat_max_ratio)

            if rcs_condition and cat_max_condition:
                crop_image_tensor = FV.crop(image_tensor, *params)
                crop_ignore_mask = FV.crop(ignore_mask, *params)
                return crop_image_tensor, crop_target_tensor.to(torch.long), crop_ignore_mask

        params = TV.RandomCrop.get_params(image_tensor, (self.crop_size, self.crop_size))
        crop_image_tensor = FV.crop(image_tensor, *params)
        crop_target_tensor = FV.crop(target_tensor, *params)
        crop_ignore_mask = FV.crop(ignore_mask, *params)
        return crop_image_tensor, crop_target_tensor.to(torch.long), crop_ignore_mask

    def get_rcs_class_probs(self, data_root, temperature):
        with open(join(data_root, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)
        overall_class_stats = {}
        for s in sample_class_stats:
            s.pop('file')
            for c, n in s.items():
                c = int(c)
                if c not in overall_class_stats:
                    overall_class_stats[c] = n
                else:
                    overall_class_stats[c] += n
        overall_class_stats = {
            k: v
            for k, v in sorted(
                overall_class_stats.items(), key=lambda item: item[1])
        }
        freq = torch.tensor(list(overall_class_stats.values()))
        freq = freq / torch.sum(freq)
        freq = 1 - freq
        freq = torch.softmax(freq / temperature, dim=-1)

        return list(overall_class_stats.keys()), freq.numpy()
