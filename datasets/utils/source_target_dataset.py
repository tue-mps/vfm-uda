from typing import Tuple, Union

import torch
from torch.utils.data import get_worker_info

from datasets.zip_dataset import ZipDataset


class SourceTargetDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            source: Union[ZipDataset, torch.utils.data.ConcatDataset],
            target: Union[ZipDataset, torch.utils.data.ConcatDataset]):
        self.source = source
        self.target = target

    def __len__(self):
        return len(self.source) * len(self.target)

    def __getitem__(self, index: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        source_index = index // len(self.target)
        target_index = index % len(self.target)

        sourceds_image_tensor, sourceds_color_aug_image_tensor, sourceds_target_tensor, sourceds_ignore_mask = \
            self.source[source_index]
        targetds_image_tensor, targetds_color_aug_image_tensor, targetds_target_tensor, targetds_ignore_mask = \
            self.target[target_index]

        return (sourceds_image_tensor, sourceds_color_aug_image_tensor, sourceds_target_tensor, sourceds_ignore_mask,
                targetds_image_tensor, targetds_color_aug_image_tensor, targetds_target_tensor, targetds_ignore_mask)
