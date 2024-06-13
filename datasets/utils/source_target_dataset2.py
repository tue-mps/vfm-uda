from typing import Tuple, Union

import torch
from torch.utils.data import get_worker_info

from datasets.zip_dataset import ZipDataset


class SourceTargetDataset2(torch.utils.data.Dataset):
    def __init__(
            self,
            source: Union[ZipDataset, torch.utils.data.ConcatDataset],
            target: Union[ZipDataset, torch.utils.data.ConcatDataset]):
        self.source = source
        self.target = target

    def __len__(self):
        # https://github.com/pytorch/pytorch/issues/116479
        # return len(self.source) * len(self.target) <-- don't multiply
        return min(len(self.source) * len(self.target), 999999999)

    def __getitem__(self, index: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        # source_index = index // len(self.target)
        # target_index = index % len(self.target)
        source_index = torch.randint(0, len(self.source), (1,)).item()
        target_index = torch.randint(0, len(self.target), (1,)).item()

        sourceds_image_tensor, sourceds_target_tensor, sourceds_ignore_mask = \
            self.source[source_index]
        targetds_image_tensor, targetds_target_tensor, targetds_ignore_mask = \
            self.target[target_index]

        return (sourceds_image_tensor, sourceds_target_tensor, sourceds_ignore_mask,
                targetds_image_tensor, targetds_target_tensor, targetds_ignore_mask)
