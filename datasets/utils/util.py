from typing import List, Tuple

import numpy as np
import torch
import torchvision
from torchvision import transforms as T


def create_legend(classes_in_image):
    legend = []
    for class_id in classes_in_image:
        for class_ in torchvision.datasets.Cityscapes.classes:
            if class_id == class_.train_id:
                class_color = class_.color
                class_name = f"{class_.name} ({class_id})"
                legend.append((class_id, class_color, class_name))
                break

    return legend


def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id in np.unique(mask):
        for class_ in torchvision.datasets.Cityscapes.classes:
            if class_id == class_.train_id:
                color_mask[mask == class_id] = class_.color
                break

    return color_mask


# Define the inverse normalization transformation
normalize_inverse = T.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)
normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def _eval_collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[List[torch.Tensor], ...]:
    return tuple(map(list, zip(*batch)))
