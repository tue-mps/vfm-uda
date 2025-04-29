# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union

import torch
from torch.utils.data import get_worker_info

import datasets
from datasets.rare_class_sampler_dataset import RareClassSamplerDataset
from datasets.transform.inference.resize_and_crop_transform import ResizeAndCropTransform
from datasets.transform.train.uda_transform import UDATransform
from datasets.utils.custom_lightning_data_module import CustomLightningDataModule
from datasets.utils.mappings import get_cityscapes_mapping
from datasets.utils.source_target_dataset2 import SourceTargetDataset2
from datasets.zip_dataset import ZipDataset


class UDAMultiTrainMultiValDataModule(CustomLightningDataModule):
    def __init__(
            self,
            root,
            devices,
            batch_size: int,
            train_num_workers: int,
            val_num_workers: int = 6,
            in_img_scale: float = 1.0,
            img_size: int = 1024,
            min_scale: float = 0.9,
            max_scale: float = 1.1,
            use_rcs: bool = True,
            rcs_class_temp: float = 0.05,
            cat_max_ratio: float = 0.75,
            sources: list = None,
            targets: list = None,
    ) -> None:
        super().__init__(
            root=root,
            devices=devices,
            batch_size=batch_size,
            img_size=img_size,
            train_num_workers=train_num_workers,
        )
        self.val_num_workers = val_num_workers
        self.rcs_class_temp = rcs_class_temp
        self.use_rcs = use_rcs
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.val_batch_size = 1
        self.sources = sources if sources is not None else []
        self.targets = targets if targets is not None else []
        assert "cityscapesextra" not in self.sources
        # assert self.val_batch_size == 1  # val with multi ds requires batch size 1
        assert not bool(set(self.sources) & set(self.targets)), "sources and targets shouldn't share any datasets"
        self.save_hyperparameters(ignore=['_class_path', "class_path", "init_args"])

        self.gta5_train_transforms = UDATransform(
            img_size=int(1440 * in_img_scale),
            crop_size=img_size,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            label_mapping=get_cityscapes_mapping(),
            cat_max_ratio=cat_max_ratio if ("gta5" in self.sources) and (not use_rcs) else 1.0,
            do_cropping=not ("gta5" in self.sources and use_rcs)
        )

        self.cityscapes_train_transforms = UDATransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            label_mapping=get_cityscapes_mapping(),
            cat_max_ratio=cat_max_ratio if ("cityscapes" in self.sources) and (not use_rcs) else 1.0,
            do_cropping=not ("cityscapes" in self.sources and use_rcs)
        )
        self.wilddash_train_transforms = UDATransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            label_mapping=get_cityscapes_mapping(),
            cat_max_ratio=cat_max_ratio if ("wilddash" in self.sources) and (not use_rcs) else 1.0,
            do_cropping=not ("wilddash" in self.sources and use_rcs)
        )

        self.cityscapes_val_transforms = ResizeAndCropTransform(
            label_mapping=get_cityscapes_mapping(),
            img_size=int(1024 * in_img_scale),
            crop_size=img_size
        )

        # 1080 × 1920
        self.wilddash_val_transforms = ResizeAndCropTransform(
            label_mapping=get_cityscapes_mapping(),
            img_size=int(1024 * in_img_scale),
            crop_size=img_size
        )

    def setup(self, stage: Union[str, None] = None) -> CustomLightningDataModule:
        gta5_train_datasets = [
            ZipDataset(
                zip_path=Path(self.root, f"{i:02}_images.zip"),
                target_zip_path=Path(self.root, f"{i:02}_labels.zip"),
                transforms=self.gta5_train_transforms,
                image_folder_path_in_zip=Path("./images"),
                target_folder_path_in_zip=Path("./labels"),
                image_suffix=".png",
                target_suffix=".png",
            )
            for i in range(1, 11)
        ]
        self.gta5_train_dataset = torch.utils.data.ConcatDataset(gta5_train_datasets)

        cityscapes_dataset_kwargs = {
            "image_suffix": ".png",
            "target_suffix": ".png",
            "image_stem_suffix": "leftImg8bit",
            "target_stem_suffix": "gtFine_labelIds",
            "zip_path": Path(self.root, "leftImg8bit_trainvaltest.zip"),
            "target_zip_path": Path(self.root, "gtFine_trainvaltest.zip"),
        }
        self.cityscapes_train_dataset = ZipDataset(
            transforms=self.cityscapes_train_transforms,
            image_folder_path_in_zip=Path("./leftImg8bit/train"),
            target_folder_path_in_zip=Path("./gtFine/train"),
            **cityscapes_dataset_kwargs,
        )

        self.wilddash_train_dataset = ZipDataset(
            transforms=self.wilddash_train_transforms,
            image_folder_path_in_zip=Path("wilddash/train/images"),
            target_folder_path_in_zip=Path("wilddash/train/labels"),
            image_suffix=".jpg",
            target_suffix=".png",
            zip_path=Path(self.root, "wilddash.zip", ),
        )

        self.all_train_datasets = {
            "gta5": self.gta5_train_dataset,
            "cityscapes": self.cityscapes_train_dataset,
            "wilddash": self.wilddash_train_dataset,
        }

        self.cityscapes_val_dataset = ZipDataset(
            transforms=self.cityscapes_val_transforms,
            image_folder_path_in_zip=Path("./leftImg8bit/val"),
            target_folder_path_in_zip=Path("./gtFine/val"),
            **cityscapes_dataset_kwargs,
        )

        self.wilddash_val_dataset = ZipDataset(
            transforms=self.wilddash_val_transforms,
            image_folder_path_in_zip=Path("wilddash/val/images"),
            target_folder_path_in_zip=Path("wilddash/val/labels"),
            image_suffix=".jpg",
            target_suffix=".png",
            zip_path=Path(self.root, "wilddash.zip", ),
        )

        self.all_val_datasets = {
            "cityscapes": self.cityscapes_val_dataset,
            "wilddash": self.wilddash_val_dataset,
        }

        print("Train ds sizes:", {k: len(v) for k, v in self.all_train_datasets.items()})
        print("Val ds sizes:", {k: len(v) for k, v in self.all_val_datasets.items()})

        return self

    def train_dataloader(self):
        if self.use_rcs:
            source = torch.utils.data.ConcatDataset([
                RareClassSamplerDataset(
                    source=self.all_train_datasets[ds_i],
                    source_name=ds_i,
                    crop_size=self.img_size,
                    rcs_class_temp=self.rcs_class_temp
                )
                for ds_i in self.sources
            ])
        else:
            source = torch.utils.data.ConcatDataset([
                self.all_train_datasets[ds] for ds in self.sources
            ])

        if 0 < len(self.targets):
            target = torch.utils.data.ConcatDataset([
                self.all_train_datasets[ds] for ds in self.targets
            ])

            return torch.utils.data.DataLoader(
                SourceTargetDataset2(source, target),
                shuffle=True,
                drop_last=True,
                persistent_workers=self.persistent_workers,
                num_workers=self.train_num_workers,
                pin_memory=self.pin_memory,
                batch_size=self.batch_size,
            )
        else:
            return torch.utils.data.DataLoader(
                source,
                shuffle=True,
                drop_last=True,
                persistent_workers=self.persistent_workers,
                num_workers=self.train_num_workers,
                pin_memory=self.pin_memory,
                batch_size=self.batch_size,
            )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                self.cityscapes_val_dataset,
                persistent_workers=False,
                num_workers=self.val_num_workers,
                pin_memory=self.pin_memory,
                batch_size=self.val_batch_size,
                collate_fn=datasets.utils.util._eval_collate_fn,
            ),
            torch.utils.data.DataLoader(
                self.wilddash_val_dataset,
                persistent_workers=False,
                num_workers=self.val_num_workers,
                pin_memory=self.pin_memory,
                batch_size=self.val_batch_size,
                collate_fn=datasets.utils.util._eval_collate_fn,
            ),
        ]
