# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------

import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Callable
from typing import Tuple

import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import get_worker_info


class ZipDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            transforms: Callable,
            zip_path: Path,
            image_suffix: str,
            target_suffix: str,
            image_stem_suffix: str = "",
            target_stem_suffix: str = "",
            image_sub_path: str = None,
            target_sub_path: str = None,
            target_zip_path: Optional[Path] = None,
            image_folder_path_in_zip: Path = Path("./"),
            target_folder_path_in_zip: Path = Path("./"),
            max_num_masks: Optional[int] = None,
            is_synthia: bool = False
    ):
        self.transforms = transforms
        self.max_num_masks = max_num_masks
        self.zip_path = zip_path
        self.target_zip_path = target_zip_path
        self.is_synthia = is_synthia

        image_zip, target_zip = self._load_zips()

        self.image_folder_members = {
            str(Path(m.filename)): m
            for m in sorted(image_zip.infolist(), key=self._sort_key)
        }
        self.target_folder_members = {
            str(Path(m.filename)): m for m in target_zip.infolist()
        }

        self.images = []
        self.targets = []
        for image_name, m in self.image_folder_members.items():
            if not self._valid_member(
                    m, image_folder_path_in_zip, image_stem_suffix, image_suffix, image_sub_path
            ):
                continue

            rel_path = Path(image_name).relative_to(image_folder_path_in_zip)
            target_parent = target_folder_path_in_zip / rel_path.parent
            target_stem = rel_path.stem.replace(image_stem_suffix, target_stem_suffix)
            target_name = str(target_parent / (target_stem + target_suffix))
            if (image_sub_path is not None) and (target_sub_path is not None):
                target_name = target_name.replace(image_sub_path, target_sub_path)

            self.images.append(image_name)
            self.targets.append(target_name)

    def _load_zips(self) -> Tuple[zipfile.ZipFile, zipfile.ZipFile]:
        worker = get_worker_info()
        worker = worker.id if worker else None

        if not hasattr(self, "zip"):
            self.zip = {}
        if not hasattr(self, "target_zip"):
            self.target_zip = {}

        if worker not in self.zip:
            self.zip[worker] = zipfile.ZipFile(self.zip_path)
        if worker not in self.target_zip:
            self.target_zip[worker] = (
                zipfile.ZipFile(self.target_zip_path)
                if self.target_zip_path
                else self.zip[worker]
            )

        return self.zip[worker], self.target_zip[worker]

    @staticmethod
    def _sort_key(m: zipfile.ZipInfo):
        match = re.search(r"\d+", m.filename)
        return (int(match.group()) if match else float("inf"), m.filename)

    @staticmethod
    def _valid_member(
            m: zipfile.ZipInfo,
            image_folder_path_in_zip: Path,
            image_stem_suffix: str,
            image_suffix: str,
            image_sub_path: str = None
    ):
        return (
                str(image_folder_path_in_zip) in str(m.filename)
                and m.filename.endswith(image_stem_suffix + image_suffix)
                and not m.is_dir()
                and ((image_sub_path is None) or (image_sub_path in str(m.filename)))
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_zip, target_zip = self._load_zips()
        is_success = False

        while not is_success:
            try:
                with image_zip.open(
                        self.image_folder_members[self.images[index]].filename
                ) as image_file:
                    image = Image.open(image_file).convert("RGB")

                with target_zip.open(
                        self.target_folder_members[self.targets[index]].filename
                ) as target_file:
                    if self.targets[index].endswith(".json"):
                        annotations = json.load(target_file)["annotations"]

                        if (
                                self.max_num_masks is not None
                                and len(annotations) > self.max_num_masks
                        ):
                            ranking_fn = (
                                lambda annotation: (annotation["predicted_iou"] + annotation["stability_score"]) / 2
                            )
                            annotations = sorted(annotations, key=ranking_fn, reverse=True)
                            annotations = annotations[: self.max_num_masks]

                        encoded_masks = [
                            annotation["segmentation"] for annotation in annotations
                        ]
                        decoded_masks = coco_mask.decode(encoded_masks).transpose(2, 0, 1)  # type: ignore
                        target = torch.as_tensor(decoded_masks, dtype=torch.float)
                    else:
                        if self.is_synthia:
                            target = cv2.imdecode(np.frombuffer(target_file.read()), -1)
                            target = target[..., -1]
                            target = target.astype(np.uint8)
                            target = Image.fromarray(target)
                        else:
                            target = Image.open(BytesIO(target_file.read()))

                is_success = True
            except (PIL.UnidentifiedImageError, ValueError):
                index = (index + 1) % len(self)

        return self.transforms(image, target)

    def __del__(self):
        for o in self.zip.values():
            o.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["zip"] = {}
        return state


class ImageOnlyZipDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            transforms: Callable,
            zip_path: Path,
            image_suffix: str,
            image_stem_suffix: str = "",
            image_folder_path_in_zip: Path = Path("./"),
            max_num_masks: Optional[int] = None,
    ):
        self.transforms = transforms
        self.max_num_masks = max_num_masks
        self.zip_path = zip_path

        image_zip = self._load_zips()

        self.image_folder_members = {
            str(Path(m.filename)): m
            for m in sorted(image_zip.infolist(), key=self._sort_key)
        }

        self.images = []
        for image_name, m in self.image_folder_members.items():
            if not self._valid_member(
                    m, image_folder_path_in_zip, image_stem_suffix, image_suffix
            ):
                continue

            rel_path = Path(image_name).relative_to(image_folder_path_in_zip)
            self.images.append(image_name)

    def _load_zips(self) -> zipfile.ZipFile:
        worker = get_worker_info()
        worker = worker.id if worker else None

        if not hasattr(self, "zip"):
            self.zip = {}

        if worker not in self.zip:
            self.zip[worker] = zipfile.ZipFile(self.zip_path)

        return self.zip[worker]

    @staticmethod
    def _sort_key(m: zipfile.ZipInfo):
        match = re.search(r"\d+", m.filename)
        return (int(match.group()) if match else float("inf"), m.filename)

    @staticmethod
    def _valid_member(
            m: zipfile.ZipInfo,
            image_folder_path_in_zip: Path,
            image_stem_suffix: str,
            image_suffix: str,
    ):
        return (
                str(image_folder_path_in_zip) in str(m.filename)
                and m.filename.endswith(image_stem_suffix + image_suffix)
                and not m.is_dir()
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_zip = self._load_zips()

        with image_zip.open(
                self.image_folder_members[self.images[index]].filename
        ) as image_file:
            image = Image.open(image_file).convert("RGB")

        return self.transforms(image, None)

    def __del__(self):
        for o in self.zip.values():
            o.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["zip"] = {}
        return state
