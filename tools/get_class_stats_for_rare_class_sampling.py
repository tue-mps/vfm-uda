import json
import os.path as osp
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from datasets.utils.mappings import get_cityscapes_mapping, get_mapillary_mapping
from datasets.zip_dataset import ZipDataset


def convert_to_train_id(data_index, label):
    label_ids, counts = torch.unique(label, return_counts=True)
    label_ids, counts = label_ids.cpu().numpy(), counts.cpu().numpy()
    sample_class_stats = {int(k): int(v) for (k, v) in zip(label_ids, counts)}
    sample_class_stats.pop(255, None)  # remove ignore label

    sample_class_stats['file'] = data_index
    return sample_class_stats


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


class DummyTransform:
    def __init__(self, label_mapping: Optional[dict] = None):
        self.label_mapping = label_mapping

        if self.label_mapping is not None:
            self.label_mapping = np.full((256,), 255, dtype=np.int64)
            for from_id, to_id in label_mapping.items():
                if 0 <= from_id <= 255:
                    self.label_mapping[from_id] = to_id

    def __call__(self, image, target):
        target = np.array(target, dtype=np.uint8)
        if self.label_mapping is not None:
            target = self.label_mapping[target]
        return target


def main():
    root = "/home/bruno/datasets/foundation-generalization_and_uda/"
    gta5_train_datasets = [
        ZipDataset(
            zip_path=Path(root, f"{i:02}_images.zip"),
            target_zip_path=Path(root, f"{i:02}_labels.zip"),
            transforms=DummyTransform(get_cityscapes_mapping()),
            image_folder_path_in_zip=Path("./images"),
            target_folder_path_in_zip=Path("./labels"),
            image_suffix=".png",
            target_suffix=".png",
        )
        for i in range(1, 11)
    ]
    gta5_train_dataset = torch.utils.data.ConcatDataset(gta5_train_datasets)

    synscapes_train_dataset = ZipDataset(
        transforms=DummyTransform(get_cityscapes_mapping()),
        image_folder_path_in_zip=Path("./Synscapes/img/rgb-2k"),
        target_folder_path_in_zip=Path("./Synscapes/img/class"),
        image_suffix=".png",
        target_suffix=".png",
        zip_path=Path(
            root,
            "synscapes.zip",
        ),
    )

    cityscapes_dataset_kwargs = {
        "image_suffix": ".png",
        "target_suffix": ".png",
        "image_stem_suffix": "leftImg8bit",
        "target_stem_suffix": "gtFine_labelIds",
        "zip_path": Path(root, "leftImg8bit_trainvaltest.zip"),
        "target_zip_path": Path(root, "gtFine_trainvaltest.zip"),
    }
    cityscapes_train_dataset = ZipDataset(
        transforms=DummyTransform(get_cityscapes_mapping()),
        image_folder_path_in_zip=Path("./leftImg8bit/train"),
        target_folder_path_in_zip=Path("./gtFine/train"),
        **cityscapes_dataset_kwargs,
    )

    bdd100k_train_dataset = ZipDataset(
        transforms=DummyTransform(),
        image_folder_path_in_zip=Path("./bdd100k/images/10k/train"),
        target_folder_path_in_zip=Path("./bdd100k/labels/sem_seg/masks/train"),
        image_suffix=".jpg",
        target_suffix=".png",
        zip_path=Path(root, "bdd100k_images_10k.zip"),
        target_zip_path=Path(root, "bdd100k_sem_seg_labels_trainval.zip"),
    )

    mapillary_train_dataset = ZipDataset(
        transforms=DummyTransform(get_mapillary_mapping()),
        image_folder_path_in_zip=Path("./training/images"),
        target_folder_path_in_zip=Path("./training/labels"),
        image_suffix=".jpg",
        target_suffix=".png",
        zip_path=Path(
            root,
            "An_o5cmHOsS1VbLdaKx_zfMdi0No5LUpL2htRxMwCjY_bophtOkM0-6yTKB2T2sa0yo1oP086sqiaCjmNEw5d_pofWyaE9LysYJagH8yXw_GZPzK2wfiQ9u4uAKrVcEIrkJiVuTn7JBumrA.zip",
        ),
    )

    wilddash_train_dataset = ZipDataset(
        transforms=DummyTransform(get_cityscapes_mapping()),
        image_folder_path_in_zip=Path("wilddash/train/images"),
        target_folder_path_in_zip=Path("wilddash/train/labels"),
        image_suffix=".jpg",
        target_suffix=".png",
        zip_path=Path(root, "wilddash.zip", ),
    )

    acdc_train_dataset = ZipDataset(
        transforms=DummyTransform(),
        image_folder_path_in_zip=Path("acdc_rgb_anon_train/rgb_anon"),
        target_folder_path_in_zip=Path("acdc_gt_train/gt"),
        image_suffix=".png",
        target_suffix=".png",
        image_stem_suffix="rgb_anon",
        target_stem_suffix="gt_labelTrainIds",
        zip_path=Path(root, "acdc_rgb_anon_train.zip"),
        target_zip_path=Path(root, "acdc_gt_train.zip"),
    )

    synthia_train_dataset = ZipDataset(
        transforms=DummyTransform(),
        image_folder_path_in_zip=Path("RAND_CITYSCAPES/RGB"),
        target_folder_path_in_zip=Path("RAND_CITYSCAPES/GT/LABELS"),
        image_suffix=".png",
        target_suffix=".png",
        zip_path=Path(root, "SYNTHIA_RAND_CITYSCAPES.zip"),
        is_synthia=True
    )

    all_train_datasets = {
        "gta5": gta5_train_dataset,
        "city": cityscapes_train_dataset,
        "bdd": bdd100k_train_dataset,
        "mapilllary": mapillary_train_dataset,
        "snyscapes": synscapes_train_dataset,
        "wilddash": wilddash_train_dataset,
        "acdc": acdc_train_dataset,
        "synthia": synthia_train_dataset,
    }

    for name, dataset in all_train_datasets.items():
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            batch_size=1,
        )

        sample_class_stats = []
        for i, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            x = x.cuda()
            sample_class_stats += [convert_to_train_id(i, x[0])]

        outdir = "out/{}".format(name)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        save_class_stats(outdir, sample_class_stats)  #


if __name__ == '__main__':
    main()
