"""MVTec data, fixed evaluation partitions and DTD/Perlin augmentation.

Augmentation is adapted from DRAEM; see THIRD_PARTY_NOTICES.md.
"""

import glob
import json
from pathlib import Path

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch.utils.data import Dataset

from .perlin import rand_perlin_2d_np


class MvtecTrainDataset(Dataset):
    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.image_paths = sorted(glob.glob(root_dir + "/*.png"))
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))
        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45)),
        ]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential(
            [self.augmenters[aug_ind[0]], self.augmenters[aug_ind[1]], self.augmenters[aug_ind[2]]]
        )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        if anomaly_source_img is None:
            raise FileNotFoundError(f"Cannot read texture: {anomaly_source_path}")
        anomaly_source_img = cv2.resize(
            anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0])
        )
        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
        perlin_scaley = 2 ** torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
        perlin_noise = rand_perlin_2d_np(
            (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley)
        )
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(
            perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise)
        )
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
        beta = torch.rand(1).numpy()[0] * 0.8
        augmented_image = (
            image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * perlin_thr
        )
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return (
                image,
                np.zeros_like(perlin_thr, dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = perlin_thr.astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return (augmented_image, msk, np.array([has_anomaly], dtype=np.float32))

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read training image: {image_path}")
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)
        image = (
            np.array(image)
            .reshape((image.shape[0], image.shape[1], image.shape[2]))
            .astype(np.float32)
            / 255.0
        )
        (augmented_image, anomaly_mask, has_anomaly) = self.augment_image(
            image, anomaly_source_path
        )
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return (image, augmented_image, anomaly_mask, has_anomaly)

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        (image, augmented_image, anomaly_mask, has_anomaly) = self.transform_image(
            self.image_paths[idx], self.anomaly_source_paths[anomaly_source_idx]
        )
        sample = {
            "image": image,
            "anomaly_mask": anomaly_mask,
            "augmented_image": augmented_image,
            "has_anomaly": has_anomaly,
            "idx": idx,
        }
        return sample


class TrainDataset(MvtecTrainDataset):
    def __init__(self, root, dtd, size=256):
        super().__init__(str(Path(root) / "train" / "good"), str(dtd), [size, size])
        if not self.image_paths or not self.anomaly_source_paths:
            raise ValueError(f"Empty training or DTD folder: {root}, {dtd}")

    def __getitem__(self, idx):
        source_idx = torch.randint(len(self.anomaly_source_paths), (1,)).item()
        (image, aug, mask, has) = self.transform_image(
            self.image_paths[idx], self.anomaly_source_paths[source_idx]
        )
        return dict(image=image, augmented_image=aug, anomaly_mask=mask, has_anomaly=has)


def make_split(root, path, seed=2026):
    (root, path) = (Path(root), Path(path))
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        if result.get("seed") != seed:
            raise ValueError("Requested split seed differs from the existing split")
        validate_split(root, result)
        return result
    rng = np.random.default_rng(seed)
    result = dict(
        seed=seed,
        description="Stratified 40% validation / 60% held-out test from original MVTec test; all train/good used only for training.",
        validation=[],
        test=[],
    )
    for folder in sorted((root / "test").iterdir()):
        if not folder.is_dir():
            continue
        images = sorted(folder.glob("*.png"))
        if len(images) < 2:
            raise ValueError(f"Need at least two PNG images in {folder}")
        order = rng.permutation(len(images))
        n_val = max(1, round(len(images) * 0.4))
        for split, indices in [("validation", order[:n_val]), ("test", order[n_val:])]:
            result[split].extend([images[i].relative_to(root).as_posix() for i in indices])
    validate_split(root, result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


def validate_split(root, split):
    """Reject overlap, missing files and a split from a different category."""
    root = Path(root)
    combined = []
    for name in ["validation", "test"]:
        paths = split.get(name, [])
        if not paths or len(paths) != len(set(paths)):
            raise ValueError(f"Empty or duplicated {name} split")
        for relative in paths:
            p = Path(relative)
            if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != "test":
                raise ValueError(f"Invalid relative test path: {relative}")
            if not (root / p).is_file():
                raise FileNotFoundError(root / p)
        labels = {Path(p).parent.name == "good" for p in paths}
        if labels != {False, True}:
            raise ValueError(f"{name} needs both normal and anomalous images")
        combined.extend(paths)
    if len(combined) != len(set(combined)):
        raise ValueError("Validation and test splits overlap")
    expected = {p.relative_to(root).as_posix() for p in (root / "test").glob("*/*.png")}
    if set(combined) != expected:
        raise ValueError(
            "Split does not match this category; pass a category-specific --split-file"
        )


class EvalDataset(Dataset):
    def __init__(self, root, paths, size=256):
        (self.root, self.paths, self.size) = (Path(root), paths, size)
        if not paths:
            raise ValueError("Empty evaluation split")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        relative = self.paths[idx]
        path = self.root / relative
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        anomalous = path.parent.name != "good"
        if anomalous:
            mask_path = self.root / "ground_truth" / path.parent.name / (path.stem + "_mask.png")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
        else:
            mask = np.zeros(image.shape[:2], np.uint8)
        image = cv2.resize(image, (self.size, self.size)).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST) > 127
        return dict(
            image=image.transpose(2, 0, 1),
            mask=mask.astype(np.uint8),
            label=int(anomalous),
            path=relative,
        )
