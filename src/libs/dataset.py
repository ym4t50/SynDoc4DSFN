import csv
import random
from logging import getLogger
from typing import Any, Dict, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset_csv import DATASET_CSVS

__all__ = ["get_dataloader"]

logger = getLogger(__name__)


def get_dataloader(
    dataset_name: str,
    train_model: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    transform: Optional[A.Compose] = None,
    sampling: int = 1,
    finetuning: bool = False,
    **kwargs: Any,
) -> DataLoader:
    if dataset_name not in DATASET_CSVS:
        message = f"dataset_name should be selected from {list(DATASET_CSVS.keys())}."
        logger.error(message)
        raise ValueError(message)

    if train_model not in [
        "benet",
        "cam_benet",
        "srnet",
        "bedsrnet",
        "stcgan",
        "generator",
        "discriminator",
        "dsfn",
    ]:
        message = "train_model should be selected from \
                    ['benet', 'cam_benet', 'srnet', 'bedsrnet', stcgan', \
                    'generator', 'discriminator',  \
                    'dsfn']."
        logger.error(message)
        raise ValueError(message)

    if split not in ["train", "val", "test"]:
        message = "split should be selected from ['train', 'val', 'test']."
        logger.error(message)
        raise ValueError(message)

    logger.info(f"Dataset: {dataset_name}\tSplit: {split}\tBatch size: {batch_size}.")

    data: Dataset
    csv_file = getattr(DATASET_CSVS[dataset_name], split)

    if train_model == "benet" or train_model == "bedsrnet" or train_model == "stcgan":
        use_bg_color = True
    else:
        use_bg_color = False

    if finetuning is False and dataset_name in ["SynDocDS", "SDSRD"]:
        data = DocDataset(
            csv_file,
            transform=transform,
            phase=split,
            use_bg_color=use_bg_color,
            sampling=sampling,
            bit=8,
            **kwargs,
        )
    else:
        data = DocDataset(
            csv_file,
            transform=transform,
            phase=split,
            use_bg_color=use_bg_color,
            sampling=sampling,
            color_jitter=False,
            jpeg_comp=False,
            bit=8,
            **kwargs,
        )

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


class DocDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        transform: Optional[A.Compose] = None,
        phase="train",
        use_bg_color: bool = False,
        jpeg_comp: bool = True,
        color_jitter: bool = True,
        sampling: int = 1,
        bit: int = 8,
    ) -> None:
        super().__init__()

        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError("csv file not found.") as e:  # type: ignore
            logger.exception(f"{e}")

        self.transform = transform
        self.phase = phase
        self.use_bg_color = use_bg_color
        self.jpeg_comp = jpeg_comp
        self.color_jitter = color_jitter
        self.max_value = 2**bit - 1
        self.sampling = sampling
        self.color_transforms = [
            adjust_brightness,
            adjust_contrast,
            adjust_saturation,
            hue,
        ]

        logger.info(f"the number of samples: {len(self.df)//self.sampling}")

    def __len__(self) -> int:
        return len(self.df) // self.sampling

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {}
        shadow_img = None
        shadowfree_img = None
        shadow_mask = None

        suffix = np.random.randint(0, self.sampling) if self.phase == "train" else 0

        shadow_img_path = self.df.iloc[idx * self.sampling + suffix]["shadow_image"]
        shadow_img = cv2.imread(shadow_img_path)
        shadow_img = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB)

        shadowfree_img_path = self.df.iloc[idx * self.sampling + suffix][
            "shadowfree_image"
        ]
        shadowfree_img = cv2.imread(shadowfree_img_path)
        shadowfree_img = cv2.cvtColor(shadowfree_img, cv2.COLOR_BGR2RGB)

        shadow_mask_path = self.df.iloc[idx * self.sampling + suffix]["shadow_matte"]
        shadow_mask = cv2.imread(shadow_mask_path, 0)
        shadow_mask = shadow_mask.astype(np.float64) / self.max_value

        h, w, c = shadowfree_img.shape
        if self.use_bg_color:
            bg_color_path = self.df.iloc[idx * self.sampling + suffix][
                "background_color_csv"
            ]
            with open(bg_color_path, newline="") as f:
                reader = csv.reader(f)
                bg_color = np.array(list(map(float, next(reader)))).reshape(1, 1, 3)
                bg_color = bg_color[:, :, ::-1]

            bg_color_img = np.full_like(
                shadowfree_img.flatten().reshape(h * w, c), bg_color
            ).reshape(h, w, c)

        # Data Augmentation (Color Jitter)
        if self.phase == "train" and self.color_jitter:
            params_dic = self.get_params()
            shadow_img = self.apply(
                shadow_img,
                brightness=params_dic["brightness"],
                contrast=params_dic["brightness"],
                saturation=params_dic["brightness"],
                hue=params_dic["brightness"],
                order=params_dic["order"],
            )
            shadowfree_img = self.apply(
                shadowfree_img,
                brightness=params_dic["brightness"],
                contrast=params_dic["brightness"],
                saturation=params_dic["brightness"],
                hue=params_dic["brightness"],
                order=params_dic["order"],
            )
            if self.use_bg_color:
                bg_color = self.apply(
                    bg_color.astype(np.uint8),
                    brightness=params_dic["brightness"],
                    contrast=params_dic["brightness"],
                    saturation=params_dic["brightness"],
                    hue=params_dic["brightness"],
                    order=params_dic["order"],
                )

        # Data Augmentation (jpeg compression)
        if np.random.random() > 0.5 and self.phase == "train" and self.jpeg_comp:
            quality = np.random.uniform(low=0.3, high=0.95)
            _, encoded_img = cv2.imencode(
                ".jpg", shadow_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality * 100)]
            )
            shadow_img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

            _, encoded_img = cv2.imencode(
                ".jpg",
                shadowfree_img,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(quality * 100)],
            )
            shadowfree_img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

        shadowfree_img = shadowfree_img.astype(np.float64) / self.max_value
        shadow_img = shadow_img.astype(np.float64) / self.max_value

        if self.use_bg_color:
            bg_color_img = (
                np.full_like(shadowfree_img.flatten().reshape(h * w, c), bg_color)
                .reshape(h, w, c)
                .astype(np.float64)
                / self.max_value
            )
            bg_color = bg_color.reshape(3).astype(np.float64) / self.max_value

        if self.transform is not None:
            images = [shadow_img, shadowfree_img]

            if self.use_bg_color:
                images.append(bg_color_img)

            images = np.concatenate(
                images,
                axis=2,
            )

            masks = shadow_mask[:, :, np.newaxis]

            ret = self.transform(image=images, mask=masks)
            imgs = ret["image"]
            shadow_img = imgs[0:c, :, :]
            shadowfree_img = imgs[c : c * 2, :, :]

            if self.use_bg_color:
                bg_color_img = imgs[c * 2 : c * 3, :, :]

            shadow_mask = ret["mask"][:, :, 0].unsqueeze(0)

        sample["path"] = shadow_img_path
        sample["input_shadow_img"] = shadow_img.float()
        sample["gt_shadowfree_img"] = shadowfree_img.float()
        sample["shadow"] = shadow_mask.float()

        if self.use_bg_color:
            sample["bg_color"] = torch.Tensor(bg_color).float()
            sample["color_img"] = bg_color_img.float()

        return sample

    def get_params(self):
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        hue = random.uniform(-0.5, 0.5)

        order = [0, 1, 2, 3]
        random.shuffle(order)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
            "order": order,
        }

    def apply(
        self,
        img,
        brightness=1.0,
        contrast=1.0,
        saturation=1.0,
        hue=0,
        order=[0, 1, 2, 3],
    ):
        params = [brightness, contrast, saturation, hue]
        for i in order:
            img = self.color_transforms[i](img, params[i])
        return img


def clip(img: np.ndarray, dtype: np.dtype, maxval: float) -> np.ndarray:
    return np.clip(img, 0, maxval).astype(dtype)


def adjust_brightness(img, factor):
    if factor == 0:
        return np.zeros_like(img)
    elif factor == 1:
        return img

    lut = np.arange(0, 256) * factor
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return cv2.LUT(img, lut)


def adjust_contrast(img, factor):
    mean = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()

    if factor == 0:
        if img.dtype != np.float32:
            mean = int(mean + 0.5)
        return np.full_like(img, mean, dtype=img.dtype)

    lut = np.arange(0, 256) * factor
    lut = lut + mean * (1 - factor)
    lut = clip(lut, img.dtype, 255)

    return cv2.LUT(img, lut)


def adjust_saturation(img, factor, gamma=0):
    if factor == 1:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if factor == 0:
        return gray

    result = cv2.addWeighted(img, factor, gray, 1 - factor, gamma=gamma)
    return result


def hue(img, factor):
    if factor == 0:
        return img

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
    img[..., 0] = cv2.LUT(img[..., 0], lut)

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
