import argparse
import csv
import os
import random
import shutil
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TF
from sklearn import mixture
from sklearn.cluster import KMeans
from synshadow import generate
from tqdm import tqdm


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        rendering document images for shadow removal.
        """
    )
    parser.add_argument("--dataset_name", type=str, default="SynDocDS")
    parser.add_argument("--dir_name", type=str, default="output")
    parser.add_argument("--mode", type=str, default="kmeans")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument("-r", "--ratio", type=int, default=10)
    args = parser.parse_args()
    return args


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def alpha2gray(dir_name="output"):
    bit = 8
    shadow_path = sorted(glob(f"../{dir_name}/shadow_mask_raw/*.png"))
    save_path = f"../{dir_name}/shadow_matte"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

        for path in tqdm(shadow_path):
            file_name = path.split("/")[-1]
            matte = cv2.imread(path, -1).astype(np.float64)
            matte = matte[:, :, -1]
            matte = matte / (matte.max() - matte.min())
            matte = matte * (2**bit - 1)
            matte = np.clip(matte, a_min=0, a_max=(2**bit - 1)).astype(
                np.uint8
            )  # bit
            cv2.imwrite(os.path.join(save_path, file_name), matte)


def augment_shadow(
    img: np.ndarray, shadow: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    t = TF.ToTensor()
    img_tensor = t(img)
    shadow_tensor = t(shadow)
    shadowed = generate(img_tensor, shadow_tensor * scale)
    shadowed = (shadowed.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(
        np.uint8
    )
    return shadowed


def enrich_shadow_image(dir_name="output", ratio=10):
    shadowfree_paths = sorted(glob(f"../{dir_name}/shadowfree_image/*"))
    shadow_paths = sorted(glob(f"../{dir_name}/shadow_image_org/*"))
    shadow_mask_paths = sorted(glob(f"../{dir_name}/shadow_matte/*"))

    if not os.path.exists(f"../{dir_name}/shadow_image/"):
        os.mkdir(f"../{dir_name}/shadow_image")

    for shadowfree_path, shadow_path, shadow_mask_path in tqdm(
        zip(shadowfree_paths, shadow_paths, shadow_mask_paths)
    ):
        filename = shadowfree_path.split("/")[-1]
        filename, ext = filename.split(".")

        shadow_img = cv2.imread(shadow_path)
        cv2.imwrite(
            f"../{dir_name}/shadow_image/" + filename + f"_{str(0).zfill(2)}." + ext,
            shadow_img,
        )

        shadowfree_img = cv2.imread(shadowfree_path).astype(np.float64) / 255
        shadow_mask = cv2.imread(shadow_mask_path).astype(np.float64) / 255

        for i in range(1, ratio):
            synthesized_shadow_img = augment_shadow(
                shadowfree_img, shadow_mask, scale=1.0
            )
            cv2.imwrite(
                f"../{dir_name}/shadow_image/"
                + filename
                + f"_{str(i).zfill(2)}."
                + ext,
                synthesized_shadow_img,
            )


def get_average_color(x):
    b, g, r = x[:, 0], x[:, 1], x[:, 2]

    return np.array([np.mean(b), np.mean(g), np.mean(r)])


def get_background_color(dir_name="output", mode="kmeans"):
    shadowfree_path = sorted(glob(f"../{dir_name}/shadowfree_image/*.png"))
    save_path = f"../{dir_name}/background_color_csv"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for path in tqdm(shadowfree_path):
        file_name = path.split("/")[-1]

        x = cv2.imread(path)
        x = cv2.resize(x, (512, 512))
        h, w, c = x.shape
        x = x.flatten().reshape(h * w, c)

        if mode == "gmm":
            model = mixture.GaussianMixture(n_components=2, covariance_type="full")
            model.fit(x)
        elif mode == "kmeans":
            model = KMeans(n_clusters=2)
            model.fit(x)

        cls = model.predict(x.flatten().reshape(h * w, c))
        cls0_colors = x[cls == 0]
        cls1_colors = x[cls == 1]

        cls0_avg_color = get_average_color(cls0_colors)
        cls1_avg_color = get_average_color(cls1_colors)

        if np.sum(cls0_avg_color) >= np.sum(cls1_avg_color):
            background_color = cls0_avg_color
        else:
            background_color = cls1_avg_color

        with open(os.path.join(save_path, file_name.split(".")[0] + ".csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(background_color)


def make_csv(dir_name="output", dataset_name="SynDocDS", ratio=10):
    shutil.move(f"../{dir_name}", f"../datasets/{dataset_name}")
    if not os.path.exists(f"../datasets/csv/{dataset_name}"):
        os.mkdir(f"../datasets/csv/{dataset_name}")

    data_types = [
        "shadow_image",
        "shadowfree_image",
        "shadow_matte",
        "background_image",
        "background_color_csv",
    ]
    path_dict = {}

    for data_type in data_types:
        if data_type == "shadow_image":
            path_dict[data_type] = sorted(
                glob(f"../datasets/{dataset_name}/{data_type}/*")
            )
        else:
            path_dict[data_type] = sorted(
                glob(f"../datasets/{dataset_name}/{data_type}/*") * ratio
            )

    l = len(path_dict["shadow_image"])

    pnum = 0
    for phase, num in zip(
        ["train", "val", "test"], [int(l * 0.8), int(l * 0.1), int(l * 0.1)]
    ):
        df = pd.DataFrame()

        for data_type in data_types:
            paths = []
            paths += path_dict[data_type][pnum : pnum + num]
            df[data_type] = paths

        df.to_csv(f"../datasets/csv/{dataset_name}/{phase}.csv", index_label=None)
        pnum += num


if __name__ == "__main__":

    args = get_arguments()
    set_seed(args.seed)

    dir_name = args.dir_name
    dataset_name = args.dataset_name
    mode = args.mode
    ratio = args.ratio

    alpha2gray(dir_name)
    get_background_color(dir_name, mode=mode)
    enrich_shadow_image(dir_name, ratio)
    make_csv(dir_name, dataset_name, ratio)
