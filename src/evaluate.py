import argparse
import datetime
import os
import time
from logging import DEBUG, INFO, basicConfig, getLogger

import cv2
import numpy as np
import torch
import torch.optim as optim
from albumentations import RandomResizedCrop  # noqa
from albumentations import Compose, HorizontalFlip, Resize
from albumentations.pytorch import ToTensorV2
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm

import wandb
from libs.checkpoint import (
    resume_generator_discriminator,
    save_checkpoint_generator_discriminator,
)
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.helper import evaluate, train
from libs.logger import TrainLogger
from libs.loss_fn import get_criterion
from libs.metric import calc_psnr, calc_rmse_lab, calc_ssim
from libs.models import get_model
from libs.seed import set_seed

logger = getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        train a network for document shadow removal.
        """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument("--test_data", type=str, help="name of testing dataset")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # save log files in the directory which contains config file.
    result_path = os.path.dirname(args.config)
    experiment_name = os.path.basename(result_path)

    # fix seed
    set_seed()

    # configuration
    config = get_config(args.config)

    # cpu or cuda
    device = get_device(allow_only_gpu=False)

    # Dataloader
    test_transform = Compose(
        [
            Resize(config.height, config.width),
            ToTensorV2(),
        ]
    )

    test_loader = get_dataloader(
        args.test_data,
        config.model,
        "test",
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        transform=test_transform,
    )

    # * define a model
    nets = get_model(
        config.model,
        pretrained=True,
        in_channels=3,
        weight=config.pretrained_weight,
        config=config.pretrained_config,
    )
    net, _ = nets[0], nets[1]

    # * send the model to cuda/cpu
    net.to(device)
    net.eval()

    if not os.path.exists(f"{result_path}/{args.test_data}"):
        os.mkdir(f"{result_path}/{args.test_data}")

    rmses = []
    psnrs = []
    ssims = []

    for i, sample in tqdm(enumerate(test_loader)):
        input = sample["input_shadow_img"].to(device, non_blocking=True)
        gt = sample["gt_shadowfree_img"].to(device, non_blocking=True)
        file_name = sample["path"][0].split("/")[-1]

        pred, shadow = net(input.to(device))

        input = input.detach().to("cpu").numpy()
        gt = gt.detach().to("cpu").numpy()
        pred = pred.detach().to("cpu").numpy()
        shadow = shadow.detach().to("cpu").numpy()

        rmses.append(calc_rmse_lab(list(gt), list(pred)))
        psnrs.append(calc_psnr(list(gt), list(pred)))
        ssims.append(calc_ssim(list(gt), list(pred)))

        pred = ((pred[0, :, :, :].transpose(1, 2, 0)) * 255).astype(np.uint8)

        cv2.imwrite(
            f"{result_path}/{args.test_data}/{file_name}",
            cv2.cvtColor(pred, cv2.COLOR_BGR2RGB),
        )

    with open(f"{result_path}/{args.test_data}/evaluation_score.txt", mode="w") as f:
        f.write(f"RMSE: {np.mean(rmses):.2f}\n")
        f.write(f"PSNR: {np.mean(psnrs):.2f}\n")
        f.write(f"SSIM: {np.mean(ssims):.4f}\n")


if __name__ == "__main__":
    main()
