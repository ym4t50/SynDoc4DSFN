import argparse
import datetime
import os
import time
from logging import DEBUG, INFO, basicConfig, getLogger

import torch
import torch.optim as optim
import wandb
from albumentations import RandomResizedCrop  # noqa
from albumentations import Compose, HorizontalFlip, Resize
from albumentations.pytorch import ToTensorV2
from timm.scheduler.cosine_lr import CosineLRScheduler

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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Add --use_wandb option if you want to use wandb.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add --debug option if you want to see debug-level logs.",
    )
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

    # setting logger configuration
    logname = os.path.join(result_path, f"{datetime.datetime.now():%Y-%m-%d}_train.log")
    basicConfig(
        level=DEBUG if args.debug else INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=logname,
    )

    # fix seed
    set_seed()

    # configuration
    config = get_config(args.config)

    # cpu or cuda
    device = get_device(allow_only_gpu=False)

    # Dataloader
    train_transform = Compose(
        [
            RandomResizedCrop(config.height, config.width),
            HorizontalFlip(),
            ToTensorV2(),
        ]
    )

    val_transform = Compose(
        [
            Resize(config.height, config.width),
            ToTensorV2(),
        ]
    )

    train_loader = get_dataloader(
        config.dataset_name,
        config.model,
        "train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        transform=train_transform,
        sampling=config.sampling,
        finetuning=config.pretrained,
    )

    val_loader = get_dataloader(
        config.dataset_name,
        config.model,
        "val",
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        transform=val_transform,
        sampling=1,
    )

    # * define a model
    nets = get_model(config.model, pretrained=config.pretrained, in_channels=3, weight=config.pretrained_weight, config=config.pretrained_config)
    generator, discriminator = nets[0], nets[1]

    # * send the model to cuda/cpu
    generator.to(device)
    discriminator.to(device)

    generator = torch.nn.DataParallel(generator)
    discriminator = torch.nn.DataParallel(discriminator)

    optimizerG = optim.Adam(
        generator.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
    )
    optimizerD = optim.Adam(
        discriminator.parameters(), lr=config.learning_rate, betas=config.betas
    )
    lambda_dict = {}
    for i, _lambda in enumerate(config.lambdas):
        lambda_dict[f"lambda{str(i)}"] = _lambda

    # keep training and validation log
    begin_epoch = 0
    best_val_rmse = float("inf")
    best_val_psnr = 0.0
    best_val_ssim = 0.0
    best_g_loss = float("inf")
    best_d_loss = float("inf")

    # resume if you want
    if args.resume:
        (
            begin_epoch,
            generator,
            discriminator,
            optimizerG,
            optimizerD,
            best_g_loss,
            best_d_loss,
        ) = resume_generator_discriminator(
            result_path, generator, discriminator, optimizerG, optimizerD
        )

    schedulerG = CosineLRScheduler(
        optimizerG,
        t_initial=config.max_epoch,
        lr_min=config.lr_finish,
        warmup_t=config.warmup_epoch,
        warmup_lr_init=config.lr_start,
        warmup_prefix=True,
    )
    schedulerD = CosineLRScheduler(
        optimizerD,
        t_initial=config.max_epoch,
        lr_min=config.lr_finish,
        warmup_t=config.warmup_epoch,
        warmup_lr_init=config.lr_start,
        warmup_prefix=True,
    )

    log_path = os.path.join(result_path, "log.csv")
    train_logger = TrainLogger(log_path, resume=args.resume)

    # criterion for loss
    criterions = {}
    loss_types = ["shadow_free", "shadow"]
    loss_configs = [config.loss_functions_free, config.loss_functions_shadow]
    for loss_type, loss_functions in zip(loss_types, loss_configs):
        criterions[loss_type] = [
            get_criterion(loss_name, device) for loss_name in loss_functions
        ]

    # Weights and biases
    if args.use_wandb:
        wandb.init(
            name=experiment_name,
            config=config,
            project="YourProject",
            job_type="training",
        )
        # Magic
        wandb.watch(generator, log="all")
        wandb.watch(discriminator, log="all")

    # train and validate model
    logger.info("Start training.")

    for epoch in range(begin_epoch, config.max_epoch):
        # training
        start = time.time()
        (
            train_g_loss,
            train_d_loss,
            train_rmse,
            train_psnr,
            train_ssim,
            train_result_images,
            train_result_other_images,
        ) = train(
            train_loader,
            generator,
            discriminator,
            criterions,
            lambda_dict,
            optimizerG,
            optimizerD,
            schedulerG,
            schedulerD,
            epoch,
            device,
        )
        train_time = int(time.time() - start)

        # validation
        start = time.time()
        (
            val_g_loss,
            val_d_loss,
            val_rmse,
            val_psnr,
            val_ssim,
            val_result_images,
            val_result_other_images,
        ) = evaluate(
            val_loader, generator, discriminator, criterions, lambda_dict, device
        )
        val_time = int(time.time() - start)

        # save a model if top1 acc is higher than ever
        if best_g_loss > val_g_loss:
            best_g_loss = val_g_loss
            best_d_loss = val_d_loss
            torch.save(
                generator.state_dict(),
                os.path.join(result_path, "best_loss_g.prm"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(result_path, "best_loss_d.prm"),
            )

        if best_val_psnr < val_psnr:
            best_val_psnr = val_psnr
            torch.save(
                generator.state_dict(),
                os.path.join(result_path, "best_val_psnr_g.prm"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(result_path, "best_val_psnr_d.prm"),
            )

        if best_val_rmse > val_rmse:
            best_val_rmse = val_rmse
            torch.save(
                generator.state_dict(),
                os.path.join(result_path, "best_val_rmse_g.prm"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(result_path, "best_val_rmse_d.prm"),
            )

        if best_val_ssim < val_ssim:
            best_val_ssim = val_ssim
            torch.save(
                generator.state_dict(),
                os.path.join(result_path, "best_val_ssim_g.prm"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(result_path, "best_val_ssim_d.prm"),
            )

        # save checkpoint every epoch
        save_checkpoint_generator_discriminator(
            result_path,
            epoch,
            generator,
            discriminator,
            optimizerG,
            optimizerD,
            best_g_loss,
            best_d_loss,
            config.save_period,
        )

        # write logs to dataframe and csv file
        train_logger.update(
            epoch,
            optimizerG.param_groups[0]["lr"],
            optimizerD.param_groups[0]["lr"],
            train_time,
            train_g_loss,
            train_d_loss,
            val_time,
            val_g_loss,
            val_d_loss,
            train_rmse,
            train_psnr,
            train_ssim,
            val_rmse,
            val_psnr,
            val_ssim,
        )

        # save logs to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "lrG": optimizerG.param_groups[0]["lr"],
                    "lrD": optimizerD.param_groups[0]["lr"],
                    "train_time[sec]": train_time,
                    "train_g_loss": train_g_loss,
                    "train_d_loss": train_d_loss,
                    "val_time[sec]": val_time,
                    "val_g_loss": val_g_loss,
                    "val_d_loss": val_d_loss,
                    "train_rmse": train_rmse,
                    "train_psnr": train_psnr,
                    "train_ssim": train_ssim,
                    "train_image": wandb.Image(train_result_images, caption="train"),
                    "train_other_image": wandb.Image(
                        train_result_other_images, caption="train"
                    ),
                    "val_rmse": val_rmse,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "val_image": wandb.Image(val_result_images, caption="val"),
                    "val_other_image": wandb.Image(
                        val_result_other_images, caption="val"
                    ),
                },
                step=epoch,
            )

    # save models
    torch.save(generator.state_dict(), os.path.join(result_path, "g_final.prm"))
    torch.save(discriminator.state_dict(), os.path.join(result_path, "d_final.prm"))

    # delete checkpoint
    # os.remove(os.path.join(result_path, "g_checkpoint.pth"))
    # os.remove(os.path.join(result_path, "d_checkpoint.pth"))

    logger.info("Done")


if __name__ == "__main__":
    main()
