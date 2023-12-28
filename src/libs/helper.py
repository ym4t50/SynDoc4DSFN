import time
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timm.scheduler.scheduler import Scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .meter import AverageMeter, ProgressMeter
from .metric import calc_psnr, calc_rmse_lab, calc_ssim
from .visualize_grid import make_grid, make_grid_gray

__all__ = ["train", "evaluate"]

logger = getLogger(__name__)


def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def do_one_iteration(
    sample: Dict[str, Any],
    generator: nn.Module,
    discriminator: nn.Module,
    criterions: Dict,
    device: str,
    iter_type: str,
    lambda_dict: Dict,
    optimizerG: Optional[optim.Optimizer] = None,
    optimizerD: Optional[optim.Optimizer] = None,
    schedulerG: Optional[Scheduler] = None,
    schedulerD: Optional[Scheduler] = None,
    epoch: Optional[int] = None,
) -> Tuple[
    int,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    if iter_type not in ["train", "evaluate"]:
        message = "iter_type must be either 'train' or 'evaluate'"
        logger.error(message)
        raise ValueError(message)

    if iter_type == "train" and (optimizerG is None or optimizerD is None):
        message = "optimizer must be set during training."
        logger.error(message)
        raise ValueError(message)

    Tensor = (
        torch.cuda.FloatTensor  # type: ignore
        if device != torch.device("cpu")
        else torch.FloatTensor
    )

    input = sample["input_shadow_img"].to(device)
    gt = sample["gt_shadowfree_img"].to(device)

    shadow = sample["shadow"].to(device)

    batch_size, c, h, w = input.shape

    # compute output and loss
    # train discriminator
    if iter_type == "train" and optimizerD is not None:
        set_requires_grad([discriminator], True)  # enable backprop
        optimizerD.zero_grad()

    out_shadowfree, out_shadow = generator(input.to(device))

    fake = torch.cat([input, out_shadowfree], dim=1)
    real = torch.cat([input, gt], dim=1)
    out_D_fake = discriminator(fake.detach())

    out_D_real = discriminator(real.detach())

    label_D_fake = Variable(Tensor(np.zeros(out_D_fake.size())), requires_grad=True)
    label_D_real = Variable(Tensor(np.ones(out_D_fake.size())), requires_grad=True)

    loss_D_fake = criterions["shadow_free"][1](out_D_fake, label_D_fake)
    loss_D_real = criterions["shadow_free"][1](out_D_real, label_D_real)
    D_L_CGAN = loss_D_fake + loss_D_real

    D_loss = D_L_CGAN

    if iter_type == "train" and optimizerD is not None:
        D_loss.backward()
        optimizerD.step()
        schedulerD.step(epoch + 1)

    if iter_type == "train" and optimizerG is not None:
        set_requires_grad([discriminator], False)
        optimizerG.zero_grad()

    fake = torch.cat([input, out_shadowfree], dim=1)
    out_D_fake = discriminator(fake.detach())
    G_cGAN_Loss = criterions["shadow_free"][1](out_D_fake, label_D_real)
    G_shadow_Loss = criterions["shadow"][0](out_shadow, shadow)

    G_l1_Loss = criterions["shadow_free"][0](gt, out_shadowfree)
    G_percep_Loss = criterions["shadow_free"][2](gt, out_shadowfree)

    G_loss = (
        lambda_dict["lambda1"] * G_shadow_Loss
        + lambda_dict["lambda0"] * G_l1_Loss
        + lambda_dict["lambda0"] * G_cGAN_Loss
        + lambda_dict["lambda2"] * G_percep_Loss
    )

    if iter_type == "train" and optimizerG is not None:
        G_loss.backward()
        optimizerG.step()
        schedulerG.step(epoch + 1)

    input = input.detach().to("cpu").numpy()
    gt = gt.detach().to("cpu").numpy()
    shadow = shadow.detach().to("cpu").numpy()
    out_shadowfree = out_shadowfree.detach().to("cpu").numpy()
    out_shadow = out_shadow.detach().to("cpu").numpy()
    
    rmse_score = calc_rmse_lab(list(gt), list(out_shadowfree))
    psnr_score = calc_psnr(list(gt), list(out_shadowfree))
    ssim_score = calc_ssim(list(gt), list(out_shadowfree))

    return (
        batch_size,
        G_loss.item(),
        D_loss.item(),
        input,
        gt,
        out_shadowfree,
        shadow,
        out_shadow,
        rmse_score,
        psnr_score,
        ssim_score,
    )


def train(
    loader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    criterions: Dict,
    lambda_dict: Dict,
    optimizerG: optim.Optimizer,
    optimizerD: optim.Optimizer,
    schedulerG: Scheduler,
    schedulerD: Scheduler,
    epoch: int,
    device: str,
    interval_of_progress: int = 50,
) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    torch.cuda.empty_cache()
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    g_losses = AverageMeter("Loss", ":.4e")
    d_losses = AverageMeter("Loss", ":.4e")
    rmse_scores = AverageMeter("RMSE", ":.4e")
    psnr_scores = AverageMeter("PSNR", ":.4e")
    ssim_scores = AverageMeter("SSIM", ":.4e")

    progress = ProgressMeter(
        len(loader),
        [
            batch_time,
            data_time,
            g_losses,
            d_losses,
            rmse_scores,
            psnr_scores,
            ssim_scores,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # * keep predicted results and gts for calculate score
    inputs: List[np.ndarary] = []
    gts: List[np.ndarray] = []
    shadowfrees: List[np.ndarray] = []
    shadows: List[np.ndarray] = []
    pred_shadows: List[np.ndarray] = []

    # switch to train mode
    generator.train()
    discriminator.train()

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        (
            batch_size,
            g_loss,
            d_loss,
            input,
            gt,
            shadowfree,
            shadow,
            pred_shadow,
            rmse_score,
            psnr_score,
            ssim_score,
        ) = do_one_iteration(
            sample,
            generator,
            discriminator,
            criterions,
            device,
            "train",
            lambda_dict,
            optimizerG,
            optimizerD,
            schedulerG,
            schedulerD,
            epoch,
        )

        g_losses.update(g_loss, batch_size)
        d_losses.update(d_loss, batch_size)
        rmse_scores.update(rmse_score, batch_size)
        psnr_scores.update(psnr_score, batch_size)
        ssim_scores.update(ssim_score, batch_size)

        # save the ground truths and predictions in lists
        if len(inputs) < 5:
            inputs += list(input)
            gts += list(gt)
            shadowfrees += list(shadowfree)
            shadows += list(shadow.astype(np.float32))
            pred_shadows += list(pred_shadow.astype(np.float32))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % interval_of_progress == 0:
            progress.display(i)

    result_images = make_grid([inputs[:5], shadowfrees[:5], gts[:5]])

    result_shadow_images = make_grid_gray([shadows[:5], pred_shadows[:5]])

    return (
        g_losses.get_average(),
        d_losses.get_average(),
        rmse_scores.get_average(),
        psnr_scores.get_average(),
        ssim_scores.get_average(),
        result_images,
        result_shadow_images,
    )


def evaluate(
    loader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    criterions: Dict,
    lambda_dict: Dict,
    device: str,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    torch.cuda.empty_cache()
    g_losses = AverageMeter("Loss", ":.4e")
    d_losses = AverageMeter("Loss", ":.4e")
    rmse_scores = AverageMeter("RMSE", ":.4e")
    psnr_scores = AverageMeter("PSNR", ":.4e")
    ssim_scores = AverageMeter("SSIM", ":.4e")

    # * keep predicted results and gts for calculate Score
    inputs: List[np.ndarary] = []
    gts: List[np.ndarray] = []
    shadowfrees: List[np.ndarray] = []
    shadows: List[np.ndarray] = []
    pred_shadows: List[np.ndarray] = []
    # switch to evaluate mode
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for sample in loader:
            (
                batch_size,
                g_loss,
                d_loss,
                input,
                gt,
                shadowfree,
                shadow,
                pred_shadow,
                rmse_score,
                psnr_score,
                ssim_score,
            ) = do_one_iteration(
                sample,
                generator,
                discriminator,
                criterions,
                device,
                "evaluate",
                lambda_dict,
            )

            g_losses.update(g_loss, batch_size)
            d_losses.update(d_loss, batch_size)
            rmse_scores.update(rmse_score, batch_size)
            psnr_scores.update(psnr_score, batch_size)
            ssim_scores.update(ssim_score, batch_size)

            # save the ground truths and predictions in lists
            if len(inputs) < 5:
                inputs += list(input)
                gts += list(gt)
                shadowfrees += list(shadowfree)
                shadows += list(shadow.astype(np.float32))
                pred_shadows += list(pred_shadow.astype(np.float32))

    result_images = make_grid([inputs[:5], shadowfrees[:5], gts[:5]])

    result_shadow_images = make_grid_gray([shadows[:5], pred_shadows[:5]])

    return (
        g_losses.get_average(),
        d_losses.get_average(),
        rmse_scores.get_average(),
        psnr_scores.get_average(),
        ssim_scores.get_average(),
        result_images,
        result_shadow_images,
    )
