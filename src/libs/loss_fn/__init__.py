from logging import getLogger
from typing import Any, List, Optional, Union

import torch.nn as nn

from .perceptual_loss import PerceptualLoss

__all__ = ["get_criterion"]

loss_function_names = [
    "l1",
    "bce",
    "perceptual",
]

logger = getLogger(__name__)


def get_criterion(
    loss_function: Optional[str] = None, device: Optional[str] = None, **kwargs: Any
) -> Union[nn.Module, List[nn.Module]]:
    name = loss_function.lower()
    if name not in loss_function_names:
        message = """
            There is no model appropriate to your choice.
            You have to choose l1, bce, focal, dice and perceptual
            as a loss function.
            """
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a loss function.".format(name))
    criterion: Union[nn.Module, List[nn.Module]]

    if loss_function == "l1":
        criterion = nn.L1Loss().to(device)

    elif loss_function == "bce":
        criterion = nn.BCEWithLogitsLoss().to(device)

    elif loss_function == "perceptual":
        criterion = PerceptualLoss().to(device)

    else:
        criterion = nn.L1Loss().to(device)

    return criterion
