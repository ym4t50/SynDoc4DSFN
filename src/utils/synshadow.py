# This module mainly based on SynShadow (Inoue+, TCSVT2021)
# https://github.com/naoto0804/SynShadow


import random
from typing import Tuple

import numpy as np
import torch


def sample_darkening_params(
    xmin_at_y_0: float,
    xmax_at_y_0: float,
    ymin_at_x_255: float,
    ymax_at_x_255: float,
    slope_max: float,
) -> Tuple[float, float, float, float, float]:
    """
    Sample two points, and generate its slope
    """
    assert 0.0 <= xmin_at_y_0 <= 1.0
    assert 0.0 <= xmax_at_y_0 <= 1.0
    assert 0.0 <= ymin_at_x_255 <= 1.0
    assert 0.0 <= ymax_at_x_255 <= 1.0
    assert 1.0 <= slope_max

    while True:
        x1 = random.uniform(xmin_at_y_0, xmax_at_y_0)  # l1
        y1 = 0.0
        x2 = 1.0
        y2 = random.uniform(ymin_at_x_255, ymax_at_x_255)  # s1
        a = (y2 - y1) / (x2 - x1)
        if a < 1.0:
            break
    return a, x1, y1, x2, y2


def sample_darkening_params_ranged(
    xmin_at_y_0: float,
    xmax_at_y_0: float,
    ymin_at_x_255: float,
    ymax_at_x_255: float,
    slope_max: float,
) -> Tuple[float, float, float, float, float]:
    """
    Sample two points, and generate its slope
    """

    while True:
        x1 = random.uniform(xmin_at_y_0, xmax_at_y_0)  # l1
        y1 = 0.0
        x2 = 1.0
        y2 = random.uniform(ymin_at_x_255, ymax_at_x_255)  # s1
        a = (y2 - y1) / (x2 - x1)
        if a < 1.0:
            break
    return a, x1, y1, x2, y2


def darken(
    x: torch.Tensor,
    xmin_at_y_0: float = 0.1,
    xmax_at_y_0: float = 0.125,
    ymin_at_x_255: float = 0.9,
    ymax_at_x_255: float = 0.1,
    x_turb_mu: float = 0.0,
    x_turb_sigma: float = 0.03,
    slope_max: float = 1.0,
) -> torch.Tensor:
    assert x.ndim == 3

    a, x1, y1, _, _ = sample_darkening_params_ranged(
        xmin_at_y_0, xmax_at_y_0, ymin_at_x_255, ymax_at_x_255, slope_max
    )

    x1_G = x1
    mu, sigma = x_turb_mu, x_turb_sigma
    x1_R = x1_G + np.random.normal(mu, sigma)
    x1_B = x1_G + np.random.normal(mu, sigma)

    b = [y1 - a * x1_R, y1 - a * x1_G, y1 - a * x1_B]
    b = torch.Tensor(b).view(3, 1, 1)
    b = b.repeat(1, x.size(1), x.size(2))

    y = a * x + b
    y = torch.clamp(y, min=0.0, max=1.0)
    return y


def generate(shadow_free: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert shadow_free.ndim == 3 and mask.ndim == 3
    assert shadow_free.size()[1:] == mask.size()[1:]
    assert 0.0 <= shadow_free.min() and shadow_free.max() <= 1.0
    assert 0.0 <= mask.min() and mask.max() <= 1.0
    darker_image = darken(shadow_free)
    shadow = mask * darker_image
    shadow += (1 - mask) * shadow_free
    return shadow
