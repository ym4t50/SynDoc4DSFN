from logging import getLogger
from typing import List, Union, Optional

import torch.nn as nn

from . import models

__all__ = ["get_model"]

model_names = [
    "benet",
    "cam_benet",
    "srnet",
    "stcgan",
    "generator",
    "discriminator",
    "dsfn",
]

logger = getLogger(__name__)


def get_model(
    name: str, in_channels: int = 3, pretrained: bool = True, weight="best_loss", config: Optional[str] = None
) -> Union[nn.Module, List[nn.Module]]:
    name = name.lower()
    if name not in model_names:
        message = (
            "There is no model appropriate to your choice. "
            """
            You have to choose benet/cam_benet(BENet),
            srnet(SR-Net), stcgan(ST-CGAN), dsfn(DSFN) as a model.
            """
        )
        logger.error(message)
        raise ValueError(message)
    
    if pretrained is True and config is None:
        message = (
            "You have to choose pretrained config directory."
        )
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(name))

    if name == "srnet":
        generator = getattr(models, "generator")(
            pretrained=pretrained, weight=weight+"_g.prm", in_channels=in_channels, config=config
        )
        discriminator = getattr(models, "discriminator")(
            pretrained=pretrained, weight=weight+"_d.prm", config=config
        )
        model = [generator, discriminator]

    elif name == "stcgan":
        generator_1 = getattr(models, "generator")(
            pretrained=pretrained, in_channels=in_channels, out_channels=1, weight=weight+"_g1.prm", config=config
        )
        discriminator_1 = getattr(models, "discriminator")(
            pretrained=pretrained, in_channels=in_channels + 1, weight=weight+"_d1.prm", config=config
        )
        generator_2 = getattr(models, "generator")(
            pretrained=pretrained,
            in_channels=in_channels + 1,
            out_channels=3,
            weight=weight+"_g2.prm",
            config=config
        )
        discriminator_2 = getattr(models, "discriminator")(
            pretrained=pretrained, in_channels=in_channels + 3 + 1, weight=weight+"_d2.prm", config=config
        )
        model = [generator_1, discriminator_1, generator_2, discriminator_2]

    elif name == "benet" or name == "cam_benet":
        model = getattr(models, name)(in_channels=in_channels, pretrained=pretrained)

    elif name == "discriminator":
        model = getattr(models, name)(pretrained=pretrained, in_channels=in_channels, weight=weight+"_d.prm", config=config)

    else:
        generator = getattr(models, name)(
            pretrained=pretrained, in_channels=in_channels, config=config, weight=weight+"_g.prm"
        )
        discriminator = getattr(models, "discriminator")(
            pretrained=pretrained, config=config, weight=weight+"_d.prm"
        )
        model = [generator, discriminator]

    return model
