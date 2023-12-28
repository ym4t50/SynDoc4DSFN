from typing import Any, Optional

import torch
import torch.nn as nn

from .cam import GradCAM
from .modules import (
    ConvBlock,
    Cvi,
    CvTi,
    MultiFusionBlock,
    SegformerBackBone,
    fix_model_state_dict,
    gdConv,
)


class BENet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, nLayers=64) -> None:
        super(BENet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, nLayers, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nLayers, nLayers, kernel_size=3, padding=1),
            nn.Conv2d(nLayers, nLayers * 2, kernel_size=3, padding=1),
            nn.Conv2d(nLayers * 2, nLayers * 2, kernel_size=3, padding=1),
        )
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(nLayers * 2, out_channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels: int = 7, out_channels: int = 3) -> None:
        super(Generator, self).__init__()

        self.Cv0 = Cvi(in_channels, 64)

        self.Cv1 = Cvi(64, 128, before="LReLU", after="BN")

        self.Cv2 = Cvi(128, 256, before="LReLU", after="BN")

        self.Cv3 = Cvi(256, 512, before="LReLU", after="BN")

        self.Cv4_1 = Cvi(512, 512, before="LReLU", after="BN")
        self.Cv4_2 = Cvi(512, 512, before="LReLU", after="BN")
        self.Cv4_3 = Cvi(512, 512, before="LReLU", after="BN")

        self.Cv5 = Cvi(512, 512, before="LReLU")

        self.CvT6 = CvTi(512, 512, before="ReLU", after="BN")

        self.CvT7_1 = CvTi(1024, 512, before="ReLU", after="BN")
        self.CvT7_2 = CvTi(1024, 512, before="ReLU", after="BN")
        self.CvT7_3 = CvTi(1024, 512, before="ReLU", after="BN")

        self.CvT8 = CvTi(1024, 256, before="ReLU", after="BN")

        self.CvT9 = CvTi(512, 128, before="ReLU", after="BN")

        self.CvT10 = CvTi(256, 64, before="ReLU", after="BN")

        self.CvT11 = CvTi(128, out_channels, before="ReLU", after="sigmoid")  # tanh

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # encoder
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        x4_1 = self.Cv4_1(x3)
        x4_2 = self.Cv4_2(x4_1)
        x4_3 = self.Cv4_3(x4_2)
        x5 = self.Cv5(x4_3)

        # decoder
        x6 = self.CvT6(x5)

        cat1_1 = torch.cat([x6, x4_3], dim=1)
        x7_1 = self.CvT7_1(cat1_1)
        cat1_2 = torch.cat([x7_1, x4_2], dim=1)
        x7_2 = self.CvT7_2(cat1_2)
        cat1_3 = torch.cat([x7_2, x4_1], dim=1)
        x7_3 = self.CvT7_3(cat1_3)

        cat2 = torch.cat([x7_3, x3], dim=1)
        x8 = self.CvT8(cat2)

        cat3 = torch.cat([x8, x2], dim=1)
        x9 = self.CvT9(cat3)

        cat4 = torch.cat([x9, x1], dim=1)
        x10 = self.CvT10(cat4)

        cat5 = torch.cat([x10, x0], dim=1)
        out = self.CvT11(cat5)

        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=6) -> None:
        super(Discriminator, self).__init__()

        self.Cv0 = Cvi(in_channels, 64)

        self.Cv1 = Cvi(64, 128, before="LReLU", after="BN")

        self.Cv2 = Cvi(128, 256, before="LReLU", after="BN")

        self.Cv3 = Cvi(256, 512, before="LReLU", after="BN")

        self.Cv4 = Cvi(512, 1, before="LReLU", after="sigmoid")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        out = self.Cv4(x3)

        return out


class DSFN(nn.Module):
    def __init__(
        self, in_channels=3, input_size=512, branches=["feature", "attention"]
    ):
        """Initialization"""
        super().__init__()

        self.transformerbackbone = SegformerBackBone(
            in_channels=in_channels, input_size=512
        )

        nbLayers = 64
        self.branches = branches

        self.conv1x1_1 = ConvBlock(3072 + in_channels, nbLayers, 3, 1, 1)  # 1, 0, 1
        self.conv1x1_2 = gdConv(nbLayers, 3, 1, 1)
        self.conv1 = gdConv(dim=nbLayers, kernel_size=3, padding=1, dilation=1)  # net1

        for suffix in self.branches:
            setattr(self, "agg1_" + suffix, MultiFusionBlock(dim=nbLayers, in_num=2))
            setattr(self, "convagg1_" + suffix, ConvBlock(nbLayers, nbLayers, 3, 1, 1))
        self.dualfusion1 = MultiFusionBlock(dim=nbLayers, in_num=2)

        self.conv2 = gdConv(nbLayers, 3, 2, 2)
        self.conv3 = gdConv(nbLayers, 3, 4, 4)
        for suffix in self.branches:
            setattr(self, "agg2_" + suffix, MultiFusionBlock(dim=nbLayers, in_num=3))
            setattr(self, "convagg2_" + suffix, ConvBlock(nbLayers, nbLayers, 3, 1, 1))
        self.dualfusion2 = MultiFusionBlock(dim=nbLayers, in_num=2)

        self.conv4 = gdConv(nbLayers, 3, 8, 8)
        self.conv5 = gdConv(nbLayers, 3, 16, 16)
        for suffix in self.branches:
            setattr(
                self, "agg3_" + suffix, MultiFusionBlock(dim=nbLayers, in_num=3)
            )
            setattr(self, "convagg3_" + suffix, ConvBlock(nbLayers, nbLayers, 3, 1, 1))
        self.dualfusion3 = MultiFusionBlock(dim=nbLayers, in_num=2)

        self.conv6 = gdConv(nbLayers, 3, 32, 32)
        self.conv7 = gdConv(nbLayers, 3, 64, 64)
        for suffix in self.branches:
            setattr(self, "agg4_" + suffix, MultiFusionBlock(dim=nbLayers, in_num=4))
            setattr(self, "convagg4_" + suffix, ConvBlock(nbLayers, nbLayers, 3, 1, 1))

        kernels_strides = [(2**k, 2**k) for k in range(2, 6)]

        for suffix in self.branches:
            setattr(self, "conv8_" + suffix, ConvBlock(nbLayers, nbLayers, 3, 1, 1))
            setattr(
                self,
                "spp_" + suffix,
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.AvgPool2d(kernel_size=ks, stride=ks, padding=0),
                            nn.Conv2d(64, 64, 1),
                            nn.Upsample(size=input_size, mode="bicubic"),
                        )
                        for _, ks in enumerate(kernels_strides)
                    ]
                ),
            )

            if suffix == "feature":
                setattr(self, "conv9_" + suffix, ConvBlock(nbLayers * 5, 3, 1, 0, 1))
            else:
                setattr(self, "conv9_" + suffix, ConvBlock(nbLayers * 5, 1, 1, 0, 1))

            self.weights_init(getattr(self, "conv9_" + suffix))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.transformerbackbone(x)
        x = torch.cat([features, x], dim=1)

        x = self.conv1x1_1(x)

        x0 = self.conv1x1_2(x)
        x1 = self.conv1(x0)
        cat1 = [x1, x0]

        aggs1 = []
        for suffix in self.branches:
            agg1 = getattr(self, "agg1_" + suffix)(cat1)
            agg1 = getattr(self, "convagg1_" + suffix)(agg1)
            aggs1.append(agg1)

        agg1_to_conv = self.dualfusion1([aggs1[0], aggs1[1]])
        x2 = self.conv2(agg1_to_conv)
        x3 = self.conv3(x2)

        aggs2 = []
        for i, suffix in enumerate(self.branches):
            cat2 = [aggs1[i], x3, x2]
            agg2 = getattr(self, "agg2_" + suffix)(cat2)
            agg2 = getattr(self, "convagg2_" + suffix)(agg2)
            aggs2.append(agg2)

        agg2_to_conv = self.dualfusion2([aggs2[0], aggs2[1]])
        x4 = self.conv4(agg2_to_conv)
        x5 = self.conv5(x4)

        aggs3 = []
        for i, suffix in enumerate(self.branches):
            cat3 = [aggs2[i], x5, x4]
            agg3 = getattr(self, "agg3_" + suffix)(cat3)
            agg3 = getattr(self, "convagg3_" + suffix)(agg3)
            aggs3.append(agg3)

        agg3_to_conv = self.dualfusion3([aggs3[0], aggs3[1]])
        x6 = self.conv4(agg3_to_conv)
        x7 = self.conv5(x6)

        aggs4 = []
        for i, suffix in enumerate(self.branches):
            cat4 = [aggs3[i], aggs2[i], x7, x6]
            agg4 = getattr(self, "agg4_" + suffix)(cat4)
            agg4 = getattr(self, "convagg4_" + suffix)(agg4)
            aggs4.append(agg4)

        spp_outs = []
        for i, suffix in enumerate(self.branches):
            x = getattr(self, "conv8_" + suffix)(aggs4[i])
            xs = [spp(x) for spp in getattr(self, "spp_" + suffix)]
            x = torch.cat(xs + [x], dim=1)
            spp_outs.append(x)

        feature = self.sigmoid(
            self.conv9_feature(spp_outs[0] * self.sigmoid(spp_outs[1]))
        )
        attention = self.sigmoid(self.conv9_attention(spp_outs[1]))

        return feature, attention

    def weights_init(self, m):
        """conv2d Init"""
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)


def benet(pretrained: bool = False, **kwargs: Any) -> BENet:
    model = BENet(**kwargs)
    if pretrained:
        state_dict = torch.load(
            "./pretrained/benet/pretrained_benet.prm"
        ) 
        model.load_state_dict(fix_model_state_dict(state_dict))
    return model


def cam_benet(pretrained: bool = False, **kwargs: Any) -> GradCAM:
    model = BENet(**kwargs)
    if pretrained:
        state_dict = torch.load(
            "./pretrained/benet/pretrained_benet.prm"
        )
        model.load_state_dict(fix_model_state_dict(state_dict))
    model.eval()
    target_layer = model.features[3]
    wrapped_model = GradCAM(model, target_layer)
    return wrapped_model


def generator(pretrained: bool = False, weight: str = "best_loss_g.prm", config: Optional[str] = None, **kwargs: Any) -> Generator:
    model = Generator(**kwargs)
    if pretrained:
        state_dict = torch.load(f"./configs/{config}/{weight}")
        model.load_state_dict(fix_model_state_dict(state_dict))
    return model


def discriminator(
    pretrained: bool = False, weight: str = "best_loss_d.prm", config: Optional[str] = None, **kwargs: Any
) -> Discriminator:
    model = Discriminator(**kwargs)
    if pretrained:
        state_dict = torch.load(f"./configs/{config}/{weight}")
        model.load_state_dict(fix_model_state_dict(state_dict))
    return model


def dsfn(pretrained: bool = False, weight: str = "best_loss_g.prm", config: Optional[str] = None, **kwargs: Any) -> DSFN():
    model = DSFN(**kwargs)
    if pretrained:
        state_dict = torch.load(
            f"./configs/{config}/{weight}"
        )
        model.load_state_dict(fix_model_state_dict(state_dict))
    return model
