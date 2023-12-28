# from self_attention_cv import AxialAttentionBlock

import math
from collections import OrderedDict
from typing import Any, Callable
import numpy as np

import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

def fix_model_state_dict(state_dict) -> OrderedDict:
    """
    remove 'module.' of dataparallel
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def weights_init(init_type="gaussian") -> Callable:
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            if init_type == "gaussian":
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == "default":
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class Cvi(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        before: str = None,
        after: str = None,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.after: Any[Callable]
        self.before: Any[Callable]
        self.conv.apply(weights_init("gaussian"))
        if after == "BN":
            self.after = nn.BatchNorm2d(out_channels)
        elif after == "Tanh":
            self.after = torch.tanh
        elif after == "sigmoid":
            self.after = torch.sigmoid

        if before == "ReLU":
            self.before = nn.ReLU(inplace=True)
        elif before == "LReLU":
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if hasattr(self, "before"):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, "after"):
            x = self.after(x)

        return x


class CvTi(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        before: str = None,
        after: str = None,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super(CvTi, self).__init__()
        self.after: Any[Callable]
        self.before: Any[Callable]
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.conv.apply(weights_init("gaussian"))
        if after == "BN":
            self.after = nn.BatchNorm2d(out_channels)
        elif after == "Tanh":
            self.after = torch.tanh
        elif after == "sigmoid":
            self.after = torch.sigmoid

        if before == "ReLU":
            self.before = nn.ReLU(inplace=True)
        elif before == "LReLU":
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if hasattr(self, "before"):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, "after"):
            x = self.after(x)

        return x


class AdaptiveInstanceNorm2d(nn.Module):
    """Adaptive instance normalization"""

    """Specifically, This is not AdaIN"""

    def __init__(self, num_feat, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bn = nn.InstanceNorm2d(num_feat, eps, momentum, affine)
        self.a = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)

class ConvBlock(nn.Module):

    """Convolution head"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        norm_layer: nn.Module = AdaptiveInstanceNorm2d,
    ):
        super().__init__()
        convblk = []

        convblk.extend(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            ]
        )

        self.convblk = nn.Sequential(*convblk)
        self.init_weights(self.convblk)

    def identity_init(self, shape):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[2] // 2, shape[3] // 2
        for i in range(np.minimum(shape[0], shape[1])):
            array[i, i, cx, cy] = 1

        return array

    def init_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                weights = self.identity_init(m.weight.shape)
                with torch.no_grad():
                    m.weight.copy_(torch.from_numpy(weights).float())
                torch.nn.init.zeros_(m.bias)

    def forward(self, *inputs):
        return self.convblk(inputs[0])

class gConv(nn.Module):
    def __init__(self, dim, kernel_size=5, norm_layer: nn.Module = nn.BatchNorm2d):
        super(gConv, self).__init__()

        self.kernel_size = kernel_size

        self.norm = norm_layer(dim)

        self.point_depthwise = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                padding_mode="reflect",
            ),
        )

        self.pointwise_act = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1), nn.Sigmoid()
        )

        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1)

        self.apply(self.weights_init)

    def weights_init(self, m):
        """conv2d Init"""
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.norm(x)
        out = self.point_depthwise(x) * self.pointwise_act(x)
        out = self.conv1x1(out)
        out = x + out
        return out


class gdConv(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        norm_layer: nn.Module = AdaptiveInstanceNorm2d,
    ):
        super(gdConv, self).__init__()

        self.kernel_size = kernel_size

        self.norm = norm_layer(dim)

        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            padding=padding,
            dilation=dilation,
            padding_mode="reflect",
        )

        self.pointwise_act = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1), nn.Sigmoid()
        )

        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1)

        self.weights_init(self.pointwise)
        self.weights_init(self.pointwise_act)
        self.weights_init(self.conv1x1)
        self.weights_init(self.depthwise)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def weights_init(self, m):
        """conv2d Init"""
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.norm(x)
        out = self.depthwise(self.pointwise(x)) * self.pointwise_act(x)  # out
        out = self.conv1x1(out)
        out = x + self.act(out)
        return out


class SegformerBackBone(nn.Module):
    def __init__(self, in_channels: int = 3, input_size=512) -> None:
        super(SegformerBackBone, self).__init__()

        pretrained_model_name = "nvidia/mit-b3"
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name
        )
        self.segformer.decode_head.linear_fuse = nn.Identity()
        self.segformer.decode_head.batch_norm = nn.Identity()
        self.segformer.decode_head.activation = nn.Identity()
        self.segformer.decode_head.dropout = nn.Identity()
        self.segformer.decode_head.classifier = nn.Identity()

        self.upsample = nn.Upsample(size=input_size, mode="bilinear")

        # fix
        for name, param in self.segformer.segformer.named_parameters():
            # print(name)
            param.requires_grad = False

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        input = (input - self.mean.to(device)) / self.std.to(device)
        feature = self.segformer(input).logits
        feature = self.upsample(feature)

        return feature


class MultiFusionBlock(nn.Module):
    def __init__(self, dim, reduction=8, in_num=2):
        super(MultiFusionBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim * in_num, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)
        self.in_num = in_num

    def forward(self, inputs):
        xsum = torch.sum(torch.stack(inputs, dim=0), dim=0)
        b, c, _, _ = xsum.size()
        attn = self.avg_pool(xsum).view(b, c)
        attn = self.fc(attn).view(b, self.in_num, c, 1, 1)
        attn = self.softmax(attn)

        outs = [inputs[i] * attn[:, i, :, :, :] for i in range(self.in_num)]
        out = torch.sum(torch.stack(outs, dim=0), dim=0)

        return out
