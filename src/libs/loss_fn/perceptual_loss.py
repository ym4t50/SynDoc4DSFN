import torch
import torch.nn as nn
import torchvision.models as models


class VGG16map(nn.Module):
    def __init__(self) -> None:
        super(VGG16map, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.layer1 = nn.Sequential(*vgg16.features[:5])
        self.layer2 = nn.Sequential(*vgg16.features[5:10])
        self.layer3 = nn.Sequential(*vgg16.features[10:17])

        # fix
        for i in range(3):
            for param in getattr(self, "layer" + str(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, input):
        pool1 = self.layer1(input)
        pool2 = self.layer2(pool1)
        pool3 = self.layer3(pool2)

        return [pool1, pool2, pool3]


class VGG19map(nn.Module):
    def __init__(self) -> None:
        super(VGG19map, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.layer1 = nn.Sequential(*vgg19.features[:5])
        self.layer2 = nn.Sequential(*vgg19.features[5:10])
        self.layer3 = nn.Sequential(*vgg19.features[10:19])
        self.layer4 = nn.Sequential(*vgg19.features[19:28])
        self.layer5 = nn.Sequential(*vgg19.features[28:37])

        # fix
        for i in range(5):
            for param in getattr(self, "layer" + str(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, input):
        pool1 = self.layer1(input)
        pool2 = self.layer2(pool1)
        pool3 = self.layer3(pool2)
        pool4 = self.layer4(pool3)
        pool5 = self.layer5(pool4)

        return [pool1, pool2, pool3, pool4, pool5]


class PerceptualLoss(nn.Module):
    def __init__(self, vgg19=True) -> None:
        super(PerceptualLoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        if vgg19 is True:
            self.vgg_map = VGG19map()
            self.lambdas = [1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5]
        else:
            self.vgg_map = VGG16map()
            self.lambdas = [1, 1, 1]

    def gram_matrix(self, input: torch.Tensor) -> torch.Tensor:
        (b, ch, h, w) = input.size()
        features = input.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        input = torch.zeros(b, ch, ch).type(features.type())
        gram = torch.baddbmm(
            input, features, features_t, beta=0, alpha=1.0 / (ch * h * w), out=None
        )
        return gram

    def forward(
        self,
        gt: torch.Tensor,
        out: torch.Tensor,
    ) -> float:

        loss = 0.0

        # L_perceptual
        out_feature_map = self.vgg_map(out)
        gt_feature_map = self.vgg_map(gt)

        loss = 0.0
        for i in range(len(out_feature_map)):
            loss += (self.lambdas[i] * 255) * self.l1_loss(
                out_feature_map[i], gt_feature_map[i]
            )

        return loss
