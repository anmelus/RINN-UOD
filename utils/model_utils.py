from torch import nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from torchvision.models.resnet import conv1x1, conv3x3, resnet18, ResNet18_Weights


# Taken from https://github.com/mbanani/byoc/blob/main/byoc/models/backbones.py
class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_c, out_c, batchnorm = True, activation = True, k = 3):
        super().__init__()
        if k == 3:
            self.conv = conv3x3(in_c, out_c)
        elif k == 1:
            self.conv = conv1x1(in_c, out_c)
        else:
            raise ValueError()

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_c)
        else:
            self.bn = nn.Identity()

        if activation:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.Identity()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Taken from https://github.com/mbanani/byoc/blob/main/byoc/models/backbones.py
class ResNetEncoder(nn.Module):
    def __init__(self, chan_in = 3, chan_out = 64):
        super().__init__()
        self.inconv = ConvBlock(chan_in, 64, k = 3)
        self.layer1 = resnet18().layer1
        self.layer2 = resnet18().layer1
        self.pool = nn.MaxPool2d(2, 2)
        self.outconv = ConvBlock(64, chan_out, k = 1, activation = False)

    def forward(self, x):
        x = self.inconv(x)
        x = self.pool(self.layer1(x))
        x = self.pool(self.layer2(x))
        x = self.outconv(x)
        return x


def create_model(model_name):
    """Creates the CNN depending on the model we want to train (either trained or randomized)"""
    if model_name == "trained":
        res = resnet18(weights = ResNet18_Weights.DEFAULT)
        res = list(res.children())[:-2]
        return nn.Sequential(*res)
    elif model_name == "random":
        return ResNetEncoder()
