# -*- coding: utf-8 -*-
from typing import List

import torch
import torchvision

from ..config import Config


class ImageEncoderConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop("arch", "ResNet")
        self.in_channels = kwargs.pop("in_channels", 3)
        self.transforms: torch.nn.Module = kwargs.pop("transforms")
        self.backbone: torch.nn.Module = kwargs.pop("backbone")

        if self.arch.lower() == "vgg":
            self.out_dim = self.backbone.features[-3].out_channels
        elif self.arch.lower() == "resnet":
            last_convs = [
                conv
                for key, conv in self.backbone.layer3[-1]._modules.items()
                if key.startswith("conv")
            ]
            self.out_dim = last_convs[-1].out_channels
        else:
            raise ValueError(f"Invalid image encoder architecture {self.arch}")

        self.freeze = kwargs.pop("freeze", True)
        self.use_cache = kwargs.pop("use_cache", True)

        self.height = kwargs.pop("height", 14)
        self.width = kwargs.pop("width", 14)
        super().__init__(**kwargs)

    @property
    def name(self):
        return self.arch

    def __getstate__(self):
        state = self.__dict__.copy()
        state["backbone"] = None
        return state

    def _read_image(self, img_path: str):
        img = torchvision.io.read_image(img_path).float().div(255)
        assert img.dim() == 3
        if img.size(0) == 1 and self.in_channels > 1:
            img = img.expand(self.in_channels, -1, -1)
        return img

    def exemplify(self, entry: dict, training: bool = True):
        if self.use_cache:
            if "img" not in entry:
                entry["img"] = self._read_image(entry["img_path"])
            img = entry["img"]
        else:
            img = self._read_image(entry["img_path"])

        # `transforms` may include random augmentation
        return {"img": self.transforms(img)}

    def batchify(self, batch_examples: List[dict]):
        # The cached `img` will not be passed to cuda device
        return {"img": torch.stack([ex["img"] for ex in batch_examples])}

    def instantiate(self):
        if self.arch.lower() == "vgg":
            return VGGEncoder(self)
        elif self.arch.lower() == "resnet":
            return ResNetEncoder(self)


class ImageEncoder(torch.nn.Module):
    def __init__(self, config: ImageEncoderConfig):
        super().__init__()
        self.backbone = config.backbone
        self.pool = torch.nn.AdaptiveAvgPool2d((config.height, config.width))
        self.freeze = config.freeze

    @property
    def freeze(self):
        return self._freeze

    @freeze.setter
    def freeze(self, freeze: bool):
        self._freeze = freeze
        self.backbone.requires_grad_(not freeze)

    def forward(self, x: torch.Tensor):
        # x: (batch, channel, height, width)
        return self.pool(self._forward_backbone(x))


class VGGEncoder(ImageEncoder):
    def __init__(self, config: ImageEncoderConfig):
        super().__init__(config)

    def _forward_backbone(self, x: torch.Tensor):
        # Refer to torchvision/models/vgg.py/VGG
        # (224, 224) -> (7, 7)
        # return self.backbone.features(x)
        # (224, 224) -> (14, 14)
        return self.backbone.features[:-1](x)


class ResNetEncoder(ImageEncoder):
    def __init__(self, config: ImageEncoderConfig):
        super().__init__(config)

    def _forward_backbone(self, x: torch.Tensor):
        # Refer to torchvision/models/resnet.py/ResNet
        # conv1[kernel=3, stride=2, padding=3]: (224, 224) -> (112, 112)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        # maxpool[kernel=2]: (112, 112) -> (56, 56)
        x = self.backbone.maxpool(x)

        # layer1: (56, 56) -> (56, 56)
        x = self.backbone.layer1(x)
        # layer2: (56, 56) -> (28, 28)
        x = self.backbone.layer2(x)
        # layer3: (28, 28) -> (14, 14)
        x = self.backbone.layer3(x)
        # layer4: (14, 14) -> (7, 7)
        # x = self.backbone.layer4(x)
        return x
