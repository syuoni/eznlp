# -*- coding: utf-8 -*-
from typing import List
import torch
import torchvision


from ..config import Config


class ImageEncoderConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'ResNet')
        self.backbone = kwargs.pop('backbone', None)
        self.height = kwargs.pop('height', 14)
        self.width = kwargs.pop('width', 14)
        
        self.folder = kwargs.pop('folder', None)
        self.transforms = kwargs.pop('transforms', None)
        super().__init__(**kwargs)
        
    @property
    def name(self):
        return self.arch
        
    @property
    def num_channels(self):
        last_convs = [conv for key, conv in self.backbone[-1][-1]._modules.items() if key.startswith('conv')]
        return last_convs[-1].out_channels
        
    @property
    def out_dim(self):
        return self.num_channels
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state['backbone'] = None
        return state
        
    def exemplify(self, image_fn: str):
        img = torchvision.io.read_image(f"{self.folder}/{image_fn}")
        return self.transforms(img.float().div(255))
        
    def batchify(self, batch_examples: List[torch.Tensor]):
        return torch.stack(batch_examples)
        
    def instantiate(self):
        return ImageEncoder(self)



class ImageEncoder(torch.nn.Module):
    def __init__(self, config: ImageEncoderConfig):
        super().__init__()
        self.backbone = config.backbone
        self.pool = torch.nn.AdaptiveAvgPool2d((config.height, config.width))
        
    def forward(self, x: torch.FloatTensor):
        # x: (batch, channel, height, width)
        return self.pool(self.backbone(x))
