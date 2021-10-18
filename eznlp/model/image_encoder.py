# -*- coding: utf-8 -*-
from typing import List
import torch
import torchvision

from ..config import Config


class ResNetEncoderConfig(Config):
    def __init__(self, **kwargs):
        self.transforms: torch.nn.Module = kwargs.pop('transforms')
        self.resnet: torchvision.models.ResNet = kwargs.pop('resnet')
        last_convs = [conv for key, conv in self.resnet.layer4[-1]._modules.items() if key.startswith('conv')]
        self.out_dim = last_convs[-1].out_channels
        
        self.arch = kwargs.pop('arch', 'ResNet')
        self.freeze = kwargs.pop('freeze', True)
        
        self.use_cache = kwargs.pop('use_cache', True)
        
        self.height = kwargs.pop('height', 14)
        self.width = kwargs.pop('width', 14)
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self.arch
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state['resnet'] = None
        return state
        
    def exemplify(self, entry: dict, training: bool=True):
        if self.use_cache:
            if 'img' not in entry:
                entry['img'] = torchvision.io.read_image(entry['img_path']).float().div(255)
            img = entry['img']
        else:
            img = torchvision.io.read_image(entry['img_path']).float().div(255)
        
        # `transforms` may include random augmentation
        return {'img': self.transforms(img)}
        
        
    def batchify(self, batch_examples: List[dict]):
        # The cached `img` will not be passed to cuda device
        return {'img': torch.stack([ex['img'] for ex in batch_examples])}
        
    def instantiate(self):
        return ResNetEncoder(self)



class ResNetEncoder(torch.nn.Module):
    def __init__(self, config: ResNetEncoderConfig):
        super().__init__()
        self.resnet = config.resnet
        self.pool = torch.nn.AdaptiveAvgPool2d((config.height, config.width))
        
        self.freeze = config.freeze
        
        
    @property
    def freeze(self):
        return self._freeze
        
    @freeze.setter
    def freeze(self, freeze: bool):
        self._freeze = freeze
        self.resnet.requires_grad_(not freeze)
        
    def _forward_resnet(self, x: torch.Tensor):
        # Refer to torchvision/models/resnet.py/ResNet
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x
        
    def forward(self, x: torch.Tensor):
        # x: (batch, channel, height, width)
        return self.pool(self._forward_resnet(x))
