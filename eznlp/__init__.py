# -*- coding: utf-8 -*-
import torch
import flair
flair.device = torch.device('cpu')

__version__ = '0.0.6'

from .training.utils import auto_device
