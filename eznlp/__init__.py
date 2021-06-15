# -*- coding: utf-8 -*-
import torch
import flair
flair.device = torch.device('cpu')

__version__ = '0.1.2'

from .training.utils import auto_device
