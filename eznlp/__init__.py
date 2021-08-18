# -*- coding: utf-8 -*-
import torch
import flair
flair.device = torch.device('cpu')

__version__ = '0.2.1'

from .training.utils import auto_device

import eznlp.io
import eznlp.nn
import eznlp.model
import eznlp.training
import eznlp.utils
