# -*- coding: utf-8 -*-
import torch
import flair
flair.device = torch.device('cpu')

__version__ = '0.2.6rc1ds2'

from .training import auto_device

from eznlp import io
from eznlp import nn
from eznlp import model
from eznlp import training
from eznlp import utils
