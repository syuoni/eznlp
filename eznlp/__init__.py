# -*- coding: utf-8 -*-
import torch
import flair
flair.device = torch.device('cpu')


__version__ = '0.0.5'

from .config import ConfigList, ConfigDict
from .encoder import CharConfig, TokenConfig, EnumConfig, ValConfig, EmbedderConfig
from .encoder import EncoderConfig, PreTrainedEmbedderConfig
