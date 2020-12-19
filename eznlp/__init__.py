# -*- coding: utf-8 -*-
import torch
import flair
flair.device = torch.device('cpu')


__version__ = '0.0.4'

from .token import Token, TokenSequence, custom_spacy_tokenizer
from .config import ConfigList, ConfigDict
from .encoder import CharConfig, TokenConfig, EnumConfig, ValConfig, EmbedderConfig
from .encoder import EncoderConfig, PreTrainedEmbedderConfig
from .optim_utils import count_params, build_param_groups_with_keyword2lr, check_param_groups_no_missing
from .trainer import disp_running_info
