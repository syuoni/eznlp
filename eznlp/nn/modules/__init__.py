# -*- coding: utf-8 -*-
from .aggregation import SequencePooling, SequenceGroupAggregating, ScalarMix
from .attention import SequenceAttention
from .dropout import WordDropout, LockedDropout, CombinedDropout
from .crf import CRF
from .loss import SoftLabelCrossEntropyLoss, SmoothLabelCrossEntropyLoss, FocalLoss
