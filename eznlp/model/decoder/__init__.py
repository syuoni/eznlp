# -*- coding: utf-8 -*-
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase

from .text_classification import TextClassificationDecoderConfig

from .sequence_tagging import SequenceTaggingDecoderConfig
from .span_classification import SpanClassificationDecoderConfig
from .span_attr_classification import SpanAttrClassificationDecoderConfig
from .span_rel_classification import SpanRelClassificationDecoderConfig
from .boundary_selection import BoundarySelectionDecoderConfig
from .joint_extraction import JointExtractionDecoderConfig
from .specific_span_classification import SpecificSpanClsDecoderConfig
from .specific_span_rel_classification import SpecificSpanRelClsDecoderConfig
from .specific_span_sparse_rel_classification import SpecificSpanSparseRelClsDecoderConfig
from .masked_span_rel_classification import MaskedSpanRelClsDecoderConfig

from .generator import GeneratorConfig
