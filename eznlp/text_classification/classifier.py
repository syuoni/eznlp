# -*- coding: utf-8 -*-
from torchtext.experimental.vectors import Vectors
import allennlp.modules
import transformers
import flair

from ..model import ModelConfig, Model
from .decoder import TextClassificationDecoderConfig


class TextClassifierConfig(ModelConfig):
    def __init__(self, **kwargs):
        self.decoder: TextClassificationDecoderConfig = kwargs.pop('decoder', TextClassificationDecoderConfig())
        super().__init__(**kwargs)
        
    @property
    def extra_name(self):
        return self.decoder.pooling
    
    def instantiate(self, 
                    pretrained_vectors: Vectors=None, 
                    elmo: allennlp.modules.elmo.Elmo=None, 
                    bert_like: transformers.PreTrainedModel=None, 
                    flair_fw_lm: flair.models.LanguageModel=None, 
                    flair_bw_lm: flair.models.LanguageModel=None):
        # Only check validity at the most outside level
        assert self.is_valid
        return TextClassifier(self, pretrained_vectors, elmo, bert_like, flair_fw_lm, flair_bw_lm)
    
    
class TextClassifier(Model):
    def __init__(self, config: TextClassifierConfig, 
                 pretrained_vectors: Vectors=None, 
                 elmo: allennlp.modules.elmo.Elmo=None, 
                 bert_like: transformers.PreTrainedModel=None, 
                 flair_fw_lm: flair.models.LanguageModel=None, 
                 flair_bw_lm: flair.models.LanguageModel=None):
        super().__init__(config, pretrained_vectors, elmo, bert_like, flair_fw_lm, flair_bw_lm)
        
        