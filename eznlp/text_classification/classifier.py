# -*- coding: utf-8 -*-
from torchtext.experimental.vectors import Vectors
import allennlp.modules
import transformers
import flair

from ..data import Batch
from ..model import ModelConfig, Model
from .decoder import TextClassificationDecoderConfig


class TextClassifierConfig(ModelConfig):
    def __init__(self, **kwargs):
        self.decoder: TextClassificationDecoderConfig = kwargs.pop('decoder', TextClassificationDecoderConfig())
        super().__init__(**kwargs)
        
    @property
    def extra_name(self):
        if self.decoder.use_attention:
            return self.decoder.attention_scoring
        else:
            return self.decoder.pooling_mode
    
    def instantiate(self, 
                    pretrained_vectors: Vectors=None, 
                    elmo: allennlp.modules.elmo.Elmo=None, 
                    bert_like: transformers.PreTrainedModel=None, 
                    flair_fw_lm: flair.models.LanguageModel=None, 
                    flair_bw_lm: flair.models.LanguageModel=None):
        # Only check validity at the most outside level
        assert self.is_valid
        if (self.bert_like_embedder is not None) and (not self.bert_like_embedder.from_tokenized):
            assert len(self.optional_assemblies) == 1
            return BERTTextClassifier(self, bert_like)
        else:
            return TextClassifier(self, pretrained_vectors, elmo, bert_like, flair_fw_lm, flair_bw_lm)
        
        
        
class TextClassifier(Model):
    def __init__(self, config: TextClassifierConfig, 
                 pretrained_vectors: Vectors=None, 
                 elmo: allennlp.modules.elmo.Elmo=None, 
                 bert_like: transformers.PreTrainedModel=None, 
                 flair_fw_lm: flair.models.LanguageModel=None, 
                 flair_bw_lm: flair.models.LanguageModel=None):
        super().__init__(config, pretrained_vectors, elmo, bert_like, flair_fw_lm, flair_bw_lm)
        
        
        
class BERTTextClassifier(Model):
    def __init__(self, config: TextClassifierConfig, bert_like: transformers.PreTrainedModel=None):
        super().__init__(config, bert_like=bert_like)
        
    def get_full_hidden(self, batch: Batch):
        full_hidden, (seq_lens, mask) = self.bert_like_embedder(batch)
        # Replace seq_lens / mask
        batch.seq_lens = seq_lens
        batch.tok_mask = mask
        
        if not hasattr(self, 'intermediate'):
            return full_hidden
        else:
            return self.intermediate(batch, full_hidden)
        
        