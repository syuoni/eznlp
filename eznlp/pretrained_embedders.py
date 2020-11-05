# -*- coding: utf-8 -*-
from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import allennlp.modules
import transformers
import flair

from .datasets_utils import Batch
from .functional import aggregate_tensor_by_group
from .config import Config


class PreTrainedEmbedderConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch')
        self.out_dim = kwargs.pop('out_dim')
        self.freeze = kwargs.pop('freeze', True)
        
        if self.arch.lower() == 'elmo':
            self.lstm_stateful = kwargs.pop('lstm_stateful', False)
        elif self.arch.lower() in ('bert', 'roberta', 'albert'):
            self.tokenizer = kwargs.pop('tokenizer')
        elif self.arch.lower() == 'flair':
            pass
        else:
            raise ValueError(f"Invalid pretrained embedder architecture {self.arch}")
        super().__init__(**kwargs)
        
    def instantiate(self, pretrained_model: nn.Module):
        if self.arch.lower() == 'elmo':
            return ELMoEmbedder(self, pretrained_model)
        elif self.arch.lower() in ('bert', 'roberta', 'albert'):
            return BertLikeEmbedder(self, pretrained_model)
        elif self.arch.lower() == 'flair':
            return FlairEmbedder(self, pretrained_model)
        
        
class PreTrainedEmbedder(nn.Module):
    """
    `PreTrainedEmbedder` forwards from inputs to hidden states. 
    
    NOTE: No need to use `torch.no_grad()` if the `requires_grad` attributes of 
    the `pretrained_model` have been properly set. 
    `torch.no_grad()` enforces the result of every computation in its context 
    to have `requires_grad=False`, even when the inputs have `requires_grad=True`.
    """
    def __init__(self, config: PreTrainedEmbedderConfig, pretrained_model: nn.Module):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.freeze = config.freeze
        
    @property
    def freeze(self):
        return self._freeze
    
    @freeze.setter
    def freeze(self, value: bool):
        assert isinstance(value, bool)
        self._freeze = value
        self.pretrained_model.requires_grad_(not self._freeze)
        
        
        
class ScalarMix(nn.Module):
    """
    Mix multi-layer hidden states by corresponding scalar weights. 
    
    Computes a parameterised scalar mixture of N tensors, 
    ``mixture = gamma * \sum_k(s_k * tensor_k)``
    where ``s = softmax(w)``, with `w` and `gamma` scalar parameters.
    
    References
    ----------
    [1] Peters et al. 2018. Deep contextualized word representations. 
    [2] https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py
    """
    def __init__(self, mix_dim: int):
        super().__init__()
        self.scalars = nn.Parameter(torch.zeros(mix_dim))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        
    def __repr__(self):
        return f"{self.__class__.__name__}(mix_dim={self.scalars.size(0)})"
        
    def forward(self, tensors: Union[torch.FloatTensor, List[torch.FloatTensor]]):
        if isinstance(tensors, (list, tuple)):
            tensors = torch.stack(tensors)
        
        norm_weights_shape = tuple([-1] + [1] * (tensors.dim()-1))
        norm_weights = F.softmax(self.scalars, dim=0).view(*norm_weights_shape)
        return self.gamma * (tensors * norm_weights).sum(dim=0)
        
        
        
class BertLikeEmbedder(PreTrainedEmbedder):
    def __init__(self, config: PreTrainedEmbedderConfig, bert_like: transformers.PreTrainedModel):
        super().__init__(config, bert_like)
        self.scalar_mix = ScalarMix(bert_like.config.num_hidden_layers + 1)
        
    def forward(self, batch: Batch):
        # bert_outs: (batch, sub_tok_step+2, hid_dim)
        # hidden: list of (batch, sub_tok_step+2, hid_dim)
        bert_outs, _, hidden = self.pretrained_model(input_ids=batch.sub_tok_ids, 
                                                     attention_mask=(~batch.sub_tok_mask).type(torch.long), 
                                                     output_hidden_states=True)
        if hasattr(self, 'scalar_mix'):
            bert_outs = self.scalar_mix(hidden)
        
        # Remove the `[CLS]` and `[SEP]` positions. 
        bert_outs = bert_outs[:, 1:-1]
        
        # agg_bert_outs: (batch, tok_step, hid_dim)
        agg_bert_outs = aggregate_tensor_by_group(bert_outs, batch.ori_indexes, agg_step=batch.tok_ids.size(1))
        return agg_bert_outs
    
    
    
class ELMoEmbedder(PreTrainedEmbedder):
    """
    `Elmo` consists two parts: 
        (1) `_elmo_lstm` includes the (character-based) embedder and (two-layer) BiLSTMs
        (2) `scalar_mix_{i}` includes the "layer weights", which define how different 
            ELMo layers are combined, i.e., `s^{task}` and `gamma^{task}` in Eq. (1) in 
            Peters et al. (2018). This part should always be trainable. 
            
    Setting the `stateful` attribute to False can make the ELMo outputs consistent. 
    
    References
    ----------
    [1] Peters et al. 2018. Deep contextualized word representations. 
    [2] https://github.com/allenai/allennlp/issues/2398
    """
    def __init__(self, config: PreTrainedEmbedderConfig, elmo: allennlp.modules.elmo.Elmo):
        elmo._elmo_lstm._elmo_lstm.stateful = config.lstm_stateful
        super().__init__(config, elmo)
        
    @property
    def freeze(self):
        return self._freeze
    
    @freeze.setter
    def freeze(self, value: bool):
        assert isinstance(value, bool)
        self._freeze = value
        self.pretrained_model._elmo_lstm.requires_grad_(not self._freeze)
        
    def forward(self, batch: Batch):
        # TODO: use `word_inputs`?
        elmo_outs = self.pretrained_model(inputs=batch.elmo_char_ids)
        
        return elmo_outs['elmo_representations'][0]
    
    
class FlairEmbedder(PreTrainedEmbedder):
    def __init__(self, config: PreTrainedEmbedderConfig, flair_emb: flair.embeddings.TokenEmbeddings):
        super().__init__(config, flair_emb)
        self.gamma = nn.Parameter(torch.tensor(1.0))
    
    @property
    def freeze(self):
        return self._freeze
    
    @freeze.setter
    def freeze(self, value: bool):
        assert isinstance(value, bool)
        self._freeze = value
        if isinstance(self.pretrained_model, flair.embeddings.StackedEmbeddings):
            for emb in self.pretrained_model.embeddings:
                emb.requires_grad_(not self._freeze)
                emb.fine_tune = (not self._freeze)
        else:
            self.pretrained_model.requires_grad_(not self._freeze)
            self.pretrained_model.fine_tune = (not self._freeze)
                
                
    def forward(self, batch: Batch):
        flair_sentences = [flair.data.Sentence(sent, use_tokenizer=False) for sent in batch.flair_sentences]
        self.pretrained_model.embed(flair_sentences)
        flair_outs = pad_sequence([torch.stack([tok.embedding for tok in sent]) for sent in flair_sentences], 
                                  batch_first=True, padding_value=0.0)
        # flair would automatically convert to CUDA?
        # flair_outs = flair_outs.to(batch.tok_ids.device)
        
        return self.gamma * flair_outs
        

