# -*- coding: utf-8 -*-
from typing import List
import torch
import allennlp.modules

from ..token import TokenSequence
from ..config import Config


class ELMoConfig(Config):
    def __init__(self, **kwargs):
        self.elmo: allennlp.modules.Elmo = kwargs.pop('elmo')
        self.out_dim = self.elmo.get_output_dim()
        
        self.arch = kwargs.pop('arch', 'ELMo')
        self.freeze = kwargs.pop('freeze', True)
        
        self.lstm_stateful = kwargs.pop('lstm_stateful', False)
        self.mix_layers = kwargs.pop('mix_layers', 'trainable')
        self.use_gamma = kwargs.pop('use_gamma', True)
        
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self.arch
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state['elmo'] = None
        return state
        
        
    def exemplify(self, tokens: TokenSequence):
        return {'tokenized_raw_text': tokens.raw_text}
        
    def batchify(self, batch_ex: List[dict]):
        batch_tokenized_raw_text = [ex['tokenized_raw_text'] for ex in batch_ex]
        return {'char_ids': allennlp.modules.elmo.batch_to_ids(batch_tokenized_raw_text)}
        
    def instantiate(self):
        return ELMoEmbedder(self)



class ELMoEmbedder(torch.nn.Module):
    """
    An embedder based on ELMo representations. 
    
    `Elmo` consists two parts: 
        (1) `_elmo_lstm` includes the (character-based) embedder and (two-layer) BiLSTMs
        (2) `scalar_mix_{i}` includes the "layer weights", which define how different 
            ELMo layers are combined, i.e., `s^{task}` and `gamma^{task}` in Eq. (1) in 
            Peters et al. (2018). This part should always be trainable. 
            
    Setting the `stateful` attribute to False can make the ELMo outputs consistent. 
    
    NOTE: No need to use `torch.no_grad()` if the `requires_grad` attributes of 
    the pretrained model have been properly set. 
    `torch.no_grad()` enforces the result of every computation in its context 
    to have `requires_grad=False`, even when the inputs have `requires_grad=True`.
    
    References
    ----------
    [1] Peters et al. 2018. Deep contextualized word representations. 
    [2] https://github.com/allenai/allennlp/issues/2398
    """
    def __init__(self, config: ELMoConfig):
        super().__init__()
        self.elmo = config.elmo
        
        self.lstm_stateful = config.lstm_stateful
        self.freeze = config.freeze
        self.mix_layers = config.mix_layers
        self.use_gamma = config.use_gamma
        
        # Register ELMo configurations
        self.elmo._elmo_lstm._elmo_lstm.stateful = self.lstm_stateful
        
        if self.mix_layers.lower() != 'trainable':
            self.elmo.scalar_mix_0.scalar_parameters.requires_grad_(False)
            if self.mix_layers.lower() == 'top':
                for scalar_param in self.elmo.scalar_mix_0.scalar_parameters[:-1]:
                    scalar_param.data.fill_(-9e10)
        
        if not self.use_gamma:
            self.elmo.scalar_mix_0.gamma.requires_grad_(False)
        
        
    @property
    def freeze(self):
        return self._freeze
        
    @freeze.setter
    def freeze(self, freeze: bool):
        self._freeze = freeze
        self.elmo._elmo_lstm.requires_grad_(not freeze)
        
        
    def forward(self, char_ids: torch.LongTensor):
        # TODO: use `word_inputs`?
        elmo_outs = self.elmo(inputs=char_ids)
        
        # Reset lstm states if not stateful
        if not self.lstm_stateful:
            self.elmo._elmo_lstm._elmo_lstm.reset_states()
        
        return elmo_outs['elmo_representations'][0]
