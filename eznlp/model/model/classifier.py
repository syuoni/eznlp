# -*- coding: utf-8 -*-
from typing import List, Union
import torch

from ...wrapper import Batch
from ...nn.functional import mask2seq_lens
from ...config import Config, ConfigDict
from ..embedder import OneHotConfig
from ..encoder import EncoderConfig
from ..nested_embedder import SoftLexiconConfig
from ..decoder import TextClassificationDecoderConfig
from .base import ModelConfigBase, ModelBase


class ClassifierConfig(ModelConfigBase):
    """Configurations of a classifier. 
    
    classifier
      ├─decoder(*)
      └─intermediate2
          ├─intermediate1
          │   ├─ohots
          │   ├─mhots
          │   └─nested_ohots
          ├─elmo
          ├─bert_like
          ├─flair_fw
          └─flair_bw
    """
    
    _embedder_names = ['ohots', 'mhots', 'nested_ohots']
    _encoder_names = ['intermediate1', 'intermediate2']
    _pretrained_names = ['elmo', 'bert_like', 'flair_fw', 'flair_bw']
    _all_names = _embedder_names + ['intermediate1'] + _pretrained_names + ['intermediate2'] + ['decoder']
    
    def __init__(self, **kwargs):
        self.ohots = kwargs.pop('ohots', ConfigDict({'text': OneHotConfig(field='text')}))
        self.mhots = kwargs.pop('mhots', None)
        self.nested_ohots = kwargs.pop('nested_ohots', None)
        self.intermediate1 = kwargs.pop('intermediate1', None)
        
        self.elmo = kwargs.pop('elmo', None)
        self.bert_like = kwargs.pop('bert_like', None)
        self.flair_fw = kwargs.pop('flair_fw', None)
        self.flair_bw = kwargs.pop('flair_bw', None)
        self.intermediate2 = kwargs.pop('intermediate2', EncoderConfig(arch='LSTM'))
        
        self.decoder = kwargs.pop('decoder', TextClassificationDecoderConfig())
        super().__init__(**kwargs)
        
        
    @property
    def _is_re_tokenized(self):
        return self.bert_like is not None and not self.bert_like.from_tokenized
        
    @property
    def valid(self):
        return super().valid and (not self._is_re_tokenized or (self.full_emb_dim == 0 and self.full_hid_dim == self.bert_like.out_dim))
        
    @property
    def full_emb_dim(self):
        return sum(getattr(self, name).out_dim for name in self._embedder_names if getattr(self, name) is not None)
        
    @property
    def full_hid_dim(self):
        if self.intermediate1 is not None:
            full_hid_dim = self.intermediate1.out_dim
        else:
            full_hid_dim = self.full_emb_dim
        full_hid_dim += sum(getattr(self, name).out_dim for name in self._pretrained_names if getattr(self, name) is not None)
        return full_hid_dim
        
    def build_vocabs_and_dims(self, *partitions):
        if self.ohots is not None:
            for c in self.ohots.values():
                c.build_vocab(*partitions)
        
        if self.mhots is not None:
            for c in self.mhots.values():
                c.build_dim(partitions[0][0]['tokens'])
        
        if self.nested_ohots is not None:
            for c in self.nested_ohots.values():
                c.build_vocab(*partitions)
                if isinstance(c, SoftLexiconConfig):
                    # Skip the last split (assumed to be test set)
                    c.build_freqs(*partitions[:-1])
        
        if self.intermediate1 is not None:
            self.intermediate1.in_dim = self.full_emb_dim
        
        if self.intermediate2 is not None:
            self.intermediate2.in_dim = self.full_hid_dim
            self.decoder.in_dim = self.intermediate2.out_dim
        else:
            self.decoder.in_dim = self.full_hid_dim
        
        self.decoder.build_vocab(*partitions)
        
        
    def exemplify(self, data_entry: dict, training: bool=True):
        example = {}
        
        if self.ohots is not None:
            example['ohots'] = {f: c.exemplify(data_entry['tokens']) for f, c in self.ohots.items()}
        
        if self.mhots is not None:
            example['mhots'] = {f: c.exemplify(data_entry['tokens']) for f, c in self.mhots.items()}
        
        if self.nested_ohots is not None:
            example['nested_ohots'] = {f: c.exemplify(data_entry['tokens']) for f, c in self.nested_ohots.items()}
        
        for name in self._pretrained_names:
            if getattr(self, name) is not None:
                example[name] = getattr(self, name).exemplify(data_entry['tokens'])
        
        example.update(self.decoder.exemplify(data_entry, training=training))
        return example
        
        
    def batchify(self, batch_examples: List[dict]):
        batch = {}
        
        if self.ohots is not None:
            batch['ohots'] = {f: c.batchify([ex['ohots'][f] for ex in batch_examples]) for f, c in self.ohots.items()}
        
        if self.mhots is not None:
            batch['mhots'] = {f: c.batchify([ex['mhots'][f] for ex in batch_examples]) for f, c in self.mhots.items()}
        
        if self.nested_ohots is not None:
            batch['nested_ohots'] = {f: c.batchify([ex['nested_ohots'][f] for ex in batch_examples]) for f, c in self.nested_ohots.items()}
        
        for name in self._pretrained_names:
            if getattr(self, name) is not None:
                batch[name] = getattr(self, name).batchify([ex[name] for ex in batch_examples])
        
        # Replace mask/seq_lens if re-tokenization
        if self._is_re_tokenized:
            batch['mask'] = batch['bert_like']['sub_mask'][:, 2:]
            batch['seq_lens'] = mask2seq_lens(batch['mask'])
        
        batch.update(self.decoder.batchify(batch_examples))
        return batch
        
        
    def instantiate(self):
        # Only check validity at the most outside level
        assert self.valid
        return Classifier(self)



class Classifier(ModelBase):
    def __init__(self, config: ClassifierConfig):
        super().__init__(config)
        
        
    def _get_full_embedded(self, batch: Batch):
        embedded = []
        
        if hasattr(self, 'ohots'):
            ohots_embedded = [self.ohots[f](batch.ohots[f]) for f in self.ohots]
            embedded.extend(ohots_embedded)
        
        if hasattr(self, 'mhots'):
            mhots_embedded = [self.mhots[f](batch.mhots[f]) for f in self.mhots]
            embedded.extend(mhots_embedded)
        
        if hasattr(self, 'nested_ohots'):
            nested_ohots_embedded = [self.nested_ohots[f](**batch.nested_ohots[f], seq_lens=batch.seq_lens) for f in self.nested_ohots]
            embedded.extend(nested_ohots_embedded)
        
        return torch.cat(embedded, dim=-1)
        
        
    def _get_full_hidden(self, batch: Batch):
        full_hidden = []
        
        if any([hasattr(self, name) for name in ClassifierConfig._embedder_names]):
            embedded = self._get_full_embedded(batch)
            if hasattr(self, 'intermediate1'):
                full_hidden.append(self.intermediate1(embedded, batch.mask))
            else:
                full_hidden.append(embedded)
        
        for name in ClassifierConfig._pretrained_names:
            if hasattr(self, name):
                full_hidden.append(getattr(self, name)(**getattr(batch, name)))
        
        full_hidden = torch.cat(full_hidden, dim=-1)
        
        if hasattr(self, 'intermediate2'):
            return self.intermediate2(full_hidden, batch.mask)
        else:
            return full_hidden
        
        
    def pretrained_parameters(self):
        params = []
        
        if hasattr(self, 'elmo'):
            params.extend(self.elmo.elmo._elmo_lstm.parameters())
        
        if hasattr(self, 'bert_like'):
            params.extend(self.bert_like.bert_like.parameters())
        
        if hasattr(self, 'flair_fw'):
            params.extend(self.flair_fw.flair_lm.parameters())
        if hasattr(self, 'flair_bw'):
            params.extend(self.flair_bw.flair_lm.parameters())
        
        return params
        
        
    def forward2states(self, batch: Batch):
        return {'full_hidden': self._get_full_hidden(batch)}
