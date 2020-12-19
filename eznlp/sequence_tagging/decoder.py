# -*- coding: utf-8 -*-
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

from ..data import Batch
from ..data.dataset import unpad_seqs
from ..nn.init import reinit_layer_
from ..nn import CombinedDropout
from ..config import Config
from .crf import CRF


class DecoderConfig(Config):
    def __init__(self, **kwargs):
        self.in_dim = kwargs.pop('in_dim', None)
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
        
        self.arch = kwargs.pop('arch', 'CRF')
        if self.arch.lower() not in ('softmax', 'crf'):
            raise ValueError(f"Invalid decoder architecture {self.arch}")
            
        self.scheme = kwargs.pop('scheme', 'BIOES')
        idx2tag = kwargs.pop('idx2tag', None)
        self.set_vocab(idx2tag)
        super().__init__(**kwargs)
        
        
    def set_vocab(self, idx2tag: List[str]):
        self.idx2tag = idx2tag
        self.tag2idx = {t: i for i, t in enumerate(idx2tag)} if idx2tag is not None else None
        
    def ids2tags(self, tag_ids):
        return [self.idx2tag[idx] for idx in tag_ids]
    
    def tags2ids(self, tags):
        return [self.tag2idx[tag] for tag in tags]
        
    def __repr__(self):
        repr_attr_dict = {key: self.__dict__[key] for key in ['arch', 'in_dim', 'scheme', 'in_drop_rates']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def voc_dim(self):
        return len(self.tag2idx)
        
    @property
    def pad_idx(self):
        return self.tag2idx['<pad>']
    
    def instantiate(self):
        if self.arch.lower() == 'softmax':
            return SoftMaxDecoder(self)
        elif self.arch.lower() == 'crf':
            return CRFDecoder(self)
        
        
class Decoder(torch.nn.Module):
    def __init__(self, config: DecoderConfig):
        """
        `Decoder` forward from hidden states to outputs. 
        """
        super().__init__()
        self.hid2tag = torch.nn.Linear(config.in_dim, config.voc_dim)
        self.dropout = CombinedDropout(*config.in_drop_rates)
        reinit_layer_(self.hid2tag, 'sigmoid')
        
        self.idx2tag = config.idx2tag
        self.tag2idx = config.tag2idx
        self.scheme = config.scheme
        
        
    def ids2tags(self, tag_ids):
        return [self.idx2tag[idx] for idx in tag_ids]
    
    def tags2ids(self, tags):
        return [self.tag2idx[tag] for tag in tags]
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        raise NotImplementedError("Not Implemented `forward`")
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        raise NotImplementedError("Not Implemented `decode`")
        
        
        
class SoftMaxDecoder(Decoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=config.pad_idx, reduction='sum')
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        # tag_feats: (batch, step, tag_dim)
        tag_feats = self.hid2tag(self.dropout(full_hidden))
        
        losses = [self.criterion(tfeats[:slen], tags_obj.tag_ids) for tfeats, tags_obj, slen in zip(tag_feats, batch.tags_objs, batch.seq_lens.cpu().tolist())]
        # `torch.stack`: Concatenates sequence of tensors along a new dimension. 
        return torch.stack(losses, dim=0)
    
    
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        # tag_feats: (batch, step, tag_dim)
        tag_feats = self.hid2tag(full_hidden)
        
        best_paths = tag_feats.argmax(dim=-1)
        batch_tag_ids = unpad_seqs(best_paths, batch.seq_lens)
        return [self.ids2tags(tag_ids) for tag_ids in batch_tag_ids]
        
    
    
class CRFDecoder(Decoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.crf = CRF(tag_dim=config.voc_dim, pad_idx=config.pad_idx, batch_first=True)
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        # tag_feats: (batch, step, tag_dim)
        tag_feats = self.hid2tag(self.dropout(full_hidden))
        
        batch_tag_ids = pad_sequence([tags_obj.tag_ids for tags_obj in batch.tags_objs], batch_first=True, padding_value=self.crf.pad_idx)
        losses = self.crf(tag_feats, batch_tag_ids, mask=batch.tok_mask)
        return losses
            
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        # tag_feats: (batch, step, tag_dim)
        tag_feats = self.hid2tag(full_hidden)
        
        # List of List of predicted-tag-ids
        batch_tag_ids = self.crf.decode(tag_feats, mask=batch.tok_mask)
        return [self.ids2tags(tag_ids) for tag_ids in batch_tag_ids]
    