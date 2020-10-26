# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import torch.nn as nn

from ..datasets_utils import Batch, unpad_seqs
from ..nn_utils import reinit_layer_
from .config import DecoderConfig
from .crf import CRF


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        """
        `Decoder` forward from hidden states to outputs. 
        """
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, batch: Batch, full_hidden: Tensor):
        raise NotImplementedError("Not Implemented `forward`")
        
        
    def decode(self, batch: Batch, full_hidden: Tensor):
        raise NotImplementedError("Not Implemented `decode`")
        
        
        
class SoftMaxDecoder(Decoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.hid2tag = nn.Linear(config.in_dim, config.modeling_tag_voc_dim)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.modeling_tag_pad_idx, reduction='sum')
        reinit_layer_(self.hid2tag, 'sigmoid')
        
        
    def forward(self, batch: Batch, full_hidden: Tensor):
        # tag_feats: (batch, step, tag_dim)
        tag_feats = self.hid2tag(self.dropout(full_hidden))
        
        batch_tag_ids = self.config.fetch_batch_modeling_tag_ids(batch.tags_objs)
        losses = [self.criterion(tfeats[:slen], tids) for tfeats, tids, slen in zip(tag_feats, batch_tag_ids, batch.seq_lens.cpu().tolist())]
        # `torch.stack`: Concatenates sequence of tensors along a new dimension. 
        return torch.stack(losses, dim=0)
    
    
    def decode(self, batch: Batch, full_hidden: Tensor):
        # tag_feats: (batch, step, tag_dim)
        tag_feats = self.hid2tag(full_hidden)
        
        best_paths = tag_feats.argmax(dim=-1)
        batch_tag_ids = unpad_seqs(best_paths, batch.seq_lens)
        return [self.config.ids2modeling_tags(tag_ids) for tag_ids in batch_tag_ids]
        
    
class CRFDecoder(Decoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.hid2tag = nn.Linear(config.in_dim, config.modeling_tag_voc_dim)
        self.crf = CRF(tag_dim=config.modeling_tag_voc_dim, 
                       pad_idx=config.modeling_tag_pad_idx, 
                       batch_first=True)
        reinit_layer_(self.hid2tag, 'sigmoid')
        
        
    def forward(self, batch: Batch, full_hidden: Tensor):
        # tag_feats: (batch, step, tag_dim)
        tag_feats = self.hid2tag(self.dropout(full_hidden))
        
        batch_tag_ids = self.config.fetch_batch_modeling_tag_ids(batch.tags_objs, padding=True)
        losses = self.crf(tag_feats, batch_tag_ids, mask=batch.tok_mask)
        return losses
            
        
    def decode(self, batch: Batch, full_hidden: Tensor):
        # tag_feats: (batch, step, tag_dim)
        tag_feats = self.hid2tag(full_hidden)
        
        # List of List of predicted-tag-ids
        batch_tag_ids = self.crf.decode(tag_feats, mask=batch.tok_mask)
        return [self.config.ids2modeling_tags(tag_ids) for tag_ids in batch_tag_ids]
    
    
class CascadeDecoder(Decoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        
        if config.arch.lower() == 'softmax':
            self.base_decoder = SoftMaxDecoder(config)
        elif config.arch.lower() == 'crf':
            self.base_decoder = CRFDecoder(config)
        else:
            raise ValueError(f"Invalid decoder architecture {config.arch}")
        
        self.cas_hid2type = nn.Linear(config.in_dim, config.cas_type_voc_dim)
        self.cas_criterion = nn.CrossEntropyLoss(ignore_index=config.cas_type_pad_idx, reduction='sum')
        reinit_layer_(self.cas_hid2type, 'sigmoid')
        
        
    def forward(self, batch: Batch, full_hidden: Tensor):
        tag_losses = self.base_decoder(batch, full_hidden)
        
        # type_feats: (batch, step, type_dim)
        type_feats = self.cas_hid2type(full_hidden)
        
        if self.config.cascade_mode.lower() == 'straight':
            batch_type_ids = self.config.fetch_batch_cas_type_ids(batch.tags_objs)
            type_losses = [self.cas_criterion(tfeats[:slen], tids) for tfeats, tids, slen in zip(type_feats, batch_type_ids, batch.seq_lens.cpu().tolist())]
        else:
            # TODO: Teacher-forcing?
            type_losses = []
            for tfeats, (ent_slices, ent_type_ids) in zip(type_feats, self.config.fetch_batch_ent_slices_and_type_ids(batch.tags_objs)):
                if len(ent_slices) == 0:
                    type_losses.append(torch.tensor(0.0, device=tag_losses.device))
                else:
                    ent_type_feats = torch.stack([tfeats[sli].mean(dim=0) for sli in ent_slices], dim=0)
                    type_losses.append(self.cas_criterion(ent_type_feats, ent_type_ids))
        return tag_losses + torch.stack(type_losses, dim=0)
    
    
    def decode(self, batch: Batch, full_hidden: Tensor):
        batch_cas_tags = self.base_decoder.decode(batch, full_hidden)
        
        # type_feats: (batch, step, type_dim)
        type_feats = self.cas_hid2type(full_hidden)
        
        batch_tags = []
        if self.config.cascade_mode.lower() == 'straight':
            batch_type_ids = type_feats.argmax(dim=-1)
            batch_type_ids = unpad_seqs(batch_type_ids, batch.seq_lens)
            for cas_tags, cas_type_ids in zip(batch_cas_tags, batch_type_ids):
                cas_types = self.config.ids2cas_types(cas_type_ids)
                tags = self.config.build_tags_by_cas_tags_and_types(cas_tags, cas_types)
                batch_tags.append(tags)
        else:
            for tfeats, cas_tags in zip(type_feats, batch_cas_tags):
                ent_slices = self.config.build_cas_ent_slices_by_cas_tags(cas_tags)
                
                if len(ent_slices) == 0:
                    ent_types = []
                else:
                    ent_type_feats = torch.stack([tfeats[sli].mean(dim=0) for sli in ent_slices], dim=0)
                    ent_type_ids = ent_type_feats.argmax(dim=-1).cpu().tolist()
                    ent_types = self.config.ids2cas_types(ent_type_ids)
                tags = self.config.build_tags_by_cas_tags_and_ent_slices_and_types(cas_tags, ent_slices, ent_types)
                batch_tags.append(tags)
        return batch_tags
    
        
    