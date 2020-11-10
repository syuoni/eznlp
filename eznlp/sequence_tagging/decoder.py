# -*- coding: utf-8 -*-
import torch
from torch.nn.utils.rnn import pad_sequence

from ..dataset_utils import Batch, unpad_seqs
from ..nn.init import reinit_layer_
from ..config import Config
from .crf import CRF
from .transition import ChunksTagsTranslator


class DecoderConfig(Config):
    def __init__(self, **kwargs):
        # TODO: Seperate some methods...
        self.arch = kwargs.pop('arch', 'CRF')
        if self.arch.lower() not in ('softmax', 'crf'):
            raise ValueError(f"Invalid decoder architecture {self.arch}")
            
        self.in_dim = kwargs.pop('in_dim', None)
        self.dropout = kwargs.pop('dropout', 0.5)
        
        self.scheme = kwargs.pop('scheme', 'BIOES')
        self.translator = ChunksTagsTranslator(scheme=self.scheme)
        
        self.cascade_mode = kwargs.pop('cascade_mode', 'None')
        if self.cascade_mode.lower() not in ('none', 'straight', 'sliced'):
            raise ValueError(f"Invalid cascade mode {self.cascade_mode}")
        self._is_cascade = (self.cascade_mode.lower() != 'none')
        
        idx2tag = kwargs.pop('idx2tag', None)
        idx2cas_tag = kwargs.pop('idx2cas_tag', None)
        idx2cas_type = kwargs.pop('idx2cas_type', None)
        self.set_vocabs(idx2tag, idx2cas_tag, idx2cas_type)
        super().__init__(**kwargs)
        
        
    def set_vocabs(self, idx2tag: list, idx2cas_tag: list, idx2cas_type: list):
        self.idx2tag = idx2tag
        self.tag2idx = {t: i for i, t in enumerate(idx2tag)} if idx2tag is not None else None
        self.idx2cas_tag = idx2cas_tag
        self.cas_tag2idx = {t: i for i, t in enumerate(idx2cas_tag)} if idx2cas_tag is not None else None
        self.idx2cas_type = idx2cas_type
        self.cas_type2idx = {t: i for i, t in enumerate(idx2cas_type)} if idx2cas_type is not None else None
        
        
    def instantiate(self):
        if self.cascade_mode.lower() == 'none':
            if self.arch.lower() == 'softmax':
                return SoftMaxDecoder(self)
            elif self.arch.lower() == 'crf':
                return CRFDecoder(self)
        else:
            return CascadeDecoder(self)
        
        
    def __repr__(self):
        repr_attr_dict = {key: self.__dict__[key] for key in ['arch', 'in_dim', 'dropout', 'scheme', 'cascade_mode']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def cas_type_voc_dim(self):
        return len(self.cas_type2idx)
    
    @property
    def cas_type_pad_idx(self):
        return self.cas_type2idx['<pad>']
    
    @property
    def modeling_tag_voc_dim(self):
        return len(self.cas_tag2idx) if self._is_cascade else len(self.tag2idx)
    
    @property
    def modeling_tag_pad_idx(self):
        return self.cas_tag2idx['<pad>'] if self._is_cascade else self.tag2idx['<pad>']
        
    def ids2tags(self, tag_ids):
        return [self.idx2tag[idx] for idx in tag_ids]
    
    def tags2ids(self, tags):
        return [self.tag2idx[tag] for tag in tags]
    
    def ids2cas_tags(self, cas_tag_ids):
        return [self.idx2cas_tag[idx] for idx in cas_tag_ids]
    
    def cas_tags2ids(self, cas_tags):
        return [self.cas_tag2idx[tag] for tag in cas_tags]
    
    def ids2cas_types(self, cas_type_ids):
        return [self.idx2cas_type[idx] for idx in cas_type_ids]
    
    def cas_types2ids(self, cas_types):
        return [self.cas_type2idx[typ] for typ in cas_types]
    
    def ids2modeling_tags(self, tag_ids):
        if self._is_cascade:
            return self.ids2cas_tags(tag_ids)
        else:
            return self.ids2tags(tag_ids)
        
    def modeling_tags2ids(self, tags):
        if self._is_cascade:
            return self.cas_tags2ids(tags)
        else:
            return self.tags2ids(tags)
        
    def build_cas_tags_by_tags(self, tags: list):
        return [tag.split('-')[0] for tag in tags]
    
    def build_cas_types_by_tags(self, tags: list):
        return [tag.split('-')[1] if '-' in tag else tag for tag in tags]
        
    def build_cas_ent_slices_and_types_by_tags(self, tags: list):
        chunks = self.translator.tags2chunks(tags)
        cas_ent_types = [typ for typ, start, end in chunks]
        cas_ent_slices = [slice(start, end) for typ, start, end in chunks]
        return cas_ent_slices, cas_ent_types
        
    def build_cas_ent_slices_by_cas_tags(self, cas_tags: list):
        """
        This function is only used for sliced cascade decoding. 
        """
        chunks = self.translator.tags2chunks(cas_tags)
        cas_ent_slices = [slice(start, end) for typ, start, end in chunks]
        return cas_ent_slices
    
    def build_tags_by_cas_tags_and_types(self, cas_tags: list, cas_types: list):
        tags = []
        for cas_tag, cas_typ in zip(cas_tags, cas_types):
            if cas_tag == '<pad>' or cas_typ == '<pad>':
                tags.append('<pad>')
            elif cas_tag == 'O' or cas_typ == 'O':
                tags.append('O')
            else:
                tags.append(cas_tag + '-' + cas_typ)
        return tags
        
    def build_tags_by_cas_tags_and_ent_slices_and_types(self, cas_tags: list, cas_ent_slices: list, cas_ent_types: list):
        cas_types = ['O' for _ in cas_tags]
        for sli, typ in zip(cas_ent_slices, cas_ent_types):
            for k in range(sli.start, sli.stop):
                cas_types[k] = typ
        return self.build_tags_by_cas_tags_and_types(cas_tags, cas_types)
    
    
    def fetch_batch_tags(self, batch_tags_objs: list):
        return [tags_obj.tags for tags_obj in batch_tags_objs]
    
    def fetch_batch_cas_tags(self, batch_tags_objs: list):
        return [tags_obj.cas_tags for tags_obj in batch_tags_objs]
    
    def fetch_batch_modeling_tags(self, batch_tags_objs: list):
        if self._is_cascade:
            return self.fetch_batch_cas_tags(batch_tags_objs)
        else:
            return self.fetch_batch_tags(batch_tags_objs)
    
    def fetch_batch_tag_ids(self, batch_tags_objs: list, padding: bool=False):
        batch_tag_ids = [tags_obj.tag_ids for tags_obj in batch_tags_objs]
        if padding:
            batch_tag_ids = pad_sequence(batch_tag_ids, batch_first=True, padding_value=self.tag2idx['<pad>'])
        return batch_tag_ids
    
    def fetch_batch_cas_tag_ids(self, batch_tags_objs: list, padding: bool=False):
        batch_cas_tag_ids = [tags_obj.cas_tag_ids for tags_obj in batch_tags_objs]
        if padding:
            batch_cas_tag_ids = pad_sequence(batch_cas_tag_ids, batch_first=True, padding_value=self.cas_tag2idx['<pad>'])
        return batch_cas_tag_ids
        
    def fetch_batch_modeling_tag_ids(self, batch_tags_objs: list, padding: bool=False):
        if self._is_cascade:
            return self.fetch_batch_cas_tag_ids(batch_tags_objs, padding=padding)
        else:
            return self.fetch_batch_tag_ids(batch_tags_objs, padding=padding)
        
    def fetch_batch_cas_type_ids(self, batch_tags_objs: list, padding: bool=False):
        batch_cas_type_ids = [tags_obj.cas_type_ids for tags_obj in batch_tags_objs]
        if padding:
            batch_cas_type_ids = pad_sequence(batch_cas_type_ids, batch_first=True, padding_value=self.type2idx['<pad>'])
        return batch_cas_type_ids
        
    def fetch_batch_ent_slices_and_type_ids(self, batch_tags_objs: list):
        return [(tags_obj.cas_ent_slices, tags_obj.cas_ent_type_ids) for tags_obj in batch_tags_objs]
    


class Decoder(torch.nn.Module):
    def __init__(self, config: DecoderConfig):
        """
        `Decoder` forward from hidden states to outputs. 
        """
        super().__init__()
        self.config = config
        self.dropout = torch.nn.Dropout(config.dropout)
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        raise NotImplementedError("Not Implemented `forward`")
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        raise NotImplementedError("Not Implemented `decode`")
        
        
        
class SoftMaxDecoder(Decoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.hid2tag = torch.nn.Linear(config.in_dim, config.modeling_tag_voc_dim)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=config.modeling_tag_pad_idx, reduction='sum')
        reinit_layer_(self.hid2tag, 'sigmoid')
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        # tag_feats: (batch, step, tag_dim)
        tag_feats = self.hid2tag(self.dropout(full_hidden))
        
        batch_tag_ids = self.config.fetch_batch_modeling_tag_ids(batch.tags_objs)
        losses = [self.criterion(tfeats[:slen], tids) for tfeats, tids, slen in zip(tag_feats, batch_tag_ids, batch.seq_lens.cpu().tolist())]
        # `torch.stack`: Concatenates sequence of tensors along a new dimension. 
        return torch.stack(losses, dim=0)
    
    
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        # tag_feats: (batch, step, tag_dim)
        tag_feats = self.hid2tag(full_hidden)
        
        best_paths = tag_feats.argmax(dim=-1)
        batch_tag_ids = unpad_seqs(best_paths, batch.seq_lens)
        return [self.config.ids2modeling_tags(tag_ids) for tag_ids in batch_tag_ids]
        
    
class CRFDecoder(Decoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.hid2tag = torch.nn.Linear(config.in_dim, config.modeling_tag_voc_dim)
        self.crf = CRF(tag_dim=config.modeling_tag_voc_dim, 
                       pad_idx=config.modeling_tag_pad_idx, 
                       batch_first=True)
        reinit_layer_(self.hid2tag, 'sigmoid')
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        # tag_feats: (batch, step, tag_dim)
        tag_feats = self.hid2tag(self.dropout(full_hidden))
        
        batch_tag_ids = self.config.fetch_batch_modeling_tag_ids(batch.tags_objs, padding=True)
        losses = self.crf(tag_feats, batch_tag_ids, mask=batch.tok_mask)
        return losses
            
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
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
        
        self.cas_hid2type = torch.nn.Linear(config.in_dim, config.cas_type_voc_dim)
        self.cas_criterion = torch.nn.CrossEntropyLoss(ignore_index=config.cas_type_pad_idx, reduction='sum')
        reinit_layer_(self.cas_hid2type, 'sigmoid')
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
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
    
    
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
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
    
        
    