# -*- coding: utf-8 -*-
from torch.nn.utils.rnn import pad_sequence

from ..token import Token
from ..config import Config, ConfigList, EmbedderConfig, EncoderConfig, PreTrainedEmbedderConfig
from .transitions import ChunksTagsTranslator


class DecoderConfig(Config):
    def __init__(self, **kwargs):
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
    
    def __repr__(self):
        kwargs_repr = ', '.join(f"{key}={self.__dict__[key]}" for key in ['arch', 'in_dim', 'dropout', 'scheme', 'cascade_mode'])
        return f"{self.__class__.__name__}({kwargs_repr})"
        
        
class TaggerConfig(Config):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        embedder: EmbedderConfig
        encoders: ConfigList[EncoderConfig]
        elmo_embedder: PreTrainedEmbedderConfig
        bert_like_embedder: PreTrainedEmbedderConfig
        decoder: DecoderConfig
        """
        self.embedder = kwargs.pop('embedder', EmbedderConfig())
        self.encoders = kwargs.pop('encoders', ConfigList([EncoderConfig(arch='LSTM')]))
        
        self.elmo_embedder = kwargs.pop('elmo_embedder', None)
        self.bert_like_embedder = kwargs.pop('bert_like_embedder', None)
        
        self.decoder = kwargs.pop('decoder', DecoderConfig())
        super().__init__(**kwargs)
        
        
    @property
    def is_valid(self):
        if self.decoder is None or not self.decoder.is_valid:
            return False
        if self.embedder is None or not self.embedder.is_valid:
            return False
        
        if self.encoders is not None and self.encoders.is_valid:
            return True
        if self.elmo_embedder is not None and self.elmo_embedder.is_valid:
            return True
        if self.bert_like_embedder is not None and self.bert_like_embedder.is_valid:
            return True
        
        return False
        
    
    def _update_dims(self, ex_token: Token=None):
        if self.embedder.val is not None and ex_token is not None:
            for f, val_config in self.embedder.val.items():
                val_config.in_dim = getattr(ex_token, f).shape[0]
                
        if self.encoders is not None:
            for enc_config in self.encoders:
                enc_config.in_dim = self.embedder.out_dim
                if enc_config.arch.lower() == 'shortcut':
                    enc_config.hid_dim = self.embedder.out_dim
        
        full_hid_dim = 0
        full_hid_dim += self.encoders.hid_dim if self.encoders is not None else 0
        full_hid_dim += self.elmo_embedder.out_dim if self.elmo_embedder is not None else 0
        full_hid_dim += self.bert_like_embedder.out_dim if self.bert_like_embedder is not None else 0
        self.decoder.in_dim = full_hid_dim
        
        
    @property
    def name(self):
        name_elements = []
        if self.embedder is not None and self.embedder.char is not None:
            name_elements.append("Char" + self.embedder.char.arch)
        
        if self.encoders is not None:
            name_elements.append(self.encoders.arch)
            
        if self.elmo_embedder is not None:
            name_elements.append(self.elmo_embedder.arch)
            
        if self.bert_like_embedder is not None:
            name_elements.append(self.bert_like_embedder.arch)
            
        name_elements.append(self.decoder.arch)
        name_elements.append(self.decoder.cascade_mode)
        return '-'.join(name_elements)
    
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(\n"
                f"\tembedder={repr(self.embedder)}\n"
                f"\tencoders={repr(self.encoders)}\n"
                f"\telmo_embedder={repr(self.elmo_embedder)}\n"
                f"\tbert_like_embedder={repr(self.bert_like_embedder)}\n"
                f"\tdecoder={repr(self.decoder)})")
    
    