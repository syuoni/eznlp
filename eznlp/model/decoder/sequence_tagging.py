# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import torch

from ...wrapper import TargetWrapper, Batch
from ...utils import ChunksTagsTranslator
from ...nn.utils import unpad_seqs
from ...nn.modules import CombinedDropout, CRF
from ...nn.init import reinit_layer_
from ...metrics import precision_recall_f1_report
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase


class SequenceTaggingDecoderMixin(DecoderMixinBase):
    @property
    def scheme(self):
        return self._scheme
        
    @scheme.setter
    def scheme(self, scheme: str):
        self._scheme = scheme
        self.translator = ChunksTagsTranslator(scheme=scheme)
        
    @property
    def idx2tag(self):
        return self._idx2tag
        
    @idx2tag.setter
    def idx2tag(self, idx2tag: List[str]):
        self._idx2tag = idx2tag
        self.tag2idx = {t: i for i, t in enumerate(self.idx2tag)} if idx2tag is not None else None
        
    @property
    def voc_dim(self):
        return len(self.tag2idx)
        
    @property
    def pad_idx(self):
        return self.tag2idx['<pad>']
        
    def exemplify(self, data_entry: dict, training: bool=True):
        return {'tags_obj': Tags(data_entry, self, training=training)}
        
    def batchify(self, batch_examples: List[dict]):
        return {'tags_objs': [ex['tags_obj'] for ex in batch_examples]}
        
    def retrieve(self, batch: Batch):
        return [tags_obj.chunks for tags_obj in batch.tags_objs]
        
    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        """Micro-F1 for entity recognition. 
        
        References
        ----------
        https://www.clips.uantwerpen.be/conll2000/chunking/output.html
        """
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return ave_scores['micro']['f1']



class Tags(TargetWrapper):
    """A wrapper of tags with underlying chunks. 
    
    Parameters
    ----------
    data_entry: dict
        {'tokens': TokenSequence, 
         'chunks': List[tuple]}
    """
    def __init__(self, data_entry: dict, config: SequenceTaggingDecoderMixin, training: bool=True):
        super().__init__(training)
        
        self.chunks = data_entry.get('chunks', None)
        if self.chunks is not None:
            self.tags = config.translator.chunks2tags(data_entry['chunks'], len(data_entry['tokens']))
            self.tag_ids = torch.tensor([config.tag2idx[t] for t in self.tags], dtype=torch.long)



class SequenceTaggingDecoderConfig(SingleDecoderConfigBase, SequenceTaggingDecoderMixin):
    def __init__(self, **kwargs):
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
        
        self.scheme = kwargs.pop('scheme', 'BIOES')
        self.idx2tag = kwargs.pop('idx2tag', None)
        
        self.use_crf = kwargs.pop('use_crf', True)
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self._name_sep.join([self.scheme, self.criterion])
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'in_drop_rates', 'scheme', 'criterion']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def criterion(self):
        if self.use_crf:
            return "CRF"
        else:
            return super().criterion
        
    def instantiate_criterion(self, **kwargs):
        if self.criterion.lower().startswith('crf'):
            return CRF(tag_dim=self.voc_dim, pad_idx=self.pad_idx, batch_first=True)
        else:
            return super().instantiate_criterion(**kwargs)
        
        
    def build_vocab(self, *partitions):
        counter = Counter()
        for data in partitions:
            for data_entry in data:
                curr_tags = self.translator.chunks2tags(data_entry['chunks'], len(data_entry['tokens']))
                counter.update(curr_tags)
        self.idx2tag = ['<pad>'] + list(counter.keys())
        
        
    def instantiate(self):
        return SequenceTaggingDecoder(self)



class SequenceTaggingDecoder(DecoderBase, SequenceTaggingDecoderMixin):
    def __init__(self, config: SequenceTaggingDecoderConfig):
        super().__init__()
        self.scheme = config.scheme
        self.idx2tag = config.idx2tag
        
        self.dropout = CombinedDropout(*config.in_drop_rates)
        self.hid2logit = torch.nn.Linear(config.in_dim, config.voc_dim)
        reinit_layer_(self.hid2logit, 'sigmoid')
        
        self.criterion = config.instantiate_criterion(ignore_index=config.pad_idx, reduction='sum')
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        # logits: (batch, step, tag_dim)
        logits = self.hid2logit(self.dropout(full_hidden))
        
        if isinstance(self.criterion, CRF):
            batch_tag_ids = torch.nn.utils.rnn.pad_sequence([tags_obj.tag_ids for tags_obj in batch.tags_objs], 
                                                            batch_first=True, 
                                                            padding_value=self.criterion.pad_idx)
            losses = self.criterion(logits, batch_tag_ids, mask=batch.mask)
            
        else:
            losses = [self.criterion(lg[:slen], tags_obj.tag_ids) for lg, tags_obj, slen in zip(logits, batch.tags_objs, batch.seq_lens.cpu().tolist())]
            # `torch.stack`: Concatenates sequence of tensors along a new dimension. 
            losses = torch.stack(losses, dim=0)
        
        return losses
        
        
    def decode_tags(self, batch: Batch, full_hidden: torch.Tensor):
        # logits: (batch, step, tag_dim)
        logits = self.hid2logit(full_hidden)
        
        if isinstance(self.criterion, CRF):
            # List of List of predicted-tag-ids
            batch_tag_ids = self.criterion.decode(logits, mask=batch.mask)
            
        else:
            best_paths = logits.argmax(dim=-1)
            batch_tag_ids = unpad_seqs(best_paths, batch.seq_lens)
        
        return [[self.idx2tag[i] for i in tag_ids] for tag_ids in batch_tag_ids]
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_tags = self.decode_tags(batch, full_hidden)
        return [self.translator.tags2chunks(tags) for tags in batch_tags]
