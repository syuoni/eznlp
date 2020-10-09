# -*- coding: utf-8 -*-
import string
from collections import Counter, OrderedDict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.experimental.vocab import Vocab
from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor

from ..datasets_utils import TensorWrapper, Batch
from .data_utils import tags2simple_entities



class SequenceTaggingDataset(Dataset):
    _pre_enum_fields = ['bigram', 'trigram', 'en_pattern', 'en_pattern_sum', 
                        'prefix_2', 'prefix_3', 'prefix_4', 'prefix_5', 
                        'suffix_2', 'suffix_3', 'suffix_4', 'suffix_5']
    _pre_val_fields = ['en_shape_features', 'num_features']
    
    def __init__(self, data, enum_fields=None, val_fields=None, 
                 vocabs=None, sub_tokenizer=None, cascade=False, labeling='BIOES'):
        """
        Parameters
        ----------
        data : list of dict 
            Each dict contains `tokens` and optionally `tags`. 
        """
        super().__init__()
        self.data = data
        self._is_labeled = ('tags' in data[0])
        self._building_vocabs = (vocabs is None)
        
        self.enum_fields = enum_fields if enum_fields is not None else self._pre_enum_fields
        self.val_fields = val_fields if val_fields is not None else self._pre_val_fields
        
        if self._building_vocabs:
            assert self._is_labeled
            self.tag_helper = TagHelper(cascade=cascade, labeling=labeling)
            self._build_vocabs()
        else:
            self.char_vocab = vocabs[0]
            self.tok_vocab = vocabs[1]
            self.enum_fields_vocabs = vocabs[2]
            self.tag_helper = vocabs[3]
            assert self.tag_helper.cascade == cascade
            assert self.tag_helper.labeling == labeling
            if self._is_labeled:
                self._check_vocabs()
                
        self.sub_tokenizer = sub_tokenizer
                
        # It is generally recommended to return cpu tensors in multi-process loading. 
        # See https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
        self.char_trans = sequential_transforms(vocab_func(self.char_vocab), totensor(torch.long))
        self.tok_trans = sequential_transforms(vocab_func(self.tok_vocab), totensor(torch.long))
        self.enum_fields_trans = {f: sequential_transforms(vocab_func(v), totensor(torch.long)) \
                                  for f, v in self.enum_fields_vocabs.items()}
        self.val_fields_trans = {f: sequential_transforms(totensor(torch.float), lambda x: (x*2-1) / 10) \
                                 for f in self.val_fields}
        
            
    def _build_vocabs(self):
        char_counter = Counter()
        tok_counter = Counter()
        enum_fields_counters = {f: Counter() for f in self.enum_fields}
        tag_counter = Counter()
        cas_tag_counter = Counter()
        cas_type_counter = Counter()
        
        with tqdm(total=len(self)) as t:
            for curr_data in self.data:
                for tok in curr_data['tokens'].raw_text:
                    char_counter.update(tok)
                tok_counter.update(curr_data['tokens'].text)
                for f, c in enum_fields_counters.items():
                    c.update(getattr(curr_data['tokens'], f))
                
                tag_counter.update(curr_data['tags'])
                cas_tag_counter.update(self.tag_helper.build_cas_tags_by_tags(curr_data['tags']))
                cas_type_counter.update(self.tag_helper.build_cas_types_by_tags(curr_data['tags']))
                t.update(1)
                
        preserve_chars = string.ascii_letters + string.digits
        for c in preserve_chars:
            char_counter.pop(c)
        self.char_vocab = Vocab(OrderedDict([('<unk>', 100), ('<pad>', 100)] + [(c, 100) for c in preserve_chars] + \
                                            char_counter.most_common()), min_freq=5)
        self.tok_vocab = Vocab(OrderedDict([('<unk>', 100), ('<pad>', 100)] + tok_counter.most_common()), min_freq=2)
        self.enum_fields_vocabs = {f: Vocab(OrderedDict([('<unk>', 100), ('<pad>', 100)] + c.most_common()), min_freq=10) \
                                   for f, c in enum_fields_counters.items()}
        
        idx2tag = ['<pad>'] + list(tag_counter.keys())
        tag2idx = {t: i for i, t in enumerate(idx2tag)}
        idx2cas_tag = ['<pad>'] + list(cas_tag_counter.keys())
        cas_tag2idx = {t: i for i, t in enumerate(idx2cas_tag)}
        idx2cas_type = ['<pad>'] + list(cas_type_counter.keys())
        cas_type2idx = {t: i for i, t in enumerate(idx2cas_type)}
        self.tag_helper.set_vocabs(idx2tag, tag2idx, idx2cas_tag, cas_tag2idx, idx2cas_type, cas_type2idx)
        
        
    def _check_vocabs(self):
        tag_counter = Counter()
        for curr_data in self.data:
            tag_counter.update(curr_data['tags'])
            
        oov_tags = [tag for tag in tag_counter if tag not in self.tag_helper.tag2idx]
        if len(oov_tags) > 0:
            raise ValueError(f"OOV tags exist: {oov_tags}")
            
            
    def get_vocabs(self):
        return self.char_vocab, self.tok_vocab, self.enum_fields_vocabs, self.tag_helper
    
    
    def set_cascade(self, cascade):
        self.tag_helper.set_cascade(cascade)
        
            
    def summary(self):
        n_seqs = len(self.data)
        n_raws = len({curr_data['raw_idx'] for curr_data in self.data})
        max_len = max([len(curr_data['tokens']) for curr_data in self.data])
        print(f"The dataset consists {n_seqs} sequences built from {n_raws} raw entries")
        print(f"The max sequence length is {max_len}")
        
        
    def get_model_config(self):
        config = {'char': {'pad_idx': self.char_vocab['<pad>'], 
                           'voc_dim': len(self.char_vocab)}, 
                  'tok': {'pad_idx': self.tok_vocab['<pad>'], 
                          'voc_dim': len(self.tok_vocab)}, 
                  'enum': {f: {'pad_idx': vocab['<pad>'], 
                               'voc_dim': len(vocab)} for f, vocab in self.enum_fields_vocabs.items()}, 
                  'val': {f: {'in_dim': len(getattr(self.data[0]['tokens'], f)[0])} for f in self.val_fields}}
        return config, self.tag_helper
    
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, i):
        curr_data = self.data[i]
        
        char_ids = [self.char_trans(tok) for tok in curr_data['tokens'].raw_text]
        tok_ids = self.tok_trans(curr_data['tokens'].text)
        enum_feats = {f: trans(getattr(curr_data['tokens'], f)) for f, trans in self.enum_fields_trans.items()}
        val_feats = {f: trans(getattr(curr_data['tokens'], f)) for f, trans in self.val_fields_trans.items()}
        
        if self.sub_tokenizer is not None:
            tokens = curr_data['tokens']
            tokens.build_word_pieces(self.sub_tokenizer)
            
            wp_ids = [self.sub_tokenizer.cls_token_id] \
                   +  self.sub_tokenizer.convert_tokens_to_ids(tokens.word_pieces) \
                   + [self.sub_tokenizer.sep_token_id]
            wp_feats = {'wp_ids': torch.tensor(wp_ids), 
                        'wp_tok_pos': torch.tensor(tokens.word_piece_tok_pos)}
        else:
            wp_feats = None
        
        tags_obj = Tags(curr_data['tags'], self.tag_helper) if self._is_labeled else None
        return TensorWrapper(char_ids=char_ids, tok_ids=tok_ids, enum=enum_feats, val=val_feats, 
                             wp=wp_feats, tags_obj=tags_obj)
    
    def collate(self, batch_examples):
        batch_char_ids = []
        batch_tok_ids = []
        batch_enum = {f: [] for f in self.enum_fields}
        batch_val = {f: [] for f in self.val_fields}
        batch_wp = {'wp_ids': [], 'wp_tok_pos': []} if self.sub_tokenizer is not None else {}
        batch_tags_objs = []
        
        for ex in batch_examples:
            # batch_char_ids: List (batch*sentence) of 1D-tensors
            batch_char_ids.extend(ex.char_ids)
            batch_tok_ids.append(ex.tok_ids)
            for f in batch_enum:
                batch_enum[f].append(ex.enum[f])
            for f in batch_val:
                batch_val[f].append(ex.val[f])
            for f in batch_wp:
                batch_wp[f].append(ex.wp[f])
            batch_tags_objs.append(ex.tags_obj)
            
        
        tok_lens = torch.tensor([t.size(0) for t in batch_char_ids])
        seq_lens = torch.tensor([s.size(0) for s in batch_tok_ids])
        
        batch_char_ids = pad_sequence(batch_char_ids, batch_first=True, padding_value=self.char_vocab['<pad>'])
        batch_tok_ids = pad_sequence(batch_tok_ids, batch_first=True, padding_value=self.tok_vocab['<pad>'])
        
        for f in self.enum_fields:
            batch_enum[f] = pad_sequence(batch_enum[f], batch_first=True,
                                         padding_value=self.enum_fields_vocabs[f]['<pad>'])
        for f in self.val_fields:
            batch_val[f] = pad_sequence(batch_val[f], batch_first=True, 
                                        padding_value=0.0)
            
        if self.sub_tokenizer is not None:
            wp_lens = torch.tensor([wps.size(0) for wps in batch_wp['wp_ids']])
            batch_wp = {'wp_ids': pad_sequence(batch_wp['wp_ids'], batch_first=True, 
                                               padding_value=self.sub_tokenizer.pad_token_id), 
                        'wp_tok_pos': pad_sequence(batch_wp['wp_tok_pos'], batch_first=True, 
                                                   padding_value=-1)}
        else:
            wp_lens, batch_wp = None, None
            
        batch_tags_objs = batch_tags_objs if self._is_labeled else None
        
        
        batch = Batch(tok_lens=tok_lens, char_ids=batch_char_ids, 
                      seq_lens=seq_lens, tok_ids=batch_tok_ids, enum=batch_enum, val=batch_val, 
                      wp_lens=wp_lens, wp=batch_wp, 
                      tags_objs=batch_tags_objs)
        batch.build_masks({'char_mask': (batch_char_ids.size(), tok_lens), 
                           'tok_mask': (batch_tok_ids.size(), seq_lens)})
        if self.sub_tokenizer is not None:
            batch.build_masks({'wp_mask': (batch_wp['wp_ids'].size(), wp_lens)})
            
        return batch
    
    
    
class TagHelper(object):
    def __init__(self, cascade=False, labeling='BIOES'):
        self.cascade = cascade
        self.labeling = labeling
        
    def set_cascade(self, cascade):
        self.cascade = cascade
        
    def set_vocabs(self, idx2tag, tag2idx, idx2cas_tag, cas_tag2idx, idx2cas_type, cas_type2idx):
        self.idx2tag = idx2tag
        self.tag2idx = tag2idx
        self.idx2cas_tag = idx2cas_tag
        self.cas_tag2idx = cas_tag2idx
        self.idx2cas_type = idx2cas_type
        self.cas_type2idx = cas_type2idx
        
    def build_cas_tags_by_tags(self, tags: list):
        return [tag.split('-')[0] for tag in tags]
    
    def build_cas_types_by_tags(self, tags: list):
        return [tag.split('-')[1] if '-' in tag else tag for tag in tags]
        
    def build_cas_ent_slices_and_types_by_tags(self, tags: list):
        simple_entities = tags2simple_entities(tags, labeling=self.labeling)
        cas_ent_slices = [slice(ent['start'], ent['stop']) for ent in simple_entities]
        cas_ent_types = [ent['type'] for ent in simple_entities]
        return cas_ent_slices, cas_ent_types
        
    def build_cas_ent_slices_by_cas_tags(self, cas_tags: list):
        """
        This functions is used for decoding. 
        """
        simple_entities = tags2simple_entities(cas_tags, labeling=self.labeling)
        cas_ent_slices = [slice(ent['start'], ent['stop']) for ent in simple_entities]
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
    
    
    def get_cas_type_voc_dim(self):
        return len(self.cas_type2idx)
    
    def get_cas_type_pad_idx(self):
        return self.cas_type2idx['<pad>']
    
    def get_modeling_tag_voc_dim(self):
        return len(self.cas_tag2idx) if self.cascade else len(self.tag2idx)
        
    def get_modeling_tag_pad_idx(self):
        return self.cas_tag2idx['<pad>'] if self.cascade else self.tag2idx['<pad>']
        
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
        if self.cascade:
            return self.ids2cas_tags(tag_ids)
        else:
            return self.ids2tags(tag_ids)
        
    def modeling_tags2ids(self, tags):
        if self.cascade:
            return self.cas_tags2ids(tags)
        else:
            return self.tags2ids(tags)
        
        
    def fetch_batch_tags(self, batch_tags_objs: list):
        return [tags_obj.tags for tags_obj in batch_tags_objs]
    
    def fetch_batch_cas_tags(self, batch_tags_objs: list):
        return [tags_obj.cas_tags for tags_obj in batch_tags_objs]
    
    def fetch_batch_modeling_tags(self, batch_tags_objs: list):
        if self.cascade:
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
        if self.cascade:
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
    
    
class Tags(TensorWrapper):
    def __init__(self, tags: list, tag_helper: TagHelper):
        self.tags = tags
        self.cas_tags = tag_helper.build_cas_tags_by_tags(tags)
        self.cas_types = tag_helper.build_cas_types_by_tags(tags)
        self.cas_ent_slices, self.cas_ent_types = tag_helper.build_cas_ent_slices_and_types_by_tags(tags)
        
        self.tag_ids = torch.tensor(tag_helper.tags2ids(self.tags))
        self.cas_tag_ids = torch.tensor(tag_helper.cas_tags2ids(self.cas_tags))
        self.cas_type_ids = torch.tensor(tag_helper.cas_types2ids(self.cas_types))
        self.cas_ent_type_ids = torch.tensor(tag_helper.cas_types2ids(self.cas_ent_types))
        
    def _apply_to_tensors(self, func):
        self.tag_ids = func(self.tag_ids)
        self.cas_tag_ids = func(self.cas_tag_ids)
        self.cas_type_ids = func(self.cas_type_ids)
        self.cas_ent_type_ids = func(self.cas_ent_type_ids)
        return self
    
    