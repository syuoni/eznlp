# -*- coding: utf-8 -*-
from collections import Counter, OrderedDict
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.experimental.vocab import Vocab

from ..dataset_utils import TensorWrapper, Batch, _fetch_token_id
from .decoder import DecoderConfig
from .tagger import SequenceTaggerConfig


class SequenceTaggingDataset(torch.utils.data.Dataset):
    def __init__(self, data: list, config: SequenceTaggerConfig):
        """
        Parameters
        ----------
        data : list
            [{'tokens': TokenSequence, 'tags': list of str}, ...]
            
            `tags` is an optional key, which does not exist if the data are unlabeled. 
        """
        super().__init__()
        self.data = data
        self.config = config
        
        self._is_labeled = ('tags' in data[0])
        self._building_vocabs = (config.decoder.idx2tag is None)
        
        if self._building_vocabs:
            assert self._is_labeled
            
            self._build_token_vocab()
            self._build_tag_vocab()
            
            if self.config.embedder.char is not None:
                self._build_char_vocab()
            
            if self.config.embedder.enum is not None:
                self._build_enum_vocabs()
                
            self.config._update_dims(self.data[0]['tokens'][0])
        else:
            if self._is_labeled:
                self._check_tag_vocabs()
                
        
    def _build_token_vocab(self):
        counter = Counter()
        for curr_data in self.data:
            counter.update(curr_data['tokens'].text)
        self.config.embedder.token.vocab = Vocab(OrderedDict([('<unk>', 100), ('<pad>', 100)] + counter.most_common()), min_freq=1)
    
    def extend_token_vocab(self, *others):
        counter = Counter()
        for data in others:
            for curr_data in data:
                counter.update(curr_data['tokens'].text)
                
        existing_tokens = self.config.embedder.token.vocab.get_itos()
        existing_set = set(existing_tokens)
        self.config.embedder.token.vocab = Vocab(OrderedDict([(tok, 100) for tok in existing_tokens] + \
                                               [(tok, freq) for tok, freq in counter.most_common() if tok not in existing_set]), min_freq=1)
        
    
    def _build_char_vocab(self):
        counter = Counter()
        for curr_data in self.data:
            for tok in curr_data['tokens'].raw_text:
                counter.update(tok)
        self.config.embedder.char.vocab = Vocab(OrderedDict([('<unk>', 100), ('<pad>', 100)] + counter.most_common()), min_freq=1)
        
    def _build_enum_vocabs(self):
        counters = {f: Counter() for f in self.config.embedder.enum.keys()}
        for curr_data in self.data:
            for f, c in counters.items():
                c.update(getattr(curr_data['tokens'], f))
        
        for f, enum_config in self.config.embedder.enum.items():
            enum_config.vocab = Vocab(OrderedDict([('<unk>', 100), ('<pad>', 100)] + counters[f].most_common()), min_freq=1)
            
            
    def _build_tag_vocab(self):
        tag_counter = Counter()
        for curr_data in self.data:
            tag_counter.update(curr_data['tags'])
        self.config.decoder.set_vocab(idx2tag=['<pad>'] + list(tag_counter.keys()))
            
            
    # def _build_tag_vocabs(self):
    #     cas_tag_counter = Counter()
    #     cas_type_counter = Counter()
    #     for curr_data in self.data:
    #         cas_tag_counter.update(self.config.decoder.build_cas_tags_by_tags(curr_data['tags']))
    #         cas_type_counter.update(self.config.decoder.build_cas_types_by_tags(curr_data['tags']))
            
    #     self.config.decoder.set_vocabs(idx2cas_tag=['<pad>'] + list(cas_tag_counter.keys()), 
    #                                    idx2cas_type=['<pad>'] + list(cas_type_counter.keys()))
        
    def _check_tag_vocabs(self):
        tag_counter = Counter()
        for curr_data in self.data:
            tag_counter.update(curr_data['tags'])
            
        oov_tags = [tag for tag in tag_counter if tag not in self.config.decoder.tag2idx]
        if len(oov_tags) > 0:
            raise ValueError(f"OOV tags exist: {oov_tags}")
            
        
    def summary(self):
        n_seqs = len(self.data)
        if 'raw_idx' in self.data[0]:
            n_raws = len({curr_data['raw_idx'] for curr_data in self.data})
        else:
            n_raws = n_seqs
            
        max_len = max([len(curr_data['tokens']) for curr_data in self.data])
        print(f"The dataset consists {n_seqs} sequences built from {n_raws} raw entries")
        print(f"The max sequence length is {max_len}")
        
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, i):
        curr_data = self.data[i]
        
        tok_ids = self.config.embedder.token.trans(curr_data['tokens'].text)
        
        if self.config.embedder.char is not None:
            char_ids = [self.config.embedder.char.trans(tok) for tok in curr_data['tokens'].raw_text]
        else:
            char_ids = None
        if self.config.embedder.enum is not None:
            enum_feats = {f: enum_config.trans(getattr(curr_data['tokens'], f)) for f, enum_config in self.config.embedder.enum.items()}
        else:
            enum_feats = None
        if self.config.embedder.val is not None:
            val_feats = {f: val_config.trans(getattr(curr_data['tokens'], f)) for f, val_config in self.config.embedder.val.items()}
        else:
            val_feats = None
            
        # Prepare text for pretrained embedders
        tokenized_raw_text = curr_data['tokens'].raw_text
            
        # if self.config.bert_like_embedder is not None:
        #     tokens = curr_data['tokens']
        #     tokenizer = self.config.bert_like_embedder.tokenizer
            
        #     tokens.build_sub_tokens(tokenizer)
        #     sub_tok_ids = torch.tensor([tokenizer.cls_token_id] + \
        #                                 tokenizer.convert_tokens_to_ids(tokens.sub_tokens) + \
        #                                [tokenizer.sep_token_id])
        #     ori_indexes = torch.tensor(tokens.ori_indexes)
        # else:
        #     sub_tok_ids, ori_indexes = None, None
            
        tags_obj = Tags(curr_data['tags'], self.config.decoder) if self._is_labeled else None
        
        return TensorWrapper(char_ids=char_ids, tok_ids=tok_ids, enum=enum_feats, val=val_feats, 
                             tokenized_raw_text=tokenized_raw_text, 
                             # sub_tok_ids=sub_tok_ids, ori_indexes=ori_indexes, 
                             tags_obj=tags_obj)
    
    
    
    def collate(self, batch_examples):
        batch_tok_ids = [ex.tok_ids for ex in batch_examples]
        seq_lens = torch.tensor([seq.size(0) for seq in batch_tok_ids])
        batch_tok_ids = pad_sequence(batch_tok_ids, batch_first=True, padding_value=self.config.embedder.token.pad_idx)
        
        if self.config.embedder.char is not None:
            batch_char_ids = [tok for ex in batch_examples for tok in ex.char_ids]
            tok_lens = torch.tensor([tok.size(0) for tok in batch_char_ids])
            batch_char_ids = pad_sequence(batch_char_ids, batch_first=True, padding_value=self.config.embedder.char.pad_idx)
        else:
            batch_char_ids, tok_lens = None, None
            
        if self.config.embedder.enum is not None:
            batch_enum = {f: pad_sequence([ex.enum[f] for ex in batch_examples], batch_first=True, 
                                          padding_value=enum_config.pad_idx) for f, enum_config in self.config.embedder.enum.items()}
        else:
            batch_enum = None
            
        if self.config.embedder.val is not None:
            batch_val = {f: pad_sequence([ex.val[f] for ex in batch_examples], batch_first=True, 
                                         padding_value=0.0) for f in self.config.embedder.val.keys()}
        else:
            batch_val = None
        
        
        # Prepare text for pretrained embedders
        batch_tokenized_raw_text = [ex.tokenized_raw_text for ex in batch_examples]
        
        
        # if self.config.elmo_embedder is not None:
        #     batch_elmo_char_ids = batch_to_elmo_char_ids([ex.tokenized_raw_text for ex in batch_examples])
        # else:
        #     batch_elmo_char_ids = None
            
        # if self.config.bert_like_embedder is not None:
        #     batch_sub_tok_ids = [ex.sub_tok_ids for ex in batch_examples]
        #     sub_tok_seq_lens = torch.tensor([seq.size(0) for seq in batch_sub_tok_ids])
        #     batch_sub_tok_ids = pad_sequence(batch_sub_tok_ids, batch_first=True, 
        #                                      padding_value=self.config.bert_like_embedder.tokenizer.pad_token_id)
        #     batch_ori_indexes = pad_sequence([ex.ori_indexes for ex in batch_examples], batch_first=True, 
        #                                      padding_value=-1)
        # else:
        #     batch_sub_tok_ids, sub_tok_seq_lens, batch_ori_indexes = None, None, None
            
        # if self.config.flair_embedder is not None:
        #     batch_flair_sentences = [" ".join(ex.tokenized_raw_text) for ex in batch_examples]
        # else:
        #     batch_flair_sentences = None
            
        batch_tags_objs = [ex.tags_obj for ex in batch_examples] if self._is_labeled else None
        
        
        batch = Batch(tok_ids=batch_tok_ids, seq_lens=seq_lens, 
                      char_ids=batch_char_ids, tok_lens=tok_lens, 
                      enum=batch_enum, val=batch_val, 
                      tokenized_raw_text=batch_tokenized_raw_text, 
                      # elmo_char_ids=batch_elmo_char_ids, 
                      # sub_tok_ids=batch_sub_tok_ids, ori_indexes=batch_ori_indexes, sub_tok_seq_lens=sub_tok_seq_lens, 
                      # flair_sentences=batch_flair_sentences, 
                      tags_objs=batch_tags_objs)
        
        batch.build_masks({'tok_mask': (seq_lens, batch_tok_ids.size(1))})
        if self.config.embedder.char is not None:
            batch.build_masks({'char_mask': (tok_lens, batch_char_ids.size(1))})
        # if self.config.bert_like_embedder is not None:
        #     batch.build_masks({'sub_tok_mask': (sub_tok_seq_lens, batch_sub_tok_ids.size(1))})
        
        return batch
    
    
    
    
class Tags(TensorWrapper):
    def __init__(self, tags: list, config: DecoderConfig):
        self.tags = tags
        # self.cas_tags = config.build_cas_tags_by_tags(tags)
        # self.cas_types = config.build_cas_types_by_tags(tags)
        # self.cas_ent_slices, self.cas_ent_types = config.build_cas_ent_slices_and_types_by_tags(tags)
        
        self.tag_ids = torch.tensor(config.tags2ids(self.tags))
        # self.cas_tag_ids = torch.tensor(config.cas_tags2ids(self.cas_tags))
        # self.cas_type_ids = torch.tensor(config.cas_types2ids(self.cas_types))
        # self.cas_ent_type_ids = torch.tensor(config.cas_types2ids(self.cas_ent_types))
        
    def _apply_to_tensors(self, func):
        self.tag_ids = func(self.tag_ids)
        # self.cas_tag_ids = func(self.cas_tag_ids)
        # self.cas_type_ids = func(self.cas_type_ids)
        # self.cas_ent_type_ids = func(self.cas_ent_type_ids)
        return self
    
    