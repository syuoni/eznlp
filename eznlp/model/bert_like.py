# -*- coding: utf-8 -*-
from typing import List
import logging
import re
import tqdm
import numpy
import truecase
import torch
import transformers

from ..utils import find_ascending
from ..token import TokenSequence
from ..nn.modules import SequenceGroupAggregating, ScalarMix
from ..nn.functional import seq_lens2mask
from ..config import Config

logger = logging.getLogger(__name__)


class BertLikeConfig(Config):
    def __init__(self, **kwargs):
        self.tokenizer: transformers.PreTrainedTokenizer = kwargs.pop('tokenizer')
        self.bert_like: transformers.PreTrainedModel = kwargs.pop('bert_like')
        self.hid_dim = self.bert_like.config.hidden_size
        self.num_layers = self.bert_like.config.num_hidden_layers
        
        self.arch = kwargs.pop('arch', 'BERT')
        self.freeze = kwargs.pop('freeze', True)
        
        self.paired_inputs = kwargs.pop('paired_inputs', False)
        self.from_tokenized = kwargs.pop('from_tokenized', True)
        self.from_subtokenized = kwargs.pop('from_subtokenized', False)
        assert self.from_tokenized or (not self.from_subtokenized)
        self.pre_truncation = kwargs.pop('pre_truncation', False)
        self.group_agg_mode = kwargs.pop('group_agg_mode', 'mean')
        self.mix_layers = kwargs.pop('mix_layers', 'top')
        self.use_gamma = kwargs.pop('use_gamma', False)
        
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self.arch
        
    @property
    def out_dim(self):
        return self.hid_dim
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state['bert_like'] = None
        return state
        
    @property
    def sentence_A_id(self):
        # token_type_ids: 
        # - 0 corresponds to a `sentence A` token,
        # - 1 corresponds to a `sentence B` token.
        return 0 
        
    @property
    def sentence_B_id(self):
        return 1
        
        
    def _token_ids_from_string(self, raw_text: str):
        """Tokenize a non-tokenized sentence (string), and convert to sub-token indexes. 
        
        Parameters
        ----------
        raw_text : str
            e.g., "I like this movie."
        
        Returns
        -------
        sub_tok_ids: torch.LongTensor
            A 1D tensor of sub-token indexes.
        """
        sub_tokens = self.tokenizer.tokenize(raw_text)
        
        # Sequence longer than maximum length should be pre-processed
        assert len(sub_tokens) <= self.tokenizer.model_max_length - 2
        sub_tok_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(sub_tokens) + [self.tokenizer.sep_token_id]
        example = {'sub_tok_ids': torch.tensor(sub_tok_ids)}
        
        if self.paired_inputs:
            # NOTE: `[SEP]` will be retained by tokenizer, instead of being tokenized
            sep_loc = sub_tokens.index(self.tokenizer.sep_token)
            sub_tok_type_ids = [self.sentence_A_id] * (sep_loc+1 +1) + [self.sentence_B_id] * (len(sub_tokens)-sep_loc-1 +1)  # `[CLS]` and `[SEP]` at the ends
            example.update({'sub_tok_type_ids': torch.tensor(sub_tok_type_ids)})
        
        return example
        
        
    def _token_ids_from_tokenized(self, tokenized_raw_text: List[str]):
        """Further tokenize a pre-tokenized sentence, and convert to sub-token indexes. 
        
        Parameters
        ----------
        tokenized_raw_text : List[str]
            e.g., ["I", "like", "this", "movie", "."]
        
        Returns
        -------
        sub_tok_ids: torch.LongTensor
            A 1D tensor of sub-token indexes.
        ori_indexes: torch.LongTensor
            A 1D tensor indicating each sub-token's original index in `tokenized_raw_text`.
        """
        if self.from_subtokenized:
            sub_tokens = tokenized_raw_text
        else:
            nested_sub_tokens = _tokenized2nested(tokenized_raw_text, self.tokenizer)
            sub_tokens = [sub_tok for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]
            ori_indexes = [i for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]
        
        # Sequence longer than maximum length should be pre-processed
        assert len(sub_tokens) <= self.tokenizer.model_max_length - 2
        sub_tok_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(sub_tokens) + [self.tokenizer.sep_token_id]
        example = {'sub_tok_ids': torch.tensor(sub_tok_ids)}
        
        if not self.from_subtokenized:
            example.update({'ori_indexes': torch.tensor(ori_indexes)})
        
        if self.paired_inputs:
            # NOTE: `[SEP]` will be retained by tokenizer, instead of being tokenized
            sep_loc = sub_tokens.index(self.tokenizer.sep_token)
            sub_tok_type_ids = [self.sentence_A_id] * (sep_loc+1 +1) + [self.sentence_B_id] * (len(sub_tokens)-sep_loc-1 +1)  # `[CLS]` and `[SEP]` at the ends
            example.update({'sub_tok_type_ids': torch.tensor(sub_tok_type_ids)})
        
        return example
        
        
    def exemplify(self, tokens: TokenSequence):
        tokenized_raw_text = tokens.raw_text
        
        if self.from_tokenized:
            # sub_tok_ids: (step+2, )
            # sub_tok_type_ids: (step+2, )
            # ori_indexes: (step, )
            return self._token_ids_from_tokenized(tokenized_raw_text)
        else:
            # AD-HOC: Use rejoined tokenized raw text here
            return self._token_ids_from_string(" ".join(tokenized_raw_text))
        
        
    def batchify(self, batch_ex: List[dict]):
        batch_sub_tok_ids = [ex['sub_tok_ids'] for ex in batch_ex]
        sub_tok_seq_lens = torch.tensor([sub_tok_ids.size(0) for sub_tok_ids in batch_sub_tok_ids])
        batch_sub_mask = seq_lens2mask(sub_tok_seq_lens)
        batch_sub_tok_ids = torch.nn.utils.rnn.pad_sequence(batch_sub_tok_ids, 
                                                            batch_first=True, 
                                                            padding_value=self.tokenizer.pad_token_id)
        batch = {'sub_tok_ids': batch_sub_tok_ids, 
                 'sub_mask': batch_sub_mask}
        
        if self.from_tokenized and (not self.from_subtokenized):
            batch_ori_indexes = [ex['ori_indexes'] for ex in batch_ex]
            batch_ori_indexes = torch.nn.utils.rnn.pad_sequence(batch_ori_indexes, batch_first=True, padding_value=-1)
            batch.update({'ori_indexes': batch_ori_indexes})
        
        if self.paired_inputs:
            batch_sub_tok_type_ids = [ex['sub_tok_type_ids'] for ex in batch_ex]
            batch_sub_tok_type_ids = torch.nn.utils.rnn.pad_sequence(batch_sub_tok_type_ids, 
                                                                     batch_first=True, 
                                                                     padding_value=self.sentence_A_id)
            batch.update({'sub_tok_type_ids': batch_sub_tok_type_ids})
        
        return batch
        
        
    def instantiate(self):
        return BertLikeEmbedder(self)



class BertLikeEmbedder(torch.nn.Module):
    """
    An embedder based on BERT representations. 
    
    NOTE: No need to use `torch.no_grad()` if the `requires_grad` attributes of 
    the pretrained model have been properly set. 
    `torch.no_grad()` enforces the result of every computation in its context 
    to have `requires_grad=False`, even when the inputs have `requires_grad=True`.
    """
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.bert_like = config.bert_like
        
        self.freeze = config.freeze
        self.mix_layers = config.mix_layers
        self.use_gamma = config.use_gamma
        self.output_hidden_states = config.output_hidden_states
        
        if config.from_tokenized and (not config.from_subtokenized):
            self.group_aggregating = SequenceGroupAggregating(mode=config.group_agg_mode)
        if self.mix_layers.lower() == 'trainable':
            self.scalar_mix = ScalarMix(config.num_layers + 1)
        if self.use_gamma:
            self.gamma = torch.nn.Parameter(torch.tensor(1.0))
        
    @property
    def freeze(self):
        return self._freeze
        
    @freeze.setter
    def freeze(self, freeze: bool):
        self._freeze = freeze
        self.bert_like.requires_grad_(not freeze)
        
        
    def forward(self, 
                sub_tok_ids: torch.LongTensor, 
                sub_mask: torch.BoolTensor, 
                sub_tok_type_ids: torch.BoolTensor=None, 
                ori_indexes: torch.LongTensor=None):
        # last_hidden: (batch, sub_tok_step+2, hid_dim)
        # pooler_output: (batch, hid_dim)
        # hidden: a tuple of (batch, sub_tok_step+2, hid_dim)
        bert_outs = self.bert_like(input_ids=sub_tok_ids, 
                                   attention_mask=(~sub_mask).long(), 
                                   token_type_ids=sub_tok_type_ids, 
                                   output_hidden_states=True)
        
        # bert_hidden: (batch, sub_tok_step+2, hid_dim)
        if self.mix_layers.lower() == 'trainable':
            bert_hidden = self.scalar_mix(bert_outs['hidden_states'])
        elif self.mix_layers.lower() == 'top':
            bert_hidden = bert_outs['last_hidden_state']
        elif self.mix_layers.lower() == 'average':
            bert_hidden = sum(bert_outs['hidden_states']) / len(bert_outs['hidden_states'])
        
        if self.use_gamma:
            bert_hidden = self.gamma * bert_hidden
        
        # Remove the `[CLS]` and `[SEP]` positions. 
        bert_hidden = bert_hidden[:, 1:-1]
        sub_mask = sub_mask[:, 2:]
        
        if hasattr(self, 'group_aggregating'):
            # bert_hidden: (batch, tok_step, hid_dim)
            bert_hidden = self.group_aggregating(bert_hidden, ori_indexes)
        
        if self.output_hidden_states:
            all_bert_hidden = [hidden[:, 1:-1] for hidden in bert_outs['hidden_states']]
            if hasattr(self, 'group_aggregating'):
                all_bert_hidden = [self.group_aggregating(hidden, ori_indexes) for hidden in all_bert_hidden]
            return (bert_hidden, all_bert_hidden)
        else:
            return bert_hidden



def _truecase(tokenized_raw_text: List[str]):
    """
    Get the truecased text. 
    
    Original:  ['FULL', 'FEES', '1.875', 'REOFFER', '99.32', 'SPREAD', '+20', 'BP']
    Truecased: ['Full', 'fees', '1.875', 'Reoffer', '99.32', 'spread', '+20', 'BP']
    
    References
    ----------
    [1] https://github.com/google-research/bert/issues/223
    [2] https://github.com/daltonfury42/truecase
    """
    new_tokenized = tokenized_raw_text.copy()
    
    word_lst = [(w, idx) for idx, w in enumerate(new_tokenized) if all(c.isalpha() for c in w)]
    lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]
    
    if len(lst) > 0 and len(lst) == len(word_lst):
        parts = truecase.get_true_case(' '.join(lst)).split()
        
        # the trucaser have its own tokenization ...
        # skip if the number of word dosen't match
        if len(parts) == len(word_lst): 
            for (w, idx), nw in zip(word_lst, parts):
                new_tokenized[idx] = nw
    
    return new_tokenized


def truecase_for_bert_like(data: list, verbose=True):
    num_truecased = 0
    for entry in tqdm.tqdm(data, disable=not verbose, ncols=100, desc="Truecasing data"):
        tokens = entry['tokens']
        tokenized_raw_text = tokens.raw_text
        new_tokenized_raw_text = _truecase(tokenized_raw_text)
        if new_tokenized_raw_text != tokenized_raw_text:
            # Directly modify the `raw_text` attribute for each token; `text` attribute remains unchanged
            for tok, new_raw_text in zip(entry['tokens'].token_list, new_tokenized_raw_text):
                tok.raw_text = new_raw_text
            num_truecased += 1
    
    logger.info(f"Truecased sequences: {num_truecased} ({num_truecased/len(data)*100:.2f}%)")
    return data



def _tokenized2nested(tokenized_raw_text: List[str], tokenizer: transformers.PreTrainedTokenizer, max_num_from_word: int=5):
    nested_sub_tokens = []
    for word in tokenized_raw_text:
        sub_tokens = tokenizer.tokenize(word)
        if len(sub_tokens) == 0:
            # The tokenizer returns an empty list if the input is a space-like string
            sub_tokens = [tokenizer.unk_token]
        elif len(sub_tokens) > max_num_from_word:
            # The tokenizer may return a very long list if the input is a url
            sub_tokens = sub_tokens[:max_num_from_word]
        nested_sub_tokens.append(sub_tokens)
    
    return nested_sub_tokens



def _truncate_tokens(tokens: TokenSequence, sub_tok_seq_lens: List[int], max_len: int, mode: str='head+tail'):
    if mode.lower() == 'head-only':
        head_len, tail_len = max_len, 0
    elif mode.lower() == 'tail-only':
        head_len, tail_len = 0, max_len
    else:
        head_len = (max_len + 2) // 4
        tail_len = max_len - head_len
    
    # head_end/tail_begin will be 0 if head_len/tail_len == 0
    cum_lens = numpy.cumsum(sub_tok_seq_lens).tolist()
    find_head, head_end = find_ascending(cum_lens, head_len)
    if find_head:
        head_end += 1
    
    rev_cum_lens = numpy.cumsum(sub_tok_seq_lens[::-1]).tolist()
    find_tail, tail_begin = find_ascending(rev_cum_lens, tail_len)
    if find_tail:
        tail_begin += 1
    
    if tail_len == 0:
        assert head_end > 0
        return tokens[:head_end]
    elif head_len == 0:
        assert tail_begin > 0
        return tokens[-tail_begin:]
    else:
        assert head_end > 0 and tail_begin > 0
        return tokens[:head_end] + tokens[-tail_begin:]



def truncate_for_bert_like(data: list, 
                           tokenizer: transformers.PreTrainedTokenizer, 
                           mode: str='head+tail', 
                           verbose=True):
    """Truncate overlong tokens in `data`, typically for text classification. 
    
    Truncation methods:
        1. head-only: keep the first 510 tokens;
        2. tail-only: keep the last 510 tokens;
        3. head+tail: empirically select the first 128 and the last 382 tokens.
    
    References
    ----------
    [1] Sun et al. 2019. How to fine-tune BERT for text classification? CCL 2019. 
    """
    num_truncated = 0
    for entry in tqdm.tqdm(data, disable=not verbose, ncols=100, desc="Truncating data"):
        tokens = entry['tokens']
        nested_sub_tokens = _tokenized2nested(tokens.raw_text, tokenizer)
        sub_tok_seq_lens = [len(tok) for tok in nested_sub_tokens]
        
        if 'paired_tokens' not in entry:
            # Case for single sentence 
            max_len = tokenizer.model_max_length - 2
            if sum(sub_tok_seq_lens) > max_len:
                entry['tokens'] = _truncate_tokens(tokens, sub_tok_seq_lens, max_len, mode=mode)
                num_truncated += 1
        
        else:
            # Case for paired sentences
            p_tokens = entry['paired_tokens']
            p_nested_sub_tokens = _tokenized2nested(p_tokens.raw_text, tokenizer)
            p_sub_tok_seq_lens = [len(tok) for tok in p_nested_sub_tokens]
            
            max_len = tokenizer.model_max_length - 3
            num_sub_toks, num_p_sub_toks = sum(sub_tok_seq_lens), sum(p_sub_tok_seq_lens)
            
            if num_sub_toks + num_p_sub_toks > max_len:
                # AD-HOC: Other ratio?
                max_len1 = max_len // 2
                if num_sub_toks <= max_len1:
                    entry['paired_tokens'] = _truncate_tokens(p_tokens, p_sub_tok_seq_lens, max_len-num_sub_toks, mode=mode)
                elif num_p_sub_toks <= max_len-max_len1:
                    entry['tokens'] = _truncate_tokens(tokens, sub_tok_seq_lens, max_len-num_p_sub_toks, mode=mode)
                else:
                    entry['tokens'] = _truncate_tokens(tokens, sub_tok_seq_lens, max_len1, mode=mode)
                    entry['paired_tokens'] = _truncate_tokens(p_tokens, p_sub_tok_seq_lens, max_len-max_len1, mode=mode)
                num_truncated += 1
    
    logger.info(f"Truncated sequences: {num_truncated} ({num_truncated/len(data)*100:.2f}%)")
    return data



_IE_KEYS = ['tokens', 'chunks', 'relations', 'attributes']


def segment_uniformly_for_bert_like(data: list, tokenizer: transformers.PreTrainedTokenizer, update_raw_idx: bool=False, verbose=True):
    """Segment overlong tokens in `data`. 
    
    Notes: Currently only supports entity recognition. 
    """
    assert 'relations' not in data[0]
    assert 'attributes' not in data[0]
    
    max_len = tokenizer.model_max_length - 2
    new_data = []
    num_segmented = 0
    num_conflict = 0
    for raw_idx, entry in tqdm.tqdm(enumerate(data), disable=not verbose, ncols=100, desc="Segmenting data"):
        tokens = entry['tokens']
        nested_sub_tokens = _tokenized2nested(tokens.raw_text, tokenizer)
        sub_tok_seq_lens = [len(tok) for tok in nested_sub_tokens]
        
        num_sub_tokens = sum(sub_tok_seq_lens)
        if num_sub_tokens > max_len:
            # Avoid segmenting at positions inside a chunk
            is_segmentable = [True for _ in range(len(tokens))]
            for _, start, end in entry.get('chunks', []):
                for i in range(start+1, end):
                    is_segmentable[i] = False
            
            num_spans = int(num_sub_tokens / (max_len*0.8)) + 1
            span_size = num_sub_tokens / num_spans
            
            cum_lens = numpy.cumsum(sub_tok_seq_lens).tolist()
            
            new_entries = []
            span_start = 0
            for i in range(1, num_spans+1):
                if i < num_spans:
                    _, span_end = find_ascending(cum_lens, span_size*i)
                    num_conflict += (not is_segmentable[span_end])
                    while not is_segmentable[span_end]:
                        span_end -= 1
                else:
                    span_end = len(tokens)
                assert sum(sub_tok_seq_lens[span_start:span_end]) <= max_len
                
                new_entry = {'tokens': tokens[span_start:span_end]}
                if 'chunks' in entry:
                    new_entry['chunks'] = [(label, start-span_start, end-span_start) for label, start, end in entry['chunks'] if span_start <= start and end <= span_end]
                
                new_entry.update({k: v for k, v in entry.items() if k not in _IE_KEYS})
                new_entries.append(new_entry)
                span_start = span_end
            
            assert len(new_entries) == num_spans
            assert [text for entry in new_entries for text in entry['tokens'].raw_text] == entry['tokens'].raw_text
            assert [label for entry in new_entries for label, *_ in entry['chunks']] == [label for label, *_ in entry['chunks']]
            num_segmented += 1
            
        else:
            new_entries = [entry]
        
        if update_raw_idx:
            for new_entry in new_entries:
                new_entry['raw_idx'] = raw_idx
        
        new_data.extend(new_entries)
    
    logger.info(f"Segmented sequences: {num_segmented} ({num_segmented/len(data)*100:.2f}%)")
    logger.info(f"Conflict chunks: {num_conflict}")
    return new_data



def _subtokenize_tokens(entry: dict, tokenizer: transformers.PreTrainedTokenizer, num_digits: int=3):
    tokens = entry['tokens']
    nested_sub_tokens = _tokenized2nested(tokens.raw_text, tokenizer)
    sub_tokens = [sub_tok for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]
    sub2ori_idx = [i if j==0 else i+round(j/len(tok), ndigits=num_digits) for i, tok in enumerate(nested_sub_tokens) for j, sub_tok in enumerate(tok)]
    sub2ori_idx.append(len(tokens))
    ori2sub_idx = [si for si, oi in enumerate(sub2ori_idx) if isinstance(oi, int)]
    
    # Warning: Re-construct `tokens`; this may yield `text` attribute different from the original ones
    new_entry = {'tokens': TokenSequence.from_tokenized_text(sub_tokens, **tokens._tokens_kwargs), 
                 'sub2ori_idx': sub2ori_idx, 
                 'ori2sub_idx': ori2sub_idx}
    
    if 'chunks' in entry:
        new_entry['chunks'] = [(label, ori2sub_idx[start], ori2sub_idx[end]) for label, start, end in entry['chunks']]
    if 'relations' in entry:
        new_entry['relations'] = [(label, (h_label, ori2sub_idx[h_start], ori2sub_idx[h_end]), (t_label, ori2sub_idx[t_start], ori2sub_idx[t_end])) 
                                      for label, (h_label, h_start, h_end), (t_label, t_start, t_end) in entry['relations']]
    if 'attributes' in entry:
        new_entry['attributes'] = [(label, (ck_label, ori2sub_idx[ck_start], ori2sub_idx[ck_end])) 
                                       for label, (ck_label, ck_start, ck_end) in entry['attributes']]
    
    new_entry.update({k: v for k, v in entry.items() if k not in _IE_KEYS})
    return new_entry


def subtokenize_for_bert_like(data: list, tokenizer: transformers.PreTrainedTokenizer, num_digits: int=3, verbose=True):
    """For word-level tokens, sub-tokenize the words with sub-word `tokenizer`. 
    Modify the corresponding start/end indexes in `chunks`, `relations`, `attributes`.
    """
    new_data = []
    for entry in tqdm.tqdm(data, disable=not verbose, ncols=100, desc="Subtokenizing words in data"):
        new_entry = _subtokenize_tokens(entry, tokenizer, num_digits=num_digits)
        new_data.append(new_entry)
    
    num_subtokenized = sum(len(entry['sub2ori_idx'])!=len(entry['ori2sub_idx']) for entry in new_data)
    logger.info(f"Sub-tokenized sequences: {num_subtokenized} ({num_subtokenized/len(data)*100:.2f}%)")
    return new_data



def _tokenizer2sub_prefix(tokenizer: transformers.PreTrainedTokenizer):
    if isinstance(tokenizer, transformers.BertTokenizer):
        return "##"
    elif isinstance(tokenizer, transformers.RobertaTokenizer):
        return "Ġ"
    elif isinstance(tokenizer, transformers.AlbertTokenizer):
        return "▁"
    else:
        raise ValueError(f"Invalid tokenizer {tokenizer}")


def _unk_ascii_characters(tokenizer: transformers.PreTrainedTokenizer):
    return [chr(i) for i in range(128) if tokenizer.tokenize(chr(i)) in ([], [tokenizer.unk_token])]


def _merge_enchars(entry: dict, tokenizer: transformers.PreTrainedTokenizer, sub_prefix: str, unk_ascii: list, num_digits: int=3):
    tokens = entry['tokens']
    sub_tokens = []
    curr_enchars = []
    for char in tokens.raw_text:
        if char.isascii() and (char not in unk_ascii):
            curr_enchars.append(char)
        else:
            if len(curr_enchars) > 0:
                curr_sub_tokens = tokenizer.tokenize("".join(curr_enchars))
                assert "".join(stok.replace(sub_prefix, "") for stok in curr_sub_tokens).lower() == "".join(curr_enchars).lower()
                sub_tokens.extend(curr_sub_tokens)
                curr_enchars = []
            
            char_subtokenized = tokenizer.tokenize(char)
            assert len(char_subtokenized) <= 1
            if len(char_subtokenized) == 0:
                char_subtokenized = [tokenizer.unk_token]
            # assert char_subtokenized[0] in (char, char.lower(), tokenizer.unk_token)
            sub_tokens.extend(char_subtokenized) 
    
    if len(curr_enchars) > 0:
        curr_sub_tokens = tokenizer.tokenize("".join(curr_enchars))
        assert "".join(stok.replace(sub_prefix, "") for stok in curr_sub_tokens).lower() == "".join(curr_enchars).lower()
        sub_tokens.extend(curr_sub_tokens)
    
    sub_tokens_wo_prefix = [stok.replace(sub_prefix, "") for stok in sub_tokens]
    assert len("".join(sub_tokens_wo_prefix).replace(tokenizer.unk_token, "_")) == len(tokens)
    ori2sub_idx = [i if j==0 else i+round(j/len(stok), ndigits=num_digits) for i, stok in enumerate(sub_tokens_wo_prefix) for j in range(len(stok) if stok != tokenizer.unk_token else 1)]
    ori2sub_idx.append(len(sub_tokens))
    sub2ori_idx = [oi for oi, si in enumerate(ori2sub_idx) if isinstance(si, int)]
    
    # Warning: Re-construct `tokens`; this may yield `text` attribute different from the original ones
    new_entry = {'tokens': TokenSequence.from_tokenized_text(sub_tokens, **tokens._tokens_kwargs), 
                 'sub2ori_idx': sub2ori_idx, 
                 'ori2sub_idx': ori2sub_idx}
    
    if 'chunks' in entry:
        new_entry['chunks'] = [(label, ori2sub_idx[start], ori2sub_idx[end]) for label, start, end in entry['chunks']]
    if 'relations' in entry:
        new_entry['relations'] = [(label, (h_label, ori2sub_idx[h_start], ori2sub_idx[h_end]), (t_label, ori2sub_idx[t_start], ori2sub_idx[t_end])) 
                                      for label, (h_label, h_start, h_end), (t_label, t_start, t_end) in entry['relations']]
    if 'attributes' in entry:
        new_entry['attributes'] = [(label, (ck_label, ori2sub_idx[ck_start], ori2sub_idx[ck_end])) 
                                       for label, (ck_label, ck_start, ck_end) in entry['attributes']]
    
    new_entry.update({k: v for k, v in entry.items() if k not in _IE_KEYS})
    return new_entry


def merge_enchars_for_bert_like(data: list, tokenizer: transformers.PreTrainedTokenizer, num_digits: int=3, verbose=True):
    """For character-level tokens, merge the consecutive English characters with sub-word `tokenizer`. 
    Modify the corresponding start/end indexes in `chunks`, `relations`, `attributes`. 
    
    Warning: Any consecutive English characters will be first merged as a word and then subtokenized into sub-words, 
    even if the characters originally contain multiple words. 
    Note that the original word boundaries (i.e., spaces) are missing, and not retrievable. 
    """
    sub_prefix = _tokenizer2sub_prefix(tokenizer)
    unk_ascii = _unk_ascii_characters(tokenizer)
    
    new_data = []
    for entry in tqdm.tqdm(data, disable=not verbose, ncols=100, desc="Merging characters in data"):
        new_entry = _merge_enchars(entry, tokenizer, sub_prefix, unk_ascii, num_digits=num_digits)
        new_data.append(new_entry)
    
    num_merged = sum(len(entry['sub2ori_idx'])!=len(entry['ori2sub_idx']) for entry in new_data)
    logger.info(f"Merged sequences: {num_merged} ({num_merged/len(data)*100:.2f}%)")
    num_float_boundaries = sum(isinstance(start, float) or isinstance(end, float) for entry in new_data for label, start, end in entry['chunks'])
    logger.info(f"Non-integer chunks: {num_float_boundaries}")
    return new_data



def merge_sentences_for_bert_like(data: list, tokenizer: transformers.PreTrainedTokenizer, doc_key: str, verbose=True):
    """Merge sentences (according to `doc_key`) to documents in `data`. 
    Modify the corresponding start/end indexes in `chunks`, `relations`, `attributes`. 
    """
    if doc_key is None:
        logger.warning(f"Specifying `doc_key=None` will merge consecutive sentences as long as possible")
    
    max_len = tokenizer.model_max_length - 2
    new_data = []
    new_entry = {}
    for entry in tqdm.tqdm(data, disable=not verbose, ncols=100, desc="Merging sentences"):
        tokens = entry['tokens']
        num_sub_tokens = sum(len(tok) for tok in _tokenized2nested(tokens.raw_text, tokenizer))
        
        if len(new_entry) == 0:
            # Case (1): The first sentence of a document
            curr_start = 0
            new_entry['tokens'] = tokens
            cum_num_sub_tokens = num_sub_tokens
            
        elif (doc_key is None or entry[doc_key] == new_entry[doc_key]) and (cum_num_sub_tokens + num_sub_tokens <= max_len):
            # Case (2): The current sentence can be merged 
            curr_start = len(new_entry['tokens'])
            new_entry['tokens'] += tokens
            cum_num_sub_tokens += num_sub_tokens
            
        else:
            # Case (3): The current sentence cannot be merged
            new_data.append(new_entry)
            new_entry = {}
            
            curr_start = 0
            new_entry['tokens'] = tokens
            cum_num_sub_tokens = num_sub_tokens
        
        if 'chunks' in entry:
            new_chunks = [(label, start+curr_start, end+curr_start) for label, start, end in entry['chunks']]
            assert [new_entry['tokens'][s:e] for _, s, e in new_chunks] == [entry['tokens'][s:e] for _, s, e in entry['chunks']]
            if 'chunks' in new_entry:
                new_entry['chunks'].extend(new_chunks)
            else:
                new_entry['chunks'] = new_chunks
        
        if 'relations' in entry:
            new_relations = [(label, (h_label, h_start+curr_start, h_end+curr_start), (t_label, t_start+curr_start, t_end+curr_start)) 
                                 for label, (h_label, h_start, h_end), (t_label, t_start, t_end) in entry['relations']]
            if 'relations' in new_entry:
                new_entry['relations'].extend(new_relations)
            else:
                new_entry['relations'] = new_relations
        
        if 'attributes' in entry:
            new_attributes = [(label, (ck_label, ck_start+curr_start, ck_end+curr_start)) 
                                  for label, (ck_label, ck_start, ck_end) in entry['attributes']]
            if 'attributes' in new_entry:
                new_entry['attributes'].extend(new_attributes)
            else:
                new_entry['attributes'] = new_attributes
        
        if doc_key in new_entry:
            assert {k: v for k, v in entry.items() if k not in _IE_KEYS} == {k: v for k, v in new_entry.items() if k not in _IE_KEYS}
        else:
            new_entry.update({k: v for k, v in entry.items() if k not in _IE_KEYS})
    
    if len(new_entry) > 0:
        new_data.append(new_entry)
    return new_data
