# -*- coding: utf-8 -*-
import logging
import re
import unicodedata
from functools import cached_property
from typing import List

import numpy
import torch
import tqdm
import transformers
import truecase

from ..config import Config
from ..nn.functional import seq_lens2mask
from ..nn.modules import ScalarMix, SequenceGroupAggregating
from ..token import TokenSequence
from ..utils import assign_consecutive_to_buckets, find_ascending

logger = logging.getLogger(__name__)


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character.

    Copied from: transformers/tokenization_utils.py
    """
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


# Some abbreviations that are assumed without prefix space
_ABBREV = {"'m", "'re", "'s", "'ll", "'ve", "'d", "'t", "n't", "'", "''"}


def _subtokenize_word(word: str, tokenizer: transformers.PreTrainedTokenizer):
    if isinstance(tokenizer, transformers.BertTokenizer):
        return tokenizer.tokenize(word)

    elif isinstance(tokenizer, transformers.RobertaTokenizer):
        # All punctuations are assumed without prefix space
        if (len(word) == 1 and _is_punctuation(word)) or word.lower() in _ABBREV:
            return tokenizer.tokenize(word, add_prefix_space=False)
        else:
            return tokenizer.tokenize(word, add_prefix_space=True)

    elif isinstance(tokenizer, transformers.AlbertTokenizer):
        sub_tokens = tokenizer.tokenize(word)
        if len(word) == 1 or word.lower() in _ABBREV:
            if len(sub_tokens) == 1:
                assert sub_tokens[0][0] == "▁"
                sub_tokens[0] = sub_tokens[0][1:]
            elif len(sub_tokens) >= 2:
                assert sub_tokens[0] == "▁"
                sub_tokens = sub_tokens[1:]
        return sub_tokens

    else:
        raise ValueError(f"Invalid tokenizer {tokenizer}")


def _tokenized2nested(
    tokenized_raw_text: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_num_from_word: int = 5,
):
    nested_sub_tokens = []
    for word in tokenized_raw_text:
        sub_tokens = _subtokenize_word(word, tokenizer)
        if len(sub_tokens) == 0:
            # The tokenizer returns an empty list if the input is a space-like string
            sub_tokens = [tokenizer.unk_token]
        elif len(sub_tokens) > max_num_from_word:
            # The tokenizer may return a very long list if the input is a url
            sub_tokens = sub_tokens[:max_num_from_word]
        nested_sub_tokens.append(sub_tokens)

    return nested_sub_tokens


class BertLikeConfig(Config):
    def __init__(self, **kwargs):
        self.tokenizer: transformers.PreTrainedTokenizer = kwargs.pop("tokenizer")
        self.bert_like: transformers.PreTrainedModel = kwargs.pop("bert_like")
        self.hid_dim = self.bert_like.config.hidden_size
        self.num_layers = self.bert_like.config.num_hidden_layers

        self.arch = kwargs.pop("arch", "BERT")
        self.freeze = kwargs.pop("freeze", True)

        self.paired_inputs = kwargs.pop("paired_inputs", False)
        self.from_tokenized = kwargs.pop("from_tokenized", True)
        self.from_subtokenized = kwargs.pop("from_subtokenized", False)
        assert self.from_tokenized or (not self.from_subtokenized)
        self.pre_truncation = kwargs.pop("pre_truncation", False)
        self.group_agg_mode = kwargs.pop("group_agg_mode", "mean")
        self.mix_layers = kwargs.pop("mix_layers", "top")
        self.use_gamma = kwargs.pop("use_gamma", False)

        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        super().__init__(**kwargs)

    @property
    def name(self):
        return self.arch

    @property
    def out_dim(self):
        return self.hid_dim

    def __getstate__(self):
        state = self.__dict__.copy()
        state["bert_like"] = None
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
        sub_tok_ids = (
            [self.tokenizer.cls_token_id]
            + self.tokenizer.convert_tokens_to_ids(sub_tokens)
            + [self.tokenizer.sep_token_id]
        )
        example = {"sub_tok_ids": torch.tensor(sub_tok_ids)}

        if self.paired_inputs:
            # NOTE: `[SEP]` will be retained by tokenizer, instead of being tokenized
            sep_loc = sub_tokens.index(self.tokenizer.sep_token)
            sub_tok_type_ids = [self.sentence_A_id] * (sep_loc + 1 + 1) + [
                self.sentence_B_id
            ] * (
                len(sub_tokens) - sep_loc - 1 + 1
            )  # `[CLS]` and `[SEP]` at the ends
            example.update({"sub_tok_type_ids": torch.tensor(sub_tok_type_ids)})

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
            sub_tokens = [
                sub_tok for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok
            ]
            ori_indexes = [
                i for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok
            ]

        # Sequence longer than maximum length should be pre-processed
        assert len(sub_tokens) <= self.tokenizer.model_max_length - 2
        sub_tok_ids = (
            [self.tokenizer.cls_token_id]
            + self.tokenizer.convert_tokens_to_ids(sub_tokens)
            + [self.tokenizer.sep_token_id]
        )
        example = {"sub_tok_ids": torch.tensor(sub_tok_ids)}

        if not self.from_subtokenized:
            example.update({"ori_indexes": torch.tensor(ori_indexes)})

        if self.paired_inputs:
            # NOTE: `[SEP]` will be retained by tokenizer, instead of being tokenized
            sep_loc = sub_tokens.index(self.tokenizer.sep_token)
            sub_tok_type_ids = [self.sentence_A_id] * (sep_loc + 1 + 1) + [
                self.sentence_B_id
            ] * (
                len(sub_tokens) - sep_loc - 1 + 1
            )  # `[CLS]` and `[SEP]` at the ends
            example.update({"sub_tok_type_ids": torch.tensor(sub_tok_type_ids)})

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
        batch_sub_tok_ids = [ex["sub_tok_ids"] for ex in batch_ex]
        sub_tok_seq_lens = torch.tensor(
            [sub_tok_ids.size(0) for sub_tok_ids in batch_sub_tok_ids]
        )
        batch_sub_mask = seq_lens2mask(sub_tok_seq_lens)
        batch_sub_tok_ids = torch.nn.utils.rnn.pad_sequence(
            batch_sub_tok_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch = {"sub_tok_ids": batch_sub_tok_ids, "sub_mask": batch_sub_mask}

        if self.from_tokenized and (not self.from_subtokenized):
            batch_ori_indexes = [ex["ori_indexes"] for ex in batch_ex]
            batch_ori_indexes = torch.nn.utils.rnn.pad_sequence(
                batch_ori_indexes, batch_first=True, padding_value=-1
            )
            batch.update({"ori_indexes": batch_ori_indexes})

        if self.paired_inputs:
            batch_sub_tok_type_ids = [ex["sub_tok_type_ids"] for ex in batch_ex]
            batch_sub_tok_type_ids = torch.nn.utils.rnn.pad_sequence(
                batch_sub_tok_type_ids,
                batch_first=True,
                padding_value=self.sentence_A_id,
            )
            batch.update({"sub_tok_type_ids": batch_sub_tok_type_ids})

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
            self.group_aggregating = SequenceGroupAggregating(
                mode=config.group_agg_mode
            )
        if self.mix_layers.lower() == "trainable":
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

    def forward(
        self,
        sub_tok_ids: torch.LongTensor,
        sub_mask: torch.BoolTensor,
        sub_tok_type_ids: torch.BoolTensor = None,
        ori_indexes: torch.LongTensor = None,
    ):
        # last_hidden: (batch, sub_tok_step+2, hid_dim)
        # pooler_output: (batch, hid_dim)
        # hidden: a tuple of (batch, sub_tok_step+2, hid_dim)
        bert_outs = self.bert_like(
            input_ids=sub_tok_ids,
            attention_mask=(~sub_mask).long(),
            token_type_ids=sub_tok_type_ids,
            output_hidden_states=True,
        )

        # bert_hidden: (batch, sub_tok_step+2, hid_dim)
        if self.mix_layers.lower() == "trainable":
            bert_hidden = self.scalar_mix(bert_outs["hidden_states"])
        elif self.mix_layers.lower() == "top":
            bert_hidden = bert_outs["last_hidden_state"]
        elif self.mix_layers.lower() == "average":
            bert_hidden = sum(bert_outs["hidden_states"]) / len(
                bert_outs["hidden_states"]
            )

        if self.use_gamma:
            bert_hidden = self.gamma * bert_hidden

        # Remove `[CLS]` and `[SEP]`
        bert_hidden = bert_hidden[:, 1:-1]
        sub_mask = sub_mask[:, 2:]

        if hasattr(self, "group_aggregating"):
            # bert_hidden: (batch, tok_step, hid_dim)
            bert_hidden = self.group_aggregating(bert_hidden, ori_indexes)

        if self.output_hidden_states:
            all_bert_hidden = [hidden[:, 1:-1] for hidden in bert_outs["hidden_states"]]
            if hasattr(self, "group_aggregating"):
                all_bert_hidden = [
                    self.group_aggregating(hidden, ori_indexes)
                    for hidden in all_bert_hidden
                ]
            return (bert_hidden, all_bert_hidden)
        else:
            return bert_hidden


class BertLikePreProcessor(object):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        model_max_length: int = None,
        verbose: bool = True,
    ):
        self.tokenizer = tokenizer
        if model_max_length is None:
            self.model_max_length = tokenizer.model_max_length
        else:
            assert model_max_length <= tokenizer.model_max_length
            self.model_max_length = model_max_length
        self.verbose = verbose

    @classmethod
    def _truecase(cls, tokenized_raw_text: List[str]):
        """Get the truecased text.

        Original:  ['FULL', 'FEES', '1.875', 'REOFFER', '99.32', 'SPREAD', '+20', 'BP']
        Truecased: ['Full', 'fees', '1.875', 'Reoffer', '99.32', 'spread', '+20', 'BP']

        References
        ----------
        [1] https://github.com/google-research/bert/issues/223
        [2] https://github.com/daltonfury42/truecase
        """
        new_tokenized = tokenized_raw_text.copy()

        word_lst = [
            (w, idx)
            for idx, w in enumerate(new_tokenized)
            if all(c.isalpha() for c in w)
        ]
        lst = [w for w, _ in word_lst if re.match(r"\b[A-Z\.\-]+\b", w)]

        if len(lst) > 0 and len(lst) == len(word_lst):
            parts = truecase.get_true_case(" ".join(lst)).split()

            # the trucaser have its own tokenization ...
            # skip if the number of word dosen't match
            if len(parts) == len(word_lst):
                for (w, idx), nw in zip(word_lst, parts):
                    new_tokenized[idx] = nw

        return new_tokenized

    def truecase_for_data(self, data: List[dict]):
        num_truecased = 0
        for entry in tqdm.tqdm(
            data, disable=not self.verbose, ncols=100, desc="Truecasing data"
        ):
            tokens = entry["tokens"]
            tokenized_raw_text = tokens.raw_text
            new_tokenized_raw_text = self._truecase(tokenized_raw_text)
            if new_tokenized_raw_text != tokenized_raw_text:
                # Directly modify the `raw_text` attribute for each token; `text` attribute remains unchanged
                for tok, new_raw_text in zip(
                    entry["tokens"].token_list, new_tokenized_raw_text
                ):
                    tok.raw_text = new_raw_text
                num_truecased += 1

        logger.info(
            f"Truecased sequences: {num_truecased} ({num_truecased/len(data)*100 if len(data) else 0:.2f}%)"
        )
        return data

    @classmethod
    def _truncate(
        cls,
        tokens: TokenSequence,
        sub_tok_seq_lens: List[int],
        max_len: int,
        mode: str = "head+tail",
    ):
        if mode.lower() == "head-only":
            head_len, tail_len = max_len, 0
        elif mode.lower() == "tail-only":
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

    def truncate_for_data(self, data: List[dict], mode: str = "head+tail"):
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
        for entry in tqdm.tqdm(
            data, disable=not self.verbose, ncols=100, desc="Truncating data"
        ):
            tokens = entry["tokens"]
            nested_sub_tokens = _tokenized2nested(tokens.raw_text, self.tokenizer)
            sub_tok_seq_lens = [len(tok) for tok in nested_sub_tokens]

            if "paired_tokens" not in entry:
                # Case for single sentence
                max_len = self.model_max_length - 2
                if sum(sub_tok_seq_lens) > max_len:
                    entry["tokens"] = self._truncate(
                        tokens, sub_tok_seq_lens, max_len, mode=mode
                    )
                    num_truncated += 1

            else:
                # Case for paired sentences
                p_tokens = entry["paired_tokens"]
                p_nested_sub_tokens = _tokenized2nested(
                    p_tokens.raw_text, self.tokenizer
                )
                p_sub_tok_seq_lens = [len(tok) for tok in p_nested_sub_tokens]

                max_len = self.model_max_length - 3
                num_sub_toks, num_p_sub_toks = sum(sub_tok_seq_lens), sum(
                    p_sub_tok_seq_lens
                )

                if num_sub_toks + num_p_sub_toks > max_len:
                    # AD-HOC: Other ratio?
                    max_len1 = max_len // 2
                    if num_sub_toks <= max_len1:
                        entry["paired_tokens"] = self._truncate(
                            p_tokens,
                            p_sub_tok_seq_lens,
                            max_len - num_sub_toks,
                            mode=mode,
                        )
                    elif num_p_sub_toks <= max_len - max_len1:
                        entry["tokens"] = self._truncate(
                            tokens,
                            sub_tok_seq_lens,
                            max_len - num_p_sub_toks,
                            mode=mode,
                        )
                    else:
                        entry["tokens"] = self._truncate(
                            tokens, sub_tok_seq_lens, max_len1, mode=mode
                        )
                        entry["paired_tokens"] = self._truncate(
                            p_tokens, p_sub_tok_seq_lens, max_len - max_len1, mode=mode
                        )
                    num_truncated += 1

        logger.info(
            f"Truncated sequences: {num_truncated} ({num_truncated/len(data)*100 if len(data) else 0:.2f}%)"
        )
        return data

    def segment_sentences_for_data(
        self, data: List[dict], update_raw_idx: bool = False
    ):
        """Segment overlong sentences in `data`.

        Notes: Currently only supports entity recognition.
        """
        assert "relations" not in data[0]
        assert "attributes" not in data[0]

        max_len = self.model_max_length - 2
        new_data = []
        num_segmented = 0
        num_conflict = 0
        for raw_idx, entry in tqdm.tqdm(
            enumerate(data), disable=not self.verbose, ncols=100, desc="Segmenting data"
        ):
            tokens = entry["tokens"]
            nested_sub_tokens = _tokenized2nested(tokens.raw_text, self.tokenizer)
            sub_tok_seq_lens = [len(tok) for tok in nested_sub_tokens]

            num_sub_tokens = sum(sub_tok_seq_lens)
            if num_sub_tokens > max_len:
                # Avoid segmenting at positions inside a chunk
                is_segmentable = [True for _ in range(len(tokens))]
                for _, start, end in entry.get("chunks", []):
                    for i in range(start + 1, end):
                        is_segmentable[i] = False

                num_spans = int(num_sub_tokens / (max_len * 0.8)) + 1
                span_size = num_sub_tokens / num_spans

                cum_lens = numpy.cumsum(sub_tok_seq_lens).tolist()

                new_entries = []
                span_start = 0
                for i in range(1, num_spans + 1):
                    if i < num_spans:
                        _, span_end = find_ascending(cum_lens, span_size * i)
                        num_conflict += not is_segmentable[span_end]
                        while not is_segmentable[span_end]:
                            span_end -= 1
                    else:
                        span_end = len(tokens)
                    assert sum(sub_tok_seq_lens[span_start:span_end]) <= max_len

                    new_entry = {"tokens": tokens[span_start:span_end]}
                    if "chunks" in entry:
                        new_chunks = [
                            (label, start - span_start, end - span_start)
                            for label, start, end in entry["chunks"]
                            if span_start <= start and end <= span_end
                        ]
                        assert [new_entry["tokens"][s:e] for _, s, e in new_chunks] == [
                            entry["tokens"][s:e]
                            for _, s, e in entry["chunks"]
                            if span_start <= s and e <= span_end
                        ]
                        new_entry["chunks"] = new_chunks

                    others = {k: v for k, v in entry.items() if k not in new_entry}
                    new_entry.update(others)
                    new_entries.append(new_entry)
                    span_start = span_end

                assert len(new_entries) == num_spans
                assert [
                    text for entry in new_entries for text in entry["tokens"].raw_text
                ] == entry["tokens"].raw_text
                assert [
                    label for entry in new_entries for label, *_ in entry["chunks"]
                ] == [label for label, *_ in entry["chunks"]]
                num_segmented += 1

            else:
                new_entries = [entry]

            if update_raw_idx:
                for new_entry in new_entries:
                    new_entry["raw_idx"] = raw_idx

            new_data.extend(new_entries)

        logger.info(
            f"Segmented sequences: {num_segmented} ({num_segmented/len(data)*100 if len(data) else 0:.2f}%)"
        )
        logger.info(f"Conflict chunks: {num_conflict}")
        return new_data

    def _subtokenize(self, entry: dict, num_digits: int = 3):
        tokens = entry["tokens"]
        nested_sub_tokens = _tokenized2nested(tokens.raw_text, self.tokenizer)
        sub_tokens = [
            sub_tok for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok
        ]
        sub2ori_idx = [
            i if j == 0 else i + round(j / len(tok), ndigits=num_digits)
            for i, tok in enumerate(nested_sub_tokens)
            for j, sub_tok in enumerate(tok)
        ]
        sub2ori_idx.append(len(tokens))
        ori2sub_idx = [si for si, oi in enumerate(sub2ori_idx) if isinstance(oi, int)]

        # Warning: Re-construct `tokens`; this may yield `text` attribute different from the original ones
        new_entry = {
            "tokens": TokenSequence.from_tokenized_text(
                sub_tokens, **tokens._tokens_kwargs
            ),
            "sub2ori_idx": sub2ori_idx,
            "ori2sub_idx": ori2sub_idx,
        }

        if "tok2sent_idx" in entry:
            new_entry["tok2sent_idx"] = [
                entry["tok2sent_idx"][int(oi)] for oi in sub2ori_idx[:-1]
            ]
            assert len(new_entry["tok2sent_idx"]) == len(new_entry["tokens"])

        for ck_key in (key for key in entry.keys() if key.startswith("chunks")):
            new_entry[ck_key] = [
                (label, ori2sub_idx[start], ori2sub_idx[end])
                for label, start, end in entry[ck_key]
            ]

        for rel_key in (key for key in entry.keys() if key.startswith("relations")):
            ck_key = rel_key.replace("relations", "chunks")
            new_entry[rel_key] = [
                (
                    label,
                    (h_label, ori2sub_idx[h_start], ori2sub_idx[h_end]),
                    (t_label, ori2sub_idx[t_start], ori2sub_idx[t_end]),
                )
                for label, (h_label, h_start, h_end), (
                    t_label,
                    t_start,
                    t_end,
                ) in entry[rel_key]
            ]
            assert all(
                head in new_entry[ck_key] and tail in new_entry[ck_key]
                for label, head, tail in new_entry[rel_key]
            )

        for attr_key in (key for key in entry.keys() if key.startswith("attributes")):
            ck_key = attr_key.replace("attributes", "chunks")
            new_entry[attr_key] = [
                (label, (ck_label, ori2sub_idx[ck_start], ori2sub_idx[ck_end]))
                for label, (ck_label, ck_start, ck_end) in entry[attr_key]
            ]
            assert all(
                chunk in new_entry[ck_key] for label, chunk in new_entry[attr_key]
            )

        others = {k: v for k, v in entry.items() if k not in new_entry}
        new_entry.update(others)
        return new_entry

    def subtokenize_for_data(self, data: List[dict], num_digits: int = 3):
        """For word-level tokens, sub-tokenize the words with sub-word `tokenizer`.
        Modify the corresponding start/end indexes in `chunks`, `relations`, `attributes`.
        """
        new_data = []
        for entry in tqdm.tqdm(
            data,
            disable=not self.verbose,
            ncols=100,
            desc="Subtokenizing words in data",
        ):
            new_entry = self._subtokenize(entry, num_digits=num_digits)
            new_data.append(new_entry)

        num_subtokenized = sum(
            len(entry["sub2ori_idx"]) != len(entry["ori2sub_idx"]) for entry in new_data
        )
        logger.info(
            f"Sub-tokenized sequences: {num_subtokenized} ({num_subtokenized/len(data)*100 if len(data) else 0:.2f}%)"
        )
        return new_data

    @cached_property
    def tokenizer_sub_prefix(self):
        if isinstance(self.tokenizer, transformers.BertTokenizer):
            return "##"
        elif isinstance(self.tokenizer, transformers.RobertaTokenizer):
            return "Ġ"
        elif isinstance(self.tokenizer, transformers.AlbertTokenizer):
            return "▁"
        else:
            raise ValueError(f"Invalid tokenizer {self.tokenizer}")

    @cached_property
    def unk_ascii_characters(self):
        unk_chars = []
        for i in range(128):
            sub_tokens = _subtokenize_word(chr(i), self.tokenizer)
            if sub_tokens in ([], [self.tokenizer.unk_token]):
                unk_chars.append(chr(i))
        return unk_chars

    def _merge_enchars(self, entry: dict, num_digits: int = 3):
        tokens = entry["tokens"]
        sub_tokens = []
        curr_enchars = []
        for char in tokens.raw_text:
            if char.isascii() and (char not in self.unk_ascii_characters):
                curr_enchars.append(char)
            else:
                if len(curr_enchars) > 0:
                    curr_sub_tokens = self.tokenizer.tokenize("".join(curr_enchars))
                    assert (
                        "".join(
                            stok.replace(self.tokenizer_sub_prefix, "")
                            for stok in curr_sub_tokens
                        ).lower()
                        == "".join(curr_enchars).lower()
                    )
                    sub_tokens.extend(curr_sub_tokens)
                    curr_enchars = []

                char_subtokenized = self.tokenizer.tokenize(char)
                assert len(char_subtokenized) <= 1
                if len(char_subtokenized) == 0:
                    char_subtokenized = [self.tokenizer.unk_token]
                # assert char_subtokenized[0] in (char, char.lower(), tokenizer.unk_token)
                sub_tokens.extend(char_subtokenized)

        if len(curr_enchars) > 0:
            curr_sub_tokens = self.tokenizer.tokenize("".join(curr_enchars))
            assert (
                "".join(
                    stok.replace(self.tokenizer_sub_prefix, "")
                    for stok in curr_sub_tokens
                ).lower()
                == "".join(curr_enchars).lower()
            )
            sub_tokens.extend(curr_sub_tokens)

        sub_tokens_wo_prefix = [
            stok.replace(self.tokenizer_sub_prefix, "") for stok in sub_tokens
        ]
        assert len(
            "".join(sub_tokens_wo_prefix).replace(self.tokenizer.unk_token, "_")
        ) == len(tokens)
        ori2sub_idx = [
            i if j == 0 else i + round(j / len(stok), ndigits=num_digits)
            for i, stok in enumerate(sub_tokens_wo_prefix)
            for j in range(len(stok) if stok != self.tokenizer.unk_token else 1)
        ]
        ori2sub_idx.append(len(sub_tokens))
        sub2ori_idx = [oi for oi, si in enumerate(ori2sub_idx) if isinstance(si, int)]

        # Warning: Re-construct `tokens`; this may yield `text` attribute different from the original ones
        new_entry = {
            "tokens": TokenSequence.from_tokenized_text(
                sub_tokens, **tokens._tokens_kwargs
            ),
            "sub2ori_idx": sub2ori_idx,
            "ori2sub_idx": ori2sub_idx,
        }

        if "tok2sent_idx" in entry:
            new_entry["tok2sent_idx"] = [
                entry["tok2sent_idx"][oi] for oi in sub2ori_idx[:-1]
            ]
            assert len(new_entry["tok2sent_idx"]) == len(new_entry["tokens"])

        for ck_key in (key for key in entry.keys() if key.startswith("chunks")):
            new_entry[ck_key] = [
                (label, ori2sub_idx[start], ori2sub_idx[end])
                for label, start, end in entry[ck_key]
            ]

        for rel_key in (key for key in entry.keys() if key.startswith("relations")):
            ck_key = rel_key.replace("relations", "chunks")
            new_entry[rel_key] = [
                (
                    label,
                    (h_label, ori2sub_idx[h_start], ori2sub_idx[h_end]),
                    (t_label, ori2sub_idx[t_start], ori2sub_idx[t_end]),
                )
                for label, (h_label, h_start, h_end), (
                    t_label,
                    t_start,
                    t_end,
                ) in entry[rel_key]
            ]
            assert all(
                head in new_entry[ck_key] and tail in new_entry[ck_key]
                for label, head, tail in new_entry[rel_key]
            )

        for attr_key in (key for key in entry.keys() if key.startswith("attributes")):
            ck_key = attr_key.replace("attributes", "chunks")
            new_entry[attr_key] = [
                (label, (ck_label, ori2sub_idx[ck_start], ori2sub_idx[ck_end]))
                for label, (ck_label, ck_start, ck_end) in entry[attr_key]
            ]
            assert all(
                chunk in new_entry[ck_key] for label, chunk in new_entry[attr_key]
            )

        others = {k: v for k, v in entry.items() if k not in new_entry}
        new_entry.update(others)
        return new_entry

    def merge_enchars_for_data(self, data: List[dict], num_digits: int = 3):
        """For character-level tokens, merge the consecutive English characters with sub-word `tokenizer`.
        Modify the corresponding start/end indexes in `chunks`, `relations`, `attributes`.

        Warning: Any consecutive English characters will be first merged as a word and then subtokenized into sub-words,
        even if the characters originally contain multiple words.
        Note that the original word boundaries (i.e., spaces) are missing, and not retrievable.
        """
        new_data = []
        for entry in tqdm.tqdm(
            data, disable=not self.verbose, ncols=100, desc="Merging characters in data"
        ):
            new_entry = self._merge_enchars(entry, num_digits=num_digits)
            new_data.append(new_entry)

        num_merged = sum(
            len(entry["sub2ori_idx"]) != len(entry["ori2sub_idx"]) for entry in new_data
        )
        logger.info(
            f"Merged sequences: {num_merged} ({num_merged/len(data)*100 if len(data) else 0:.2f}%)"
        )
        num_float_boundaries = sum(
            isinstance(start, float) or isinstance(end, float)
            for entry in new_data
            for label, start, end in entry["chunks"]
        )
        logger.info(f"Non-integer chunks: {num_float_boundaries}")
        return new_data

    def _merge_sentences_for_doc(
        self, doc: List[dict], doc_sub_lens: List[int], mode: str = "min_max_length"
    ):
        max_len = self.model_max_length - 2

        # Greedy
        cum_sub_lens = 0
        buckets = [0]
        k = 0
        for sub_len in doc_sub_lens:
            if cum_sub_lens + sub_len <= max_len:
                cum_sub_lens += sub_len
                buckets[k] += 1
            else:
                k += 1
                cum_sub_lens = sub_len
                buckets.append(1)

        # Minimize the maximum length of merged sequences
        if mode.lower() == "min_max_length" and max(doc_sub_lens) <= max_len:
            buckets = assign_consecutive_to_buckets(doc_sub_lens, len(buckets))

        assert all(b > 0 for b in buckets)
        assert sum(buckets) == len(doc)

        new_doc = []
        i = 0
        for num_entries in buckets:
            new_entry = {}
            for sidx in range(num_entries):
                entry = doc[i]
                if len(new_entry) == 0:
                    curr_start = 0
                    new_entry["tokens"] = entry["tokens"]
                    new_entry["tok2sent_idx"] = [
                        sidx for _ in range(len(entry["tokens"]))
                    ]
                else:
                    curr_start = len(new_entry["tokens"])
                    new_entry["tokens"] += entry["tokens"]
                    new_entry["tok2sent_idx"].extend(
                        [sidx for _ in range(len(entry["tokens"]))]
                    )

                for ck_key in (key for key in entry.keys() if key.startswith("chunks")):
                    new_chunks = [
                        (label, start + curr_start, end + curr_start)
                        for label, start, end in entry[ck_key]
                    ]
                    assert [new_entry["tokens"][s:e] for _, s, e in new_chunks] == [
                        entry["tokens"][s:e] for _, s, e in entry[ck_key]
                    ]
                    if ck_key in new_entry:
                        new_entry[ck_key].extend(new_chunks)
                    else:
                        new_entry[ck_key] = new_chunks

                for rel_key in (
                    key for key in entry.keys() if key.startswith("relations")
                ):
                    ck_key = rel_key.replace("relations", "chunks")
                    new_relations = [
                        (
                            label,
                            (h_label, h_start + curr_start, h_end + curr_start),
                            (t_label, t_start + curr_start, t_end + curr_start),
                        )
                        for label, (h_label, h_start, h_end), (
                            t_label,
                            t_start,
                            t_end,
                        ) in entry[rel_key]
                    ]
                    assert all(
                        head in new_entry[ck_key] and tail in new_entry[ck_key]
                        for label, head, tail in new_relations
                    )
                    if rel_key in new_entry:
                        new_entry[rel_key].extend(new_relations)
                    else:
                        new_entry[rel_key] = new_relations

                for attr_key in (
                    key for key in entry.keys() if key.startswith("attributes")
                ):
                    ck_key = attr_key.replace("attributes", "chunks")
                    new_attributes = [
                        (label, (ck_label, ck_start + curr_start, ck_end + curr_start))
                        for label, (ck_label, ck_start, ck_end) in entry[attr_key]
                    ]
                    assert all(
                        chunk in new_entry[ck_key] for label, chunk in new_attributes
                    )
                    if attr_key in new_entry:
                        new_entry[attr_key].extend(new_attributes)
                    else:
                        new_entry[attr_key] = new_attributes

                # Check other fields (e.g., meta info) consistent?
                others = {k: v for k, v in entry.items() if k not in new_entry}
                new_entry.update(others)
                i += 1

            new_doc.append(new_entry)

        return new_doc

    def merge_sentences_for_data(
        self, data: List[dict], doc_key: str, mode: str = "min_max_length"
    ):
        """Merge sentences (according to `doc_key`) to documents in `data`.
        Modify the corresponding start/end indexes in `chunks`, `relations`, `attributes`.

        Merging methods:
            1. greedy;
            2. min_max_length: minimize the maximum length of merged sequences for each document.
        """
        if doc_key is None:
            logger.warning(
                f"Specifying `doc_key=None` will regard all sentences as a document"
            )

        max_len = self.model_max_length - 2
        new_data = []
        doc, doc_sub_lens = [], []
        num_overlong = 0
        for entry in tqdm.tqdm(
            data, disable=not self.verbose, ncols=100, desc="Merging sentences"
        ):
            tokens = entry["tokens"]
            num_sub_tokens = sum(
                len(tok) for tok in _tokenized2nested(tokens.raw_text, self.tokenizer)
            )
            num_overlong += num_sub_tokens > max_len

            if doc_key is None or len(doc) == 0 or entry[doc_key] == doc[0][doc_key]:
                doc.append(entry)
                doc_sub_lens.append(num_sub_tokens)
            else:
                new_doc = self._merge_sentences_for_doc(doc, doc_sub_lens, mode=mode)
                new_data.extend(new_doc)

                doc, doc_sub_lens = [entry], [num_sub_tokens]

        if len(doc) > 0:
            new_doc = self._merge_sentences_for_doc(doc, doc_sub_lens, mode=mode)
            new_data.extend(new_doc)

        if num_overlong > 0:
            logger.warning(f"Overlong sentences: {num_overlong}")
        return new_data


class BertLikePostProcessor(object):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def restore_for_data(self, data: List[dict]):
        """Restore `chunks`, `relations`, `attributes` that are pre-processed by `merge_sentences_for_data` and/or `subtokenize_for_data`.

        Notes: Currently NOT supports `merge_enchars_for_data`.
        """
        oori_data = []
        for entry in tqdm.tqdm(
            data, disable=not self.verbose, ncols=100, desc="Restoring"
        ):
            num_tokens = len(entry["tokens"])
            sub2ori_idx = entry.get("sub2ori_idx", [i for i in range(num_tokens + 1)])
            ori2sub_idx = entry.get("ori2sub_idx", [i for i in range(num_tokens + 1)])
            tok2sent_idx = entry.get("tok2sent_idx", [0 for _ in range(num_tokens)])

            # Restore start/end at the document leval
            ori_entry = {}
            for ck_key in (key for key in entry.keys() if key.startswith("chunks")):
                ori_entry[ck_key] = [
                    (label, sub2ori_idx[start], sub2ori_idx[end])
                    for label, start, end in entry[ck_key]
                ]

            for rel_key in (key for key in entry.keys() if key.startswith("relations")):
                ck_key = rel_key.replace("relations", "chunks")
                ori_entry[rel_key] = [
                    (
                        label,
                        (h_label, sub2ori_idx[h_start], sub2ori_idx[h_end]),
                        (t_label, sub2ori_idx[t_start], sub2ori_idx[t_end]),
                    )
                    for label, (h_label, h_start, h_end), (
                        t_label,
                        t_start,
                        t_end,
                    ) in entry[rel_key]
                ]
                assert all(
                    head in ori_entry[ck_key] and tail in ori_entry[ck_key]
                    for label, head, tail in ori_entry[rel_key]
                )

            for attr_key in (
                key for key in entry.keys() if key.startswith("attributes")
            ):
                ck_key = attr_key.replace("attributes", "chunks")
                ori_entry[attr_key] = [
                    (label, (ck_label, sub2ori_idx[ck_start], sub2ori_idx[ck_end]))
                    for label, (ck_label, ck_start, ck_end) in entry[attr_key]
                ]
                assert all(
                    chunk in ori_entry[ck_key] for label, chunk in ori_entry[attr_key]
                )

            # Restore to the sentence level
            # Restore `tok2sent_idx` first, because `tok2sent_idx` has been changed to sub-token indexes in pre-processing
            tok2sent_idx = [tok2sent_idx[si] for si in ori2sub_idx[:-1]]
            sent_starts = [
                tok2sent_idx.index(sidx) for sidx in range(tok2sent_idx[-1] + 1)
            ]
            oori_doc = [
                {key: [] for key in ori_entry.keys()}
                for _ in range(tok2sent_idx[-1] + 1)
            ]

            for ck_key in (key for key in ori_entry.keys() if key.startswith("chunks")):
                for label, start, end in ori_entry[ck_key]:
                    sidx = tok2sent_idx[start]
                    assert tok2sent_idx[end - 1] == sidx
                    curr_start = sent_starts[sidx]
                    oori_doc[sidx][ck_key].append(
                        (label, start - curr_start, end - curr_start)
                    )

            for rel_key in (
                key for key in ori_entry.keys() if key.startswith("relations")
            ):
                ck_key = rel_key.replace("relations", "chunks")
                for (
                    label,
                    (h_label, h_start, h_end),
                    (t_label, t_start, t_end),
                ) in ori_entry[rel_key]:
                    sidx = tok2sent_idx[h_start]
                    assert tok2sent_idx[t_start] == sidx
                    curr_start = sent_starts[sidx]
                    oori_doc[sidx][rel_key].append(
                        (
                            label,
                            (h_label, h_start - curr_start, h_end - curr_start),
                            (t_label, t_start - curr_start, t_end - curr_start),
                        )
                    )
                assert all(
                    head in oori_entry[ck_key] and tail in oori_entry[ck_key]
                    for oori_entry in oori_doc
                    for label, head, tail in oori_entry[rel_key]
                )

            for attr_key in (
                key for key in ori_entry.keys() if key.startswith("attributes")
            ):
                ck_key = attr_key.replace("attributes", "chunks")
                for label, (ck_label, ck_start, ck_end) in ori_entry[attr_key]:
                    sidx = tok2sent_idx[ck_start]
                    curr_start = sent_starts[sidx]
                    oori_doc[sidx][attr_key].append(
                        (label, (ck_label, ck_start - curr_start, ck_end - curr_start))
                    )
                assert all(
                    chunk in oori_entry[ck_key]
                    for oori_entry in oori_doc
                    for label, chunk in oori_entry[attr_key]
                )

            oori_data.extend(oori_doc)

        return oori_data
