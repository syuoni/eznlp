# -*- coding: utf-8 -*-
from collections import Counter
from typing import List


class Vocab(object):
    """A vocabulary object.

    This class is modified based on `torchtext.vocab.Vocab` of version 0.8.1.
    """

    def __init__(
        self,
        counter: Counter,
        max_size=None,
        min_freq=1,
        specials=("<unk>", "<pad>"),
        specials_first=True,
    ):
        self.freqs = counter

        counter = counter.copy()
        min_freq = max(min_freq, 1)

        itos = []
        if specials_first:
            itos = list(specials)
            # only extend max size if specials are prepended
            max_size = None if max_size is None else max_size + len(specials)

        # frequencies of special tokens are not counted when building vocabulary in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(itos) == max_size:
                break
            itos.append(word)

        if not specials_first:
            itos.extend(list(specials))

        self.itos = itos

    @property
    def itos(self):
        return self._itos

    @itos.setter
    def itos(self, itos: List[str]):
        self._itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get("<unk>"))

    def __len__(self):
        return len(self.itos)

    def lookup_indices(self, tokens):
        return [self.__getitem__(token) for token in tokens]
