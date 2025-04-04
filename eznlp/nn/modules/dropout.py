# -*- coding: utf-8 -*-
import torch


class CombinedDropout(torch.nn.Module):
    def __init__(self, p: float = 0.0, word_p: float = 0.05, locked_p: float = 0.5):
        super().__init__()
        if p > 0:
            self.dropout = torch.nn.Dropout(p)
        if word_p > 0:
            self.word_dropout = WordDropout(word_p)
        if locked_p > 0:
            self.locked_dropout = LockedDropout(locked_p)

    def forward(self, x: torch.Tensor):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if hasattr(self, "word_dropout"):
            x = self.word_dropout(x)
        if hasattr(self, "locked_dropout"):
            x = self.locked_dropout(x)
        return x


class LockedDropout(torch.nn.Module):
    """
    Locked (or variational) dropout, which drops out all elements (over steps)
    in randomly chosen dimension of embeddings or hidden states.

    References
    ----------
    https://github.com/flairNLP/flair/blob/master/flair/nn.py
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        if (not self.training) or self.p == 0:
            return x

        # x: (batch, step, hidden)
        m = torch.empty(x.size(0), 1, x.size(2), device=x.device).bernoulli(
            p=1 - self.p
        )
        return x * m / (1 - self.p)

    def extra_repr(self):
        return f"p={self.p}"


class WordDropout(torch.nn.Module):
    """
    Word dropout, which drops out all elements (over embeddings or hidden states)
    in randomly chosen word.

    References
    ----------
    https://github.com/flairNLP/flair/blob/master/flair/nn.py
    """

    def __init__(self, p: float = 0.05, keep_exp=False):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p
        self.keep_exp = keep_exp

    def forward(self, x: torch.Tensor):
        if (not self.training) or self.p == 0:
            return x

        # x: (batch, step, hidden)
        m = torch.empty(x.size(0), x.size(1), 1, device=x.device).bernoulli(
            p=1 - self.p
        )
        # Do NOT adjust values to keep the expectation, according to flair implementation.
        if self.keep_exp:
            return x * m / (1 - self.p)
        else:
            return x * m

    def extra_repr(self):
        return f"p={self.p}, keep_exp={self.keep_exp}"
