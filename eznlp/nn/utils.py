# -*- coding: utf-8 -*-
import torch


def pad_seqs(seqs, padding_value=0.0, length=None):
    """Pad a list of list, making it prepared as a tensor. 
    """
    # Alternative: `torch.nn.utils.rnn.pad_sequence` which pads a list of tensors.
    maxlen = max(len(s) for s in seqs)
    length = maxlen if length is None else max(length, maxlen)
    
    if isinstance(seqs[0][0], list):
        # Each sequence is a sequence of lists (of values). 
        padding_item = [padding_value] * len(seqs[0][0])
    else:
        # Each sequence is a sequence of indexes
        padding_item = padding_value
    
    return [s + [padding_item for _ in range(length-len(s))] for s in seqs]


def unpad_seqs(seqs, seq_lens):
    """Retrieve the list of list from a padded tensor. 
    
    Returns
    -------
    list
        A list of list of values. 
    """
    if isinstance(seq_lens, torch.LongTensor):
        seq_lens = seq_lens.cpu().tolist()
    
    return [seq[:seq_len] for seq, seq_len in zip(seqs.cpu().tolist(), seq_lens)]



def _nonlinearity2activation(nonlinearity: str, **kwargs):
    if nonlinearity.lower() in ('linear', 'identidy'):
        return torch.nn.Identidy()
    elif nonlinearity.lower() == 'sigmoid':
        return torch.nn.Sigmoid()
    elif nonlinearity.lower() == 'tanh':
        return torch.nn.Tanh()
    elif nonlinearity.lower() == 'relu':
        return torch.nn.ReLU()
    elif nonlinearity.lower() in ('leaky_relu', 'leakyrelu'):
        return torch.nn.LeakyReLU(**kwargs)
    elif nonlinearity.lower() == 'glu':
        return torch.nn.GLU(**kwargs)
    else:
        raise ValueError(f"Invalid nonlinearity {nonlinearity}")
