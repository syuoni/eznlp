# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from torchtext.experimental.vocab import Vocab
    

def _fetch_token_id(token: str, vocab: Vocab):
    tried_set = set()
    unk_id = vocab['<unk>']
    for possible_token in [token, token.lower(), token.title(), token.upper()]:
        if possible_token in tried_set:
            continue
        
        token_id = vocab[possible_token]
        if token_id != unk_id:
            return token_id
        tried_set.add(possible_token)
        
    return unk_id


def pad_seqs(seqs, padding_value=0.0, length=None):
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
    """
    Returns a List of List of values.
    """
    return [seq[:seq_len] for seq, seq_len in zip(seqs.cpu().tolist(), seq_lens.cpu().tolist())]


class TensorWrapper(object):
    def __init__(self, **kwargs):
        for possible_name, possible_attr in kwargs.items():
            if possible_attr is None or isinstance(possible_attr, (Tensor, TensorWrapper)):
                pass
            elif isinstance(possible_attr, list):
                assert all(isinstance(sub_attr, (Tensor, TensorWrapper, str)) for sub_attr in possible_attr)
            elif isinstance(possible_attr, dict):
                assert all(isinstance(sub_attr, (Tensor, TensorWrapper, str)) for sub_attr in possible_attr.values())
            else:
                raise TypeError(f"Invalid input to `TensorWrapper`: {possible_attr}")
                
            setattr(self, possible_name, possible_attr)
            
            
    def _apply_to_tensors(self, func):
        """
        This function must return `self`.
        """
        for attr_name in self.__dict__:
            attr = getattr(self, attr_name)
            if isinstance(attr, Tensor):
                setattr(self, attr_name, func(attr))
            elif isinstance(attr, TensorWrapper):
                setattr(self, attr_name, attr._apply_to_tensors(func))
            elif isinstance(attr, list):
                if len(attr) == 0 or isinstance(attr[0], Tensor):
                    setattr(self, attr_name, [func(x) for x in attr])
                else:
                    setattr(self, attr_name, [x._apply_to_tensors(func) for x in attr])
            elif isinstance(attr, dict):
                if len(attr) == 0 or isinstance(list(attr.values())[0], Tensor):
                    setattr(self, attr_name, {k: func(v) for k, v in attr.items()})
                else:
                    setattr(self, attr_name, {k: v._apply_to_tensors(func) for k, v in attr.items()})
                    
        return self
        
    def pin_memory(self):
        return self._apply_to_tensors(lambda x: x.pin_memory())
    
    def to(self, *args, **kwargs):
        return self._apply_to_tensors(lambda x: x.to(*args, **kwargs))
        

def _seq_lens2mask(shape, seq_lens):
    """
    shape: Tuple (batch_size, step)
    seq_lens: Tensor (batch_size, )
    """
    return torch.arange(shape[1], device=seq_lens.device).repeat(shape[0], 1) >= seq_lens.unsqueeze(1)


class Batch(TensorWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        
    def build_masks(self, mask_config):
        for mask_name, (mask_shape, lens) in mask_config.items():
            setattr(self, mask_name, _seq_lens2mask(mask_shape, lens))
    
    
    def __repr__(self):
        return "Batch with attributes: {}".format(", ".join(self.__dict__))
    
    def __str__(self):
        return "Batch with attributes: {}".format(", ".join(self.__dict__))

