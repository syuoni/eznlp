# -*- coding: utf-8 -*-
import logging
import torch
from torchtext.experimental.vectors import Vectors

PRETRAINED_VEC_EPS = 1e-6
logger = logging.getLogger(__name__)



def _fetch_token_pretrained_vector(token: str, pretrained_vectors: Vectors):
    tried_set = set()
    for possible_token in [token, token.lower(), token.title(), token.upper()]:
        if possible_token in tried_set:
            continue
        
        pretrained_vec = pretrained_vectors[possible_token]
        if pretrained_vec.abs().max().item() > PRETRAINED_VEC_EPS:
            return pretrained_vec
        tried_set.add(possible_token)
        
    return None


def reinit_embedding_(emb: torch.nn.Embedding, itos=None, pretrained_vectors=None, unk_vector='uniform'):
    emb_dim = emb.weight.size(1)
    uniform_range = (3 / emb_dim) ** 0.5
    
    if (itos is not None) and (pretrained_vectors is not None):
        assert emb.weight.size(0) == len(itos)
        assert emb.weight.size(1) == pretrained_vectors['<unk>'].size(0)
        
        oov_tokens = []
        acc_vec_abs = 0
        for idx, tok in enumerate(itos):
            pretrained_vec = _fetch_token_pretrained_vector(tok, pretrained_vectors)
            
            if pretrained_vec is None:
                oov_tokens.append(tok)
                if unk_vector.lower() == 'uniform':
                    torch.nn.init.uniform_(emb.weight.data[idx], -uniform_range, uniform_range)
                elif unk_vector.lower() == 'zeros':
                    torch.nn.init.zeros_(emb.weight.data[idx])
            else:
                acc_vec_abs += pretrained_vec.abs().mean().item()
                emb.weight.data[idx].copy_(pretrained_vec)
        
        if emb.padding_idx is not None:
            torch.nn.init.zeros_(emb.weight.data[emb.padding_idx])
        ave_vec_abs = acc_vec_abs / (len(itos) - len(oov_tokens))
        
        if unk_vector.lower() == 'uniform':
            oov_vec_abs = uniform_range / 2
        elif unk_vector.lower() == 'zeros':
            oov_vec_abs = 0.0
        
        logger.info(
            "Embedding initialization \n"
            f"OOV tokens: {len(oov_tokens)} ({len(oov_tokens)/len(itos)*100:.2f}%) \n"
            f"Pretrained      vector average absolute value: {ave_vec_abs:.4f} \n"
            f"OOV initialized vector average absolute value: {oov_vec_abs:.4f}"
        )
        return oov_tokens
    
    else:
        torch.nn.init.uniform_(emb.weight.data, -uniform_range, uniform_range)
        if emb.padding_idx is not None:
            torch.nn.init.zeros_(emb.weight.data[emb.padding_idx])
        return None
    

def reinit_layer_(layer: torch.nn.Module, nonlinearity='relu'):
    """
    Refs: 
    [1] Xavier Glorot and Yoshua Bengio. 2010. Understanding the difficulty of 
    training deep feedforward neural networks. 
    [2] Kaiming He, et al. 2015. Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification.
    """
    for name, param in layer.named_parameters():
        if name.startswith('bias'):
            torch.nn.init.zeros_(param.data)
        elif name.startswith('weight'):
            if nonlinearity.lower() in ('relu', 'leaky_relu'):
                torch.nn.init.kaiming_uniform_(param.data, nonlinearity=nonlinearity)
            else:
                torch.nn.init.xavier_uniform_(param.data, 
                                              gain=torch.nn.init.calculate_gain(nonlinearity))
            
        else:
            raise TypeError(f"Invalid Layer {layer}")
    
    
def reinit_transformer_encoder_layer_(tf_encoder_layer: torch.nn.TransformerEncoderLayer):
    for name, param in tf_encoder_layer.named_parameters():
        if name.startswith('norm'):
            pass
        elif name.endswith('bias'):
            torch.nn.init.zeros_(param.data)
        elif name.endswith('weight'):
            torch.nn.init.xavier_uniform_(param.data, 
                                          gain=torch.nn.init.calculate_gain('linear'))
        else:
            raise TypeError(f"Invalid TransformerEncoderLayer {tf_encoder_layer}")

    
def reinit_lstm_(lstm: torch.nn.LSTM):
    '''
    W_i: (W_ii|W_if|W_ig|W_io) of shape (hid_size*4, in_size)
    W_h: (W_hi|W_hf|W_hg|W_ho) of shape (hid_size*4, hid_size)
    W_{i, h}{i, f, o} use `sigmoid` activation function.
    W_{i, h}{g} use `tanh` activation function.
    
    The LSTM forget gate bias should be initialized to be 1. 
    
    Refs: 
    [1] Xavier Glorot and Yoshua Bengio. 2010. Understanding the difficulty of 
    training deep feedforward neural networks. 
    [2] Rafal Jozefowicz, et al. 2015. An empirical exploration of recurrent
    network architectures. 
    '''
    for name, param in lstm.named_parameters():
        if name.startswith('bias'):
            hid_size = param.size(0) // 4
            torch.nn.init.zeros_(param.data)
            torch.nn.init.ones_(param.data[hid_size:(hid_size*2)])
        elif name.startswith('weight'):
            hid_size = param.size(0) // 4
            for i, nonlinearity in enumerate(['sigmoid', 'sigmoid', 'tanh', 'sigmoid']):
                torch.nn.init.xavier_uniform_(param.data[(hid_size*i):(hid_size*(i+1))], 
                                              gain=torch.nn.init.calculate_gain(nonlinearity))
        else:
            raise TypeError(f"Invalid LSTM {lstm}")


def reinit_gru_(gru: torch.nn.GRU):
    '''
    W_i: (W_ir|W_iz|W_in) of shape (hid_size, in_size)
    W_h: (W_hr|W_hz|W_hn) of shape (hid_size, hid_size)
    W_{i, h}{r, z} use `sigmoid` activation function.
    W_{i, h}{n} use `tanh` activation function.
    
    Refs: 
    [1] Xavier Glorot and Yoshua Bengio. 2010. Understanding the difficulty of 
    training deep feedforward neural networks. 
    '''
    for name, param in gru.named_parameters():
        if name.startswith('bias'):
            torch.nn.init.zeros_(param.data)
        elif name.startswith('weight'):
            hid_size = param.size(0) // 3
            for i, nonlinearity in enumerate(['sigmoid', 'sigmoid', 'tanh']):
                torch.nn.init.xavier_uniform_(param.data[(hid_size*i):(hid_size*(i+1))], 
                                              gain=torch.nn.init.calculate_gain(nonlinearity))
        else:
            raise TypeError(f"Invalid GRU {gru}")
