# -*- coding: utf-8 -*-
import functools
import logging
from typing import List

import torch
import transformers

logger = logging.getLogger(__name__)


def reinit_embedding_(embedding: torch.nn.Embedding):
    uniform_range = (3 / embedding.weight.size(1)) ** 0.5

    torch.nn.init.uniform_(embedding.weight.data, -uniform_range, uniform_range)
    if embedding.padding_idx is not None:
        torch.nn.init.zeros_(embedding.weight.data[embedding.padding_idx])

    logger.info(
        "Embeddings initialized with randomized vectors \n"
        f"Vector average absolute value: {uniform_range/2:.4f}"
    )


def reinit_embedding_by_pretrained_(
    embedding: torch.nn.Embedding, itos: List[str], vectors, oov_init: str = "zeros"
):
    assert embedding.weight.size(0) == len(itos)
    assert embedding.weight.size(1) == vectors.emb_dim
    uniform_range = (3 / embedding.weight.size(1)) ** 0.5

    oov = []
    acc_vec_abs = 0
    for idx, tok in enumerate(itos):
        pretrained_vec = vectors.lookup(tok)
        if pretrained_vec is not None:
            acc_vec_abs += pretrained_vec.abs().mean().item()
            embedding.weight.data[idx].copy_(pretrained_vec)
        else:
            oov.append(tok)
            if oov_init.lower() == "zeros":
                torch.nn.init.zeros_(embedding.weight.data[idx])
            elif oov_init.lower() == "uniform":
                torch.nn.init.uniform_(
                    embedding.weight.data[idx], -uniform_range, uniform_range
                )

    if embedding.padding_idx is not None:
        torch.nn.init.zeros_(embedding.weight.data[embedding.padding_idx])
    ave_vec_abs = acc_vec_abs / (len(itos) - len(oov))

    if oov_init.lower() == "zeros":
        oov_vec_abs = 0.0
    elif oov_init.lower() == "uniform":
        oov_vec_abs = uniform_range / 2

    logger.info(
        "Embeddings initialized with pretrained vectors \n"
        f"OOV tokens: {len(oov)} ({len(oov)/len(itos)*100:.2f}%) \n"
        f"Pretrained      vector average absolute value: {ave_vec_abs:.4f} \n"
        f"OOV initialized vector average absolute value: {oov_vec_abs:.4f}"
    )
    return oov


def reinit_vector_parameter_(param: torch.nn.Parameter):
    assert param.dim() == 1
    uniform_range = (3 / param.size(0)) ** 0.5
    torch.nn.init.uniform_(param.data, -uniform_range, uniform_range)


def reinit_layer_(layer: torch.nn.Module, nonlinearity="relu"):
    """
    Refs:
    [1] Xavier Glorot and Yoshua Bengio. 2010. Understanding the difficulty of
    training deep feedforward neural networks.
    [2] Kaiming He, et al. 2015. Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification.
    """
    for name, param in layer.named_parameters():
        if name.startswith("bias"):
            torch.nn.init.zeros_(param.data)
        elif name.startswith("weight"):
            if nonlinearity.lower() in ("relu", "leaky_relu"):
                torch.nn.init.kaiming_uniform_(param.data, nonlinearity=nonlinearity)
            elif nonlinearity.lower() in ("glu",):
                torch.nn.init.xavier_uniform_(
                    param.data, gain=torch.nn.init.calculate_gain("sigmoid")
                )
            else:
                torch.nn.init.xavier_uniform_(
                    param.data, gain=torch.nn.init.calculate_gain(nonlinearity)
                )
        else:
            raise TypeError(f"Invalid Layer {layer}")


def reinit_transformer_encoder_layer_(
    tf_encoder_layer: torch.nn.TransformerEncoderLayer,
):
    for name, param in tf_encoder_layer.named_parameters():
        if name.startswith("norm"):
            pass
        elif name.endswith("bias"):
            torch.nn.init.zeros_(param.data)
        elif name.endswith("weight"):
            torch.nn.init.xavier_uniform_(
                param.data, gain=torch.nn.init.calculate_gain("linear")
            )
        else:
            raise TypeError(f"Invalid TransformerEncoderLayer {tf_encoder_layer}")


def reinit_lstm_(lstm: torch.nn.LSTM):
    """
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
    """
    for name, param in lstm.named_parameters():
        if name.startswith("bias"):
            hid_size = param.size(0) // 4
            torch.nn.init.zeros_(param.data)
            torch.nn.init.ones_(param.data[hid_size : (hid_size * 2)])
        elif name.startswith("weight"):
            hid_size = param.size(0) // 4
            for i, nonlinearity in enumerate(["sigmoid", "sigmoid", "tanh", "sigmoid"]):
                torch.nn.init.xavier_uniform_(
                    param.data[(hid_size * i) : (hid_size * (i + 1))],
                    gain=torch.nn.init.calculate_gain(nonlinearity),
                )
        else:
            raise TypeError(f"Invalid LSTM {lstm}")


def reinit_gru_(gru: torch.nn.GRU):
    """
    W_i: (W_ir|W_iz|W_in) of shape (hid_size, in_size)
    W_h: (W_hr|W_hz|W_hn) of shape (hid_size, hid_size)
    W_{i, h}{r, z} use `sigmoid` activation function.
    W_{i, h}{n} use `tanh` activation function.

    Refs:
    [1] Xavier Glorot and Yoshua Bengio. 2010. Understanding the difficulty of
    training deep feedforward neural networks.
    """
    for name, param in gru.named_parameters():
        if name.startswith("bias"):
            torch.nn.init.zeros_(param.data)
        elif name.startswith("weight"):
            hid_size = param.size(0) // 3
            for i, nonlinearity in enumerate(["sigmoid", "sigmoid", "tanh"]):
                torch.nn.init.xavier_uniform_(
                    param.data[(hid_size * i) : (hid_size * (i + 1))],
                    gain=torch.nn.init.calculate_gain(nonlinearity),
                )
        else:
            raise TypeError(f"Invalid GRU {gru}")


def _reinit_bert_like_module(module: torch.nn.Module, std: float = 0.02):
    """Initializes each BERT module.
    The original Google-implemented BERT uses truncated_normal for initialization
    cf https://github.com/pytorch/pytorch/pull/5617
    cf https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/init_ops.py

    Implementation follows: `transformers/models/bert/modeling_bert.py/BertPreTrainedModel`
    """
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        torch.nn.init.trunc_normal_(module.weight.data, std=std, a=-2 * std, b=2 * std)
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, torch.nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def reinit_bert_like_(bert_like: transformers.modeling_utils.PreTrainedModel):
    """Initializes BERT weights and prunes weights if needed.

    Implementation follows: `transformers/modeling_utils.py/PreTrainedModel`
    """
    # Initialize weights
    bert_like.apply(
        functools.partial(
            _reinit_bert_like_module, std=bert_like.config.initializer_range
        )
    )

    # Prune heads if needed
    if bert_like.config.pruned_heads:
        bert_like.prune_heads(bert_like.config.pruned_heads)

    # Tie weights if needed
    bert_like.tie_weights()
