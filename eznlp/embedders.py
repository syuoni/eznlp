# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .datasets_utils import Batch
from .nn_utils import reinit_embedding_, reinit_layer_



class CharCNN(nn.Module):
    def __init__(self, voc_dim: int, emb_dim: int, out_dim: int, kernel_size: int, 
                 dropout: float, pad_idx: int):
        super().__init__()
        self.emb = nn.Embedding(voc_dim, emb_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(emb_dim, out_dim*2, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.glu = nn.GLU()
        self.dropout = nn.Dropout(dropout)
        
        reinit_embedding_(self.emb)
        reinit_layer_(self.conv, 'sigmoid')
        
        
    def forward(self, char_ids: Tensor, tok_lens: Tensor, char_mask: Tensor, seq_lens: Tensor):
        # embedded: (batch*sentence, step, emb_dim)
        embedded = self.emb(char_ids)
        
        # feats: (batch*sentence, step, out_dim)
        # NOTE: The input of this convolutional layer has been ensured to be zeros
        # in padding positions.
        feats = self.conv(self.dropout(embedded).permute(0, 2, 1)).permute(0, 2, 1)
        feats = self.glu(feats)
        
        # feats: (batch*sentence, out_dim)
        feats.masked_fill_(char_mask.unsqueeze(-1), float('-inf'))
        feats = feats.max(dim=1).values
        
        # Retrieve token-level steps
        # feats: (batch, step, out_dim)
        offsets = [0] + seq_lens.cumsum(dim=0).cpu().tolist()
        feats = pad_sequence([feats[s:e] for s, e in zip(offsets[:-1], offsets[1:])], 
                             batch_first=True, padding_value=0.0)
        return feats
    
    
class EnumEmbedding(nn.Module):
    def __init__(self, voc_dim: int, emb_dim: int, pad_idx: int):
        super().__init__()
        self.emb = nn.Embedding(voc_dim, emb_dim, padding_idx=pad_idx)
        reinit_embedding_(self.emb)
        
    def forward(self, enum_ids: Tensor):
        return self.emb(enum_ids)
    
    
class ValEmbedding(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        # NOTE: Two reasons why this layer does not have activation. 
        # (1) Activation function should been applied after batch-norm / layer-norm. 
        # (2) This layer is semantically an embedding layer, which typically does NOT require activation. 
        self.proj = nn.Linear(in_dim, emb_dim)
        reinit_layer_(self.proj, 'linear')
        
    def forward(self, val_ins: Tensor):
        return self.proj(val_ins)
    
    
    
class Embedder(nn.Module):
    def __init__(self, config: dict, itos=None, pretrained_vectors=None):
        """
        `Embedder` forwards from inputs to embeddings. 
        """
        super().__init__()
        self.config = config
        
        if 'tok' in config:
            self.word_emb = nn.Embedding(config['tok']['voc_dim'], config['tok']['emb_dim'], padding_idx=config['tok']['pad_idx'])
            reinit_embedding_(self.word_emb, itos=itos, pretrained_vectors=pretrained_vectors)
            if config['tok']['use_pos_emb']:
                self.pos_emb = nn.Embedding(config['tok']['max_len'], config['tok']['emb_dim'])
                reinit_embedding_(self.pos_emb)
            
        if 'char' in config:
            self.char_cnn = CharCNN(**config['char'])
            
        if 'enum' in config:
            self.enum_embs = nn.ModuleDict({f: EnumEmbedding(**enum_config) for f, enum_config in config['enum'].items()})
                
        if 'val' in config:
            self.val_embs = nn.ModuleDict({f: ValEmbedding(**val_config) for f, val_config in config['val'].items()})
            
        
    def get_word_embedded(self, batch: Batch):
        if hasattr(self, 'word_emb'):
            # word_embedded: (batch, step, emb_dim)
            word_embedded = self.word_emb(batch.tok_ids)
            
            if hasattr(self, 'pos_emb'):
                # pos_embedded: (batch, step, emb_dim)
                pos = torch.arange(word_embedded.size(1), device=word_embedded.device).repeat(word_embedded.size(0), 1)
                pos_embedded = self.pos_emb(pos)
                return (word_embedded + pos_embedded) * (0.5**0.5)
            else:
                return word_embedded
        else:
            return None
        
        
    def get_char_cnn_embedded(self, batch: Batch):
        if hasattr(self, 'char_cnn'):
            return self.char_cnn(batch.char_ids, batch.tok_lens, batch.char_mask, batch.seq_lens)
        else:
            return None
        
        
    def get_enum_embedded(self, batch: Batch):
        if hasattr(self, 'enum_embs'):
            return torch.cat([self.enum_embs[f](batch.enum[f]) for f in self.enum_embs], dim=-1)
        else:
            return None
        
        
    def get_val_embedded(self, batch: Batch):
        if hasattr(self, 'val_embs'):
            return torch.cat([self.val_embs[f](batch.val[f]) for f in self.val_embs], dim=-1)
        else:
            return None

    
    def forward(self, batch: Batch, word=True, char_cnn=True, enum=True, val=True):
        embedded = []
        
        if word and hasattr(self, 'word_emb'):
            embedded.append(self.get_word_embedded(batch))
        if char_cnn and hasattr(self, 'char_cnn'):
            embedded.append(self.get_char_cnn_embedded(batch))
        if enum and hasattr(self, 'enum_embs'):
            embedded.append(self.get_enum_embedded(batch))
        if val and hasattr(self, 'val_embs'):
            embedded.append(self.get_val_embedded(batch))
            
        embedded = torch.cat(embedded, dim=-1)
        return embedded
    