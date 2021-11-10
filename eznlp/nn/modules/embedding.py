# -*- coding: utf-8 -*-
import torch


class SinusoidPositionalEncoding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, base: int=10000, scale: float=None):
        super().__init__()
        assert embedding_dim % 2 == 0
        self.scale = (embedding_dim**-0.5) if scale is None else scale
        assert abs(self.scale) <= 1
        
        positions = torch.arange(num_embeddings)
        i = torch.arange(embedding_dim//2)
        freqs = positions.unsqueeze(-1) / (base ** (2*i / embedding_dim))
        weight = torch.stack([freqs.sin(), freqs.cos()]).permute(1, 2, 0).contiguous().view(num_embeddings, embedding_dim)
        self.register_buffer('weight', weight*self.scale)
        
        
    def extra_repr(self):
        return f"{self.weight.size(0)}, {self.weight.size(1)}, scale={self.scale:.4f}"
        
    def forward(self, x: torch.Tensor):
        return self.weight[x]
