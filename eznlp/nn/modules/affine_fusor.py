# -*- coding: utf-8 -*-
import torch


class MultiAffineFusor(torch.nn.Module):
    """Multiple affine fusor. 
    
    References
    ----------
    [1] Dozat and Manning. 2017. Deep biaffine attention for neural dependency parsing. ICLR 2017. 
    [2] Yu et al. 2020. Named entity recognition as dependency parsing. ACL 2020. 
    """
    def __init__(self, num_affines: int, in_dim: int, out_dim: int):
        super().__init__()
        self.num_affines = num_affines
        self.in_dim = in_dim 
        self.out_dim = out_dim
        
        self.U = torch.nn.Parameter(torch.empty(out_dim, *(in_dim+1 for _ in range(num_affines))))
        torch.nn.init.orthogonal_(self.U.data)
        torch.nn.init.zeros_(self.U.view(out_dim, -1)[:, -1])  # bias initialized as zeros
        self.register_buffer('_const', torch.ones(1, dtype=torch.float))
        
        
    def extra_repr(self):
        return f"num_affines={self.num_affines}, in_dim={self.in_dim}, out_dim={self.out_dim}"
        
        
    def forward(self, *args): 
        assert len(args) == self.num_affines
        
        fused = self.U.view(-1, 1)
        for x in args:
            _const = self._const.expand(*x.size()[:-1], 1)
            x = torch.cat([x, _const], dim=-1)
            fused = fused.view(*fused.size()[:-2], -1, self.in_dim+1)
            fused = fused.matmul(x.unsqueeze(-1))
        
        return fused.squeeze(-1)


class BiAffineFusor(MultiAffineFusor):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(2, in_dim, out_dim)
        
    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"


class TriAffineFusor(MultiAffineFusor):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(3, in_dim, out_dim)
        
    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"


class QuadAffineFusor(MultiAffineFusor):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(4, in_dim, out_dim)
        
    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"
