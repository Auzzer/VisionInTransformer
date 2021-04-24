import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
# Part 1: Define Basic Module will be used later,including:
## 1 Residual
## 2 PreNorm
## 3 FFN(FeedForward)
"""

class Residual(nn.Module):
    def __init__(self, layer: nn.Module):
        super(Residual, self).__init__()
        self.layer = layer

    def forward(self, x:torch.Tensor, **kwargs) -> torch.Tensor:
        return self.layer(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim:int, layer:nn.Module):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.layer = layer

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.layer(self.norm(x), **kwargs)

class FFN(nn.Module):
    # 在文章中一般定
    def __init__(self, dim:int, hidden_dim:int, dropout=float):
        super(FFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


"""
Part 2: Multi Head Attention
"""
class MHA(nn.Module):
    def __init__(self, dim:int, heads: int, dropout:float):
        super(MHA, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5 # scale to avoid the grad disappear
        self.to_qkv = nn.Linear(dim, dim*3, bias=False)
        from einops.layers.torch import Rearrange
        self.concat = Rearrange("b h n d -> b n (h d)") # 点乘操作计算相似度
        self.linear = nn.Linear(dim, dim)
        self.attend = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)

    from typing import Tuple
    def get_qkv(self, x:torch.Tensor) -> Tuple[torch.Tensor]:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        return map(lambda t: rearrange(t, "b n (h d) -> b h n d", h = self.heads), qkv)

    def Scaled_Dot_Product(self, a, b):
        return torch.einsum("bhid, bhjd -> bhid", a, b) * self.scale

    def mask_Product(x, mask):
        mask = F.pad(mask.flatten(1), (1,0), value=True)
        mask = mask[:,None,:] * mask[:, :, None]
        x.masked_fill_(~mask, float("-inf"))

    def matmul(self, a, b):
        return torch.einsum('bhij, bhjd -> bhid', a, b)

    def forward(self, x, mask = None):
        q, k, v = self.get_qkv(x)
        dots = self.Scaled_Dot_Product(q, k)
        if mask is not None:
            dots = self.mask_Product(dots, mask)
        attention = self.attend(dots)
        out = self.matmul(attention, v)
        out = self.concat(out)
        out = self.linear(out)
        out = self.dropout(out)
        return out




"""
Part3: Define Transformer
"""


class Transformer(nn.Module):
    """ Implementation of the Transformer Encoder as decribed in the paper
    'An Image is Worth 16x16 words: Transformers for image recognition at scale',
     by Dosovitskiy et al, 2020.
    """

    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout_rate: float):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Residual(PreNorm(dim, MHA(dim, heads, dropout_rate))),
                    Residual(PreNorm(dim, FFN(dim, mlp_dim, dropout_rate)))
                ])
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for Attention, FeedForward in self.layers:
            x = FeedForward(Attention(x, mask=mask))
        return x









