import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange




class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x





class MixerGatingUnit(nn.Module):
    def __init__(self,dim, seq_len, token_dim, channel_dim, dropout):
        super().__init__()     
        self.Mixer = MixerBlock(dim, seq_len, token_dim, channel_dim, dropout)
        self.proj = nn.Linear(dim,dim)

    def forward(self, x):
        u, v = x, x 
        u = self.proj(u)  
        v = self.Mixer(v)
        out = u * v
        return out


class NiNBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len,dropout):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.mgu = MixerGatingUnit(d_model,seq_len,d_ffn,d_ffn,dropout)
        self.ffn = FeedForward(d_model,d_ffn,dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mgu(x)   
        x = x + residual      
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        out = x + residual
        return out


class NiNformer(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, num_layers,dropout):
        super().__init__()
        
        self.model = nn.Sequential(
            *[NiNBlock(d_model, d_ffn, seq_len,dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)








