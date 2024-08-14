import torch.nn as nn
import torch
import torch.nn.functional as F
import math

import MultiHeadAttention as MultiHeadAttention
import TransformerBlock as TransformerBlock


# decoder block for transformer

class DecoderBlock(nn.Module):
    def __init__(self,embed_dim,expansion_factor=4,n_heads=8):
        self.attention=MultiHeadAttention(embed_dim,n_heads=8)
        self.norm=nn.LayerNorm(embed_dim)
        self.dropout=nn.Dropout()
        self.transformer_block=TransformerBlock(embed_dim,expansion_factor,n_heads)

    

    def forward(self,key,query,x,mask):
        # pass mask for masking

        attention=self.attention(x,x,x,mask=mask)
        value=self.dropout(self.norm(attention+x))
        out=self.transformer_block(key,query,value)

        return out 




    