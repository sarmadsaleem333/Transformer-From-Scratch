
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import MultiHeadAttention as MultiHeadAttention

# transormer block
class TransformerBlock(nn.Module):

    def __init__(self,embed_dim,expansion_factor=4,n_heads=8):
        super(TransformerBlock,self).__init__()

        self.attention=MultiHeadAttention(embed_dim,n_heads)
        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)

        self.feed_forward=nn.Sequential(

            nn.Linear(embed_dim,embed_dim*expansion_factor),
            nn.ReLU(),
            nn.Linear(embed_dim*expansion_factor,embed_dim),
        )

        self.dropout1=nn.Dropout(0.1)
        self.dropout2=nn.Dropout(0.1)


    def forward(self,key,query,value,):
        attention_out=self.attention(key,query,value)  #32x10x512
        attention_residual_out=attention_out+value   #32x10x512
        norm1_out=self.dropout1(self.norm1(attention_residual_out))  #32x10x512


        feed_fwd_out=self.feed_forward  (norm1_out)  #32x10x512->32x10x2048->32x10x512
        feed_fwd_residual_out=feed_fwd_out+norm1_out #32x10x512
        norm2_out=self.drouput2(self.norm2(feed_fwd_residual_out))#32x10x512
        return norm2_out




