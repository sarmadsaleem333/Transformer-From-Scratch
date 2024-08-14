import torch.nn as nn
import torch
import torch.nn.functional as F
import math

import Embedding as Embedding
import PositionalEncoding as PositionalEncoding
import TransformerBlock as TransformerBlock


class TransformerDecoder(nn.Module):


    def__init__(self,target,vocab_size,embed_dim,seq_len,num_layers=2,expansion_factor=4,n_heads=8):
        super(TransformerDecoder.self).__init__()

        self.word_embedding=Embedding(vocab_size,embed_dim)
        self.position_embedding=PositionalEncoding(seq_len,embed_dim)
        self.layers=nn.ModuleList([
            DecoderBlock(embed_dim,expansion_factor=4,n_heads=8)
            for _ in range(num_layers)
        ])
        self.fc_out=nn.Linear(embed_dim,vocab_size)
        self.dropout=nn.Dropout(0.2)

    
    def forward(self,x,enc_out,mask):


        x=self.word_embedding(x) #32x10x512
        x=self.position_embedding(x) #32x10x512
        x=self.dropout(x)

        for layer in self.layers:
            x=layer(enc_out,x,enc_out,mask)
        out =F.softmax(self.fc_out(x))
        
        return out