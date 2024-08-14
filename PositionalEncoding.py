# class for making positional encoding for transformer model
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import Embedding as Embedding

class PositionalEncoding(nn.Module):
    
    def __init__(self,max_seq_len,embed_model_dim):

        super (PositionalEncoding,self).__init__()
        self.embed_dim=embed_model_dim
        pe=torch.zeros(max_seq_len,self.embed_dim)


        # formula taken from attention is all you need paper
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))

        pe = pe.unsqueeze(0)  # adding batch dimension [1,max_seq_len, embedding_dim]  
                              #1 would be replace by batch size

         self.register_buffer('pe', pe)

    
    def forward(self,x):
        
        # making embedding relatively larger
        x=x*math.sqrt(self.embed_dim)

        # add constant to embedding
        seq_len=x.size(1)
        x=x+torch.autograd.Variable(self.pe[:,:seq_len],require_grad=False)
        return x


