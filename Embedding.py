# create word embedding for each word in sentence

import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class Embedding(nn.Module):
# suppose  embedding size is  512 
# and vocab size is 100
# embedding vector size is 100x512

# batch size =32 , max words in sentence =10 
# so total size= 32x10x512

   def __init__(self, vocab_size,embed_dim):
        super(Embedding,self).__init__()
        self.embed=nn.Embedding(vocab_size,embed_dim)

    def forward(self,x):
        out =self.embed(x)
        return out





     
