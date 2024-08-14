import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim=512,n_heads=8):

        super(MultiHeadAttention, self)._init_()

        # 512 will be emdeddding dimension
        self.embed_dim=embed_dim  

        # number of heads will be 8
        self.n_heads=n_heads

        # query,key value pairs will be of 64 *64
        # 512/8=64
        self.single_head_dim=int (self.embed_dim/self.n_heads)  



        self.query_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.key_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.value_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        
        self.out=nn.Linear(self.single_head_dim*self.n_heads,self.embed_dim)


    def forward(self,key,query,value,mask=None):    
        
        #batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        
        batch_size=key.size(0)
        seq_len=key.size(1)

        # query length can change in decoder so we cant take general length
        seq_length_query=query.size(1)

        #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        key=key.view (batch_size,seq_len,self.n_heads,self.single_head_dim)
        query=query.view (batch_size,seq_length_query,self.n_heads,self.single_head_dim)
        value=value.view (batch_size,seq_len,self.n_heads,self.single_head_dim)


        # 32x10x8x64 
        key=self.key_matrix(key)
        query=self.query_matrix(query)
        value=self.value_matrix(value)


        # transpose the matrix to get the desired shape
        #batch_size x n_heads  x sequence_length x single_head_dim = (32x8x10x64)
        key=key.transpose(1,2)
        query=query.transpose(1,2)
        value=value.transpose(1,2)

        # for computing the attention
        # adjusting key matrix

        k_adjusted=k.transpose(-1,-2)  # 32x8x64x10  (batch_size x n_heads x single_head_dim x sequence_length)

        # computing the attention
        # 32x8x10x64  * 32x8x64x10 = 32x8x10x10
        product=torch.matmul(q,k_adjusted)  # 32x8x10x10

        # fill the mask positions

        if mask is None:
            product=product.masked_fill(mask==0,float('-1e20'))
        

        # divide the product by the sqrt of single head dimension  /64
        product =product/(math.sqrt(self.single_head_dim))

        # apply softmax
        scores=F.softmax(product,dim=-1)

        # mulityply with value matrix

        scores=torch.matmul(scores,v)  # 32x8x10x10 * 32x8x10x64= 32x8x10x64

        # concatenated output
        # 32x8x10x64  -> 32x10x8x64-> 32x10x512
        concat=scores.transpose(1,2).contiguous().view(batch_size,seq_len,self.single_head_dim*self.n_heads)

        output=self.out (concat)
        return output



