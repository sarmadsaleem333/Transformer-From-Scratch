import torch.nn as nn
import torch
import torch.nn.functional as F
import math

import TransformerDecoder as TransformerDecoder
import TransformerEncoder as TransformerEncoder


# Transformer Class:

class Transformer(nn.Module):
    def __init__(self,embed_dim,src_voab_size,target_vocab_size,seq_length,num_layers=2,expansion_factor=4,n_heads=8):
        super(Transformer,self).__init__()

        """  
        Args:
           embed_dim:  dimension of embedding 
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """

        self.target_vocab_size=target_vocab_size
        self.encoder=TransformerEncoder(seq_length,src_voab_size,embed_dim,num_layers=num_layers,expansion_factor=expansion_factor,n_heads=n_heads)
        self.decoder=TransformerDecoder(target_vocab_size,embed_dim,seq_length,num_layers=num_layers,expansion_factor=expansion_factor,n_heads=n_heads)


    def make_trg_mask(self,trg):
        # args: target sequence
        # returns mask  
        batch_size,trg_len=trg.shape
        trg_mask=torch.trill(torch.ones((trg_len,trg_len))).expand(batch_size,1,trg_len,trg_len )

        return trg_mask

    def decode(self,src,target):


        # for inference src:input to encoder ,target: input to decoder
        # out:return final prediction of sequence

        trg_mask=self.make_trg_mask(trg)
        enc_out=self.encoder(src)
        out_labels= []

        batch_size,seq_len=src.shape[0],src.shape[1]

        # output dimension=torch.zeros(seq_len,batch_size,self.target_vocab_size)
        out=trg

        for i in range(seq_len): # seq len is 10 in our case
            out =self.decoder(out,enc_out,trg_mask)
            out =out [:,-1,:]
            out =out.argmax(-1)
            out_labels.append(out.item())
            out =torch.unsqueeze(out,axis=0)

        return out_labels
    
    def forward(self,src,trg):
        # src :input to encoder
        # trg: input to decoder
        # out :final vector which return prob of each word

        trg_mask=self.make_trg_mask(trg)
        enc_out=self.encoder(src)
        outputs=self.decoder(trg,enc_out,trg_mask)

        return outputs



