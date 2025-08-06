import torch
import torch.nn as nn
import os
import math
from layer.Embedding import Embedding
from layer.Attention_Layer import Attention_Layer
from layer.ffn import FFN 
from layer.Pre_Linear import Pre_Model
from torch.utils.checkpoint import checkpoint

class Trans_Layer(nn.Module):
    def __init__(self,args,latent_dim,num_head,norm_type,activate):
        super(Trans_Layer,self).__init__()

        self.latent_dim=latent_dim
        self.args=args
        self.num_head=num_head
        self.norm_type=norm_type
        self.activate=activate

        self.backbone=nn.ModuleList()
        
        self.backbone.append(Attention_Layer(
            args=self.args,
            latent_dim=self.latent_dim,
            num_head=self.num_head,
            head_dim=self.latent_dim//self.num_head,
            attention_type='feature',
            norm_type=self.args['norm_type']
        ))
        self.backbone.append(Attention_Layer(
            args=self.args,
            latent_dim=self.latent_dim,
            num_head=self.num_head,
            head_dim=self.latent_dim//self.num_head,
            attention_type='sample',
            norm_type=self.args['norm_type']
        ))
        self.backbone.append(FFN(
            args=self.args,
            latent_dim=self.latent_dim,
            activate=self.activate,
            norm_type=self.args['norm_type'],
        ))

    def forward(self,embedding,attention_mask):

        if self.args['use_grad_checkpoint']==1:
            output=checkpoint(self.get_output,embedding,attention_mask,use_reentrant=True)
        else:
            output=self.get_output(embedding,attention_mask)
        return output
    
    def get_output(self,embedding,attention_mask):
        embedding=self.backbone[0](embedding)
        embedding=self.backbone[1](embedding.transpose(1,2),attention_mask).transpose(1,2)
        embedding=self.backbone[2](embedding)

        return embedding