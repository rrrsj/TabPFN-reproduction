import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.RMS_Norm import RMSNorm
import os
import math
from einops import rearrange
from layer.Embedding import Embedding
from layer.Attention_Layer import Attention_Layer
from layer.ffn import FFN 
from layer.Pre_Linear import Pre_Model
from layer.Trans_Layer import Trans_Layer
from torch.utils.checkpoint import checkpoint

class Encoder(nn.Module):
    def __init__(self,args,latent_dim,VQ_num,encoder_layer,num_head,norm_type,activate):
        super(Encoder,self).__init__()

        self.latent_dim=latent_dim
        self.VQ_num=VQ_num
        self.args=args
        self.encoder_layer=encoder_layer
        self.num_head=num_head
        self.norm_type=norm_type
        self.activate=activate

        self.Embedding=Embedding(
            args=self.args,
            latent_dim=self.latent_dim,
            VQ_num=self.VQ_num
        )

        self.backbone=nn.ModuleList()
        for i in range(self.encoder_layer):
            self.backbone.append(Trans_Layer(self.args,self.latent_dim,self.num_head,self.norm_type,self.activate))

        self.pred_model=Pre_Model(
            args=self.args,
            latent_dim=self.latent_dim,
            VQ_dim=self.VQ_num,
            activate=self.activate,
        )

    def forward(self,attention_mask,continue_embedding,prediction_length):
        embedding=self.Embedding(continue_embedding)

        for i in range(self.encoder_layer):
            embedding=self.backbone[i](embedding,attention_mask)

        embedding=embedding[:,-prediction_length:,-1:,:]
        ans=self.pred_model(embedding)
        return ans