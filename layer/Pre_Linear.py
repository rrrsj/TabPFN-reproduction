import torch
import torch.nn as nn
import os
import math
from layer.RMS_Norm import RMSNorm

class Pre_Model(nn.Module):
    def __init__(self,args,latent_dim,VQ_dim,activate):
        super(Pre_Model,self).__init__()

        self.args=args
        self.latent_dim=latent_dim
        self.VQ_dim=VQ_dim
        self.activate=activate

        self.pre_model=nn.Sequential(nn.Linear(self.latent_dim,4*self.latent_dim),
        self.activate,
        nn.Linear(4*self.latent_dim,VQ_dim))

        torch.nn.init.zeros_(self.layer2.weight)


    def forward(self,x):
        return self.pre_model(x)

