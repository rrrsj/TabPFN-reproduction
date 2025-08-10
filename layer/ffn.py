import torch
import torch.nn as nn
import os
import math
from layer.RMS_Norm import RMSNorm

class FFN(nn.Module):
    def __init__(self,args,latent_dim,activate,norm_type):
        super(FFN,self).__init__()

        self.args=args
        self.latent_dim=latent_dim
        self.activate=activate
        self.norm_type=norm_type

        self.layer1=nn.Linear(self.latent_dim,4*self.latent_dim,bias=False)
        self.layer2=nn.Linear(4*self.latent_dim,self.latent_dim,bias=False)

        torch.nn.init.zeros_(self.layer2.weight)

        if self.norm_type=='L':
            self.norm=nn.LayerNorm(self.latent_dim)
        if self.norm_type=='R':
            self.norm=RMSNorm(self.latent_dim)

    def forward(self,x):
        mid_x=x
        ans=self.layer2(self.activate(self.layer1(mid_x)))
        return norm(ans+x)

