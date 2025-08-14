import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,latent_dim):
        super(RMSNorm,self).__init__()
        self.latent_dim=latent_dim
        self.weight=nn.Parameter(torch.ones(self.latent_dim))

    def forward(self,x):
        var=(x**2).mean(dim=-1,keepdim=True)
        x=x*torch.rsqrt(var+1e-6)
        return self.weight*x