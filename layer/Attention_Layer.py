import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.RMS_Norm import RMSNorm
import os
import math
from einops import rearrange

class Attention_Layer(nn.Module):
    def __init__(self,args,latent_dim,num_head,head_dim,attention_type,norm_type='L'):
        super(Attention_Layer,self).__init__()

        self.latent_dim=latent_dim
        self.args=args
        self.attention_type=attention_type
        self.head_dim=head_dim
        self.num_heads=num_head
        self.norm_type=norm_type
        self.group_num=self.args['group_num']

        if self.norm_type=='L':
            self.norm=nn.LayerNorm(self.latent_dim)
        if self.norm_type=='R':
            self.norm=RMSNorm(self.latent_dim)

        self.w_q=nn.Linear(self.latent_dim,self.latent_dim)
        self.w_k=nn.Linear(self.latent_dim,self.latent_dim//self.group_num,bias=False)
        self.w_v=nn.Linear(self.latent_dim,self.latent_dim//self.group_num,bias=False)
        self.w_out=nn.Linear(self.latent_dim,self.latent_dim,bias=False)

        std=math.sqrt(2./float(num_head*head_dim+latent_dim))
        a=math.sqrt(3)*std
        torch.nn.init.uniform_(self.w_q.weight,-a,a)
        torch.nn.init.uniform_(self.w_k.weight,-a,a)
        torch.nn.init.uniform_(self.w_v.weight,-a,a)
        torch.nn.init.zeros_(self.w_out.weight)

    def forward(self,x,mask=None):
        batch,example,feature_num,latent_dim=x.shape
        mid_x=x
        Q=self.w_q(mid_x)
        K=self.w_k(mid_x)
        V=self.w_v(mid_x)

        Q=rearrange(Q,'b l f (g h d)->b l g h f d',g=self.group_num,h=self.num_heads//self.group_num,d=self.head_dim)
        K=rearrange(K,'b l f (g h d)->b l g h f d',g=1,h=self.num_heads//self.group_num,d=self.head_dim)
        V=rearrange(V,'b l f (g h d)->b l g h f d',g=1,h=self.num_heads//self.group_num,d=self.head_dim)
        if self.attention_type=='sample':
            output=F.scaled_dot_product_attention(Q,K,V,attn_mask=mask)
        else:
            output=F.scaled_dot_product_attention(Q,K,V)
        
        output=rearrange(output,'b l g h f d->b l f (g h d)')
        output=self.w_out(output)
        return self.norm(output+x)
    