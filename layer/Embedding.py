import torch
import torch.nn as nn
import os
import math

class Embedding(nn.Module):
    def __init__(self,latent_dim,VQ_num,args):
        super(Embedding,self).__init__()
        self.latent_dim=latent_dim
        self.VQ_num=VQ_num
        self.args=args

        self.x_linear_embedding=nn.Linear(1,self.latent_dim)
        self.y_linear_embedding=nn.Linear(2,self.latent_dim)

        self.prediction_embedding=nn.Parameter(torch.randn(1,1,1,self.latent_dim))
        self.position_linear=nn.Linear(self.latent_dim//4,self.latent_dim)

    def forward(self,continue_embedding):
        batch_size,sample,feature=continue_embedding.shape

        x_embedding=self.x_linear_embedding(continue_embedding.unsqueeze(3)[:,:,:-2,:])
        y_embedding=self.y_linear_embedding(continue_embedding.unsqueeze(3)[:,:,-2:,:].transpose(2,3))
        embedding=torch.concat([x_embedding,y_embedding],dim=-2)
        return self.add_position(embedding)
    
    def add_position(self,x):
        random_state=torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_random_state=torch.cuda.get_rng_state(device=x.device)
        torch.manual_seed(self.args['random_seed'])
        embedding=self.position_linear(torch.randn((1,1,x.shape[2],x.shape[3]//4)).to(x.device)).expand(x.shape[0],x.shape[1],-1,-1)

        torch.set_rng_state(random_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_random_state,device=x.device)
        
        return x+embedding