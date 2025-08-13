import torch
import sys
import os
from accelerate import Accelerator, DeepSpeedPlugin
from torch.utils.data import DataLoader,Dataset,DistributedSampler
from torch.cuda.amp import autocast
os.chdir('/content/TabPFN-reproduction')
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from deepspeed.utils import safe_get_full_grad
import gc
from data_pro.data_read import MyDataLoader
import pickle
from model.Trans_Encoder import Encoder
import torch.nn as nn
import json
import random
import numpy as np
from utils.dis_continue import get_dis
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,LambdaLR,SequentialLR

activate_map = {
    'GELU':nn.GELU()
}


writer = SummaryWriter('./log')
with open('./config/train_config.json') as f:
    args=json.loads(f.read())


torch.manual_seed(args['random_seed'])
torch.cuda.manual_seed_all(args['random_seed'])
torch.backends.cudnn.deterministic = True
random.seed(args['random_seed'])
np.random.seed(args['random_seed'])



deepspeed=DeepSpeedPlugin()

accelerator=Accelerator(deepspeed_plugin=deepspeed)
device = accelerator.device

pfn_model=Encoder(args,args['latent_dim'],args["VQ_num"],args["encoder_layer"],args["num_head"],args["norm_type"],activate_map[args['activate']]).to(device)
optimizer=torch.optim.AdamW(pfn_model.parameters(),lr=args['lr'])
dataset=MyDataLoader(args['file_num'],args)
crition=nn.CrossEntropyLoss(label_smoothing=args['loss_epsilon'])
sampler = DistributedSampler(dataset)
dataloader=DataLoader(dataset,batch_size=1,shuffle=False,sampler=sampler)

def lr_lambda(step):
    return min(1.0,(step+1)/args['warm_up_step'])
    
warmup_scheduler = LambdaLR(optimizer, lr_lambda)
cosine_scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=args['cos_time'], T_mult=2)

chained_scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[args['warm_up_step']]
)

#state_dict = torch.load("./checkpoint/pytorch_model/mp_rank_00_model_states.pt")
#pfn_model.load_state_dict(state_dict['module'],strict=False)


pfn_model,optimizer,dataloder,chained_scheduler=accelerator.prepare(pfn_model,optimizer,dataloader,chained_scheduler)


def fig_to_array(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(height, width, 4)  # RGBA
    img = img[:, :, :3]  # 转为 RGB
    return img

def train(pfn_model,dataloader,optimizer,crition):
    
    with tqdm(total=args['file_num']*args['epoch']) as _tqdm:
        now_step=0
        for now_epoch in range(args['epoch']):  
            for ans,attention_mask,continue_embedding,prediction_length in dataloader: 

                with autocast():
                    optimizer.zero_grad()
                    with accelerator.accumulate(pfn_model):
                        ans=ans.squeeze(0).to(device).detach()
                        attention_mask=attention_mask.squeeze(0).to(device).detach()
                        continue_embedding=continue_embedding.squeeze(0).to(device).detach()
                        output=pfn_model(attention_mask,continue_embedding,prediction_length)
                        prediction=output
                        loss=crition(prediction.transpose(2,3).transpose(1,2),ans)
                        
                        accelerator.backward(loss)
                        optimizer.step()
                        chained_scheduler.step()
                        
                        with torch.no_grad():
                            writer.add_scalar('train_loss', loss.item(), global_step=now_step, walltime=None)
                            writer.add_scalar('learning_rate',optimizer.param_groups[0]["lr"], global_step=now_step, walltime=None)

                    _tqdm.set_postfix(loss='{:.3f}'.format(loss.item()))
                    _tqdm.update(1)
                    with torch.no_grad():
                        if now_step%args['log_step']==0:
                            plt.figure(figsize=(50,10))
                            plt.cla()
                            plt.plot(range(5000),torch.softmax(prediction,dim=-1)[0,0,:].detach().reshape(-1).to('cpu'))
                            image = fig_to_array(plt.gcf())
                            image = np.transpose(image, (2, 0, 1))  
                            writer.add_image('distribution', image, global_step=now_step)
                            plt.close()

                            prediction=dataset.get_dis.get_continue_ans(prediction.max(dim=-1)[1])
                            ans=dataset.get_dis.get_continue_ans(ans)
                            plt.figure(figsize=(50,10))
                            plt.plot(range(continue_embedding.shape[1]-prediction.shape[1]),continue_embedding[0,:-prediction.shape[1],-2].reshape(-1).detach().to('cpu'))
                            plt.plot(range(continue_embedding.shape[1]-prediction.shape[1],continue_embedding.shape[1]),prediction.detach().reshape(-1).to('cpu'))
                            plt.plot(range(continue_embedding.shape[1]-prediction.shape[1],continue_embedding.shape[1]),ans.detach().reshape(-1).to('cpu'))
                            image = fig_to_array(plt.gcf())
                            image = np.transpose(image, (2, 0, 1))  
                            writer.add_image('prediction', image, global_step=now_step)
                            plt.close()

                        now_step=now_step+1
                        if now_step%args['save_step']==0:
                            accelerator.save_state(args['checkpoint_path'])
                


train(pfn_model,dataloader,optimizer,crition)
