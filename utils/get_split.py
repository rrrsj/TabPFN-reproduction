import torch
import sys
from torch.utils.data import DataLoader,Dataset,DistributedSampler
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import gc
from data_pro.data_read import MyDataLoader
import pickle
#from model.Trans_Encoder import Encoder
import torch.nn as nn
import json
import random
import numpy as np
import os
from utils.dis_continue import get_dis
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
import copy



with open('./config/train_config.json') as f:
    args=json.loads(f.read())


torch.manual_seed(args['random_seed'])
torch.cuda.manual_seed_all(args['random_seed'])
torch.backends.cudnn.deterministic = True
random.seed(args['random_seed'])
np.random.seed(args['random_seed'])

def train():
    now_step=0
    all_number=[]
    with open(args['data_path']+str(now_step+1)+'.pkl','rb') as f:
        value=pickle.load(f)
    
    for l in trange(len(value)):
        for i in range(value[l].shape[1]):
            for j in range(value[l].shape[2]):
                for k in range(value[l].shape[3]):
                    all_number.append(value[l][0,i,j,k].item())
            if l==1000:
                break
    
    return all_number

all_number=train()
all_number.sort()

buck_number=[]

for i in range(args['VQ_num']-1):
    buck_number.append(all_number[int(len(all_number)/args['VQ_num']*(i+1))])

print(buck_number)

with open('./utils/split.pkl','wb') as f:
    pickle.dump(buck_number,f)

