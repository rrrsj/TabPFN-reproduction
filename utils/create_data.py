import torch
import sys
import os
from torch.utils.data import DataLoader,Dataset,DistributedSampler
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import gc
from data_pro.data_provide import MyDataLoader
import pickle
import torch.nn as nn
import json
import random
import numpy as np

from utils.dis_continue import get_dis
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.preprocessing._discretization")


with open('./config/train_config.json') as f:
    args=json.loads(f.read())


torch.manual_seed(args['random_seed'])
torch.cuda.manual_seed_all(args['random_seed'])
torch.backends.cudnn.deterministic = True
random.seed(args['random_seed'])
np.random.seed(args['random_seed'])

dataset=MyDataLoader(args['train_step']*args['epoch'],args)

dataloader=DataLoader(dataset,batch_size=1,num_workers=0,shuffle=False)

def train(dataloader):
    now_step=0
    file_content=[]

    with tqdm(total=args['train_step']*args['epoch']) as _tqdm:
        for value in dataloader:
            _tqdm.update(1)
            file_content.append(value)

            if len(file_content)==args['file_num']:
                now_step=now_step+1
                with open(args['data_path']+str(now_step)+'.pkl','wb') as f:
                    pickle.dump(file_content,f,protocol=pickle.HIGHEST_PROTOCOL)
                file_content=[]
train(dataloader)


