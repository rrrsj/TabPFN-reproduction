import torch
import torch.nn as nn
import copy
import random 
import numpy as np
from torch.utils.data import DataLoader,Dataset
import pickle
import networkx as nx
from sklearn.preprocessing import KBinsDiscretizer
import os 
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import sys
sys.path.append('.')
sys.path.append('..')
from utils.dis_continue import get_dis

class MyDataLoader(Dataset):
    def __init__(self,train_step,args):
        super(MyDataLoader,self).__init__()
        self.train_step=train_step
        self.args=args
        self.get_dis=get_dis(self.args)
        self.now_file=0
        self.file_input_number=self.args['file_num']
        self.value=0
    def __getitem__(self,index):
        if self.file_input_number==self.args['file_num']:
            self.now_file=self.now_file+1
            with open(self.args['data_path']+str(self.now_file)+'.pkl','rb') as f:
                self.value=pickle.load(f)
                self.file_input_number=0
        index=index%self.args['file_num']
        value=self.value[index][0,:,:,:]
        self.file_input_number=self.file_input_number+1
        prediction_length=self.args['prediction_length']
        prediction_num=1
        dis_value_ans=self.get_dis.get_ans(value)
        ans=copy.deepcopy(dis_value_ans[:,-prediction_length:,-prediction_num:]).detach()
        attention_mask_section1=torch.ones((self.args['batch'],value.shape[1],value.shape[1]-prediction_length))
        attention_mask_section2=torch.zeros((self.args['batch'],value.shape[1]-prediction_length,prediction_length))
        attention_mask_section3=(torch.eye(prediction_length).reshape(1,prediction_length,prediction_length)).expand(self.args['batch'],-1,-1)
        attention_mask=torch.concat([attention_mask_section2,attention_mask_section3],dim=-2)
        attention_mask=torch.concat([attention_mask_section1,attention_mask],dim=-1).detach().unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(value.shape[0],value.shape[2],self.args['group_num'],self.args['num_head']//self.args['group_num'],-1,-1)
        continue_embedding=copy.deepcopy(value)
        continue_embedding[:,-prediction_length:,-prediction_num:]=0

        mask_y=torch.zeros((value.shape[0],value.shape[1],1))
        mask_y[:,-prediction_length:,:]=1
        continue_embedding=torch.concat([continue_embedding,mask_y],dim=-1)
        return ans,attention_mask.bool(),continue_embedding.float(),prediction_length
    
    def __len__(self):

        return self.args['file_num']
