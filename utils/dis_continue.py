import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import pickle 
import copy
import os 

class get_dis:
    def __init__(self,args):
        self.args=args

        self.ans_bucket_number=[]

        with open('./utils/split.pkl','rb') as f:
            self.ans_bucket_number=pickle.load(f)
        
        self.ans_bucket_number_temp=[self.args['min_value']]+self.ans_bucket_number+[self.args['max_value']]
        self.bucket_width=(torch.tensor(self.ans_bucket_number[1:])-torch.tensor(self.ans_bucket_number[:-1])).reshape(1,1,1,-1).to('cuda')
        self.VQ_num=self.args['VQ_num']

    def get_ans(self,value):
        index=torch.bucketize(value,torch.tensor(self.ans_bucket_number))
        return index.detach()
    
    def get_continue_ans(self,value):
        value=torch.nn.functional.one_hot(value,num_classes=self.args['VQ_num']).float()
        temp=[self.args['min_value']]+self.ans_bucket_number+[self.args['max_value']]
        temp_ans=[]
        for i in range(len(temp)-1):
            temp_ans.append((temp[i]+temp[i+1])/2)
            ans=torch.einsum('...cd,...de->...ce',value,torch.tensor(temp_ans).unsqueeze(0).to(value.device))
        return ans.squeeze(3)