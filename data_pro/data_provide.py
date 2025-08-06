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

class activate_GeLU(nn.Module):
    def __init__(self):
        super(activate_GeLU,self).__init__()
        self.activate_layer=nn.GELU()
    def forward(self,inputs):
        return self.activate_layer(inputs)
    
class activate_Sin(nn.Module):
    def __init__(self):
        super(activate_Sin,self).__init__()
    def forward(self,inputs):
        return torch.sin(inputs)
    
class activate_ReLU(nn.Module):
    def __init__(self):
        super(activate_ReLU,self).__init__()
        self.activate_layer=nn.ReLU()
    def forward(self,inputs):
        return self.activate_layer(inputs)

class activate_Tanh(nn.Module):
    def __init__(self):
        super(activate_Tanh,self).__init__()
        self.activate_layer=nn.Tanh()
    def forward(self,inputs):
        return self.activate_layer(inputs)

class activate_Sigmoid(nn.Module):
    def __init__(self):
        super(activate_Sigmoid,self).__init__()
        self.activate_layer=nn.Sigmoid()
    def forward(self,inputs):
        return self.activate_layer(inputs)
    
class activate_LeakyRelu(nn.Module):
    def __init__(self):
        super(activate_LeakyRelu,self).__init__()
        self.activate_layer=nn.LeakyReLU()
    def forward(self,inputs):
        return self.activate_layer(inputs)
    
class activate_silu(nn.Module):
    def __init__(self):
        super(activate_silu,self).__init__()
        self.activate_layer=nn.SiLU()
    def forward(self,inputs):
        return self.activate_layer(inputs)
    
class Data_Provide:
    def __init__(self,args):
        self.args=args
        self.batch=self.args['batch']
        self.model_layer=None
        self.input_feature=None
        self.model=None
        self.node_functions = dict()
    
    def _init_model(self):
        self.model_type=random.choice(['mlp','scm'])
        if self.model_type=='mlp':
            with torch.no_grad():
                self.model=self.get_mlp_model(self.args['min_feature'],self.args['max_feature'],self.args['min_layer'],self.args['max_layer'])
                self._init_mlp_model_parameter()
        elif self.model_type=='scm':
            all_node=random.randint(self.args['min_feature']*self.args['min_layer'],self.args['max_feature']*self.args['max_layer'])
            connect=random.randint(3,self.args['min_feature']-1)
            self.model=nx.barabasi_albert_graph(all_node,connect)
            self.model=nx.DiGraph([(u,v) for u,v in self.model.edges() if u<v])
            self.get_scm_model()

    def get_data(self):
        if self.model_type=='mlp':
            all_data=self.get_mlp_data()
        elif self.model_type=='scm':
            all_data=self.get_scm_data()
        
        std=all_data.std(dim=(0,1))
        mask=std!=0
        all_data=all_data[...,mask]

        all_sample_node=np.random.beta(0.95,5.0)
        all_sample_node=min(int(round(159*all_sample_node+1)),all_data.shape[2])
        all_sample_node=min(all_sample_node,self.args['all_cells']//all_data.shape[1])
        sampled=np.random.choice([i for i in range(all_data.shape[2])],size=all_sample_node,replace=False)
        output=torch.index_select(all_data,-1,torch.tensor(sampled).to(all_data.device))
        return output
    
    def get_mlp_model(self,min_feature,max_feature,min_layer,max_layer):
        model=nn.ModuleList()
        activate_list=[activate_LeakyRelu(),activate_GeLU(),activate_ReLU(),activate_Sigmoid(),activate_silu(),activate_Sin(),activate_silu()]
        activate_number=random.randint(0,len(activate_list)-1)
        with torch.no_grad():
            self.model_layer=random.randint(min_layer,max_layer)
            self.input_feature=random.randint(min_feature,max_feature)
            self.mid_feature=random.randint(min_feature,max_feature)
            self.output_feature=random.randint(min_feature,max_feature)
            for i in range(self.model_layer):
                if i==0:
                    model.append(nn.Linear(self.input_feature,self.mid_feature))
                elif i==self.model_layer-1:
                    model.append(nn.Linear(self.mid_feature,self.mid_feature))
                else:
                    model.append(nn.Linear(self.mid_feature,self.mid_feature))
                if not i==self.model_layer-1:
                    model.append(activate_list[activate_number])
        return model
    
    def get_scm_model(self):
        for node in self.model.nodes:
            if list(self.model.predecessors(node)):
                func_type=random.choice(['linear','nonlinear','discretize'])
            else:
                func_type='noise'
            
            if func_type=='noise':
                def func(batch_size,sample_size):
                    return torch.randn((batch_size,sample_size,1)).float()
            elif func_type=='linear':
                linear_type=random.choice(['linear','max','min'])
                if linear_type=='linear':
                    def func(input_data):
                        weight=torch.randn(input_data.shape[2],1).float()
                        bias=torch.randn(1).float()
                        return (input_data.float()@weight+bias)
                elif linear_type=='max':
                    def func(input_data):
                        signal=random.choice([-1.,1.])
                        return signal*(torch.max(input_data,dim=-1)[0].unsqueeze(2)/10)
                elif linear_type=='min':
                    def func(input_data):
                        signal=random.choice([-1.,1.])
                        return signal*(torch.min(input_data,dim=-1)[0].unsqueeze(2)/10)
            elif func_type == 'nonlinear':
                func_type= random.choice(['sin','gelu','selfmul','modulo','inputmul'])
                if func_type=='sin':
                    def func(input_data):
                        return random.uniform(-2,2.)*torch.sin(input_data).mean(dim=-1).unsqueeze(2)
                
                elif func_type=='gelu':
                    def func(input_data):
                        signal = random.choice([-1, 1])
                        return signal*(F.gelu(input_data).mean(dim=-1).unsqueeze(2))
                
                elif func_type=='selfmul':
                    def func(input_data):
                        signal = random.choice([-1, 1])
                        return signal*(((input_data)**2).mean(dim=-1).unsqueeze(2)/2)
                        
                elif func_type == 'modulo':
                    def func(input_data):
                        divisor = random.uniform(0.5, 2.0)
                        signal = random.choice([-1, 1])
                        return signal*torch.fmod(input_data, divisor).mean(dim=-1, keepdim=True)

                elif func_type== 'inputmul':
                    def func(input_data):
                        num=random.randint(1,len(input_data))
                        result = random.sample([i for i in range(input_data.shape[-1])], num)
                        ans=torch.ones((input_data.shape[0],input_data.shape[1],1))
                        for i in result:
                            ans=ans*input_data[:,:,i:i+1]
                        return ans
                        

            elif func_type == 'discretize':
                def func(input_data):
                    class_temp=random.randint(2,10)
                    class_type=random.choice(['uniform','quantile','kmeans'])
                    est = KBinsDiscretizer(n_bins=class_temp, encode='ordinal', strategy=class_type)
                    signal = random.choice([-1, 1])
                    return signal*(torch.tensor(est.fit_transform(input_data.mean(dim=-1).reshape(-1,1))).reshape(input_data.shape[0],input_data.shape[1],1))
            self.node_functions[node] = func
        
    def get_mlp_data(self):
        all_data=[]
        batch=self.args['batch']
        input_feature=self.input_feature
        sample=random.randint(self.args['min_sample'],self.args['max_sample'])
        input_data=[]
        for i in range(batch):
            input_data.append(self.get_input_value((1,sample,input_feature)))
        input_data=torch.concat(input_data,dim=0).to('cuda')

        mean=input_data.mean(dim=-2).unsqueeze(1).expand(-1,input_data.shape[1],-1)
        std=input_data.std(dim=-2).unsqueeze(1).expand(-1,input_data.shape[1],-1)+1e-6
        input_data=(input_data-mean)/std

        all_data.append(input_data)
        self.model=self.model.to('cuda')

        for i in range(self.model_layer):
            if i!=(self.model_layer-1):
                input_data=self.model[i*2+0](input_data)
                noise=torch.randn((input_data.shape[0],input_data.shape[1],input_data.shape[2]))*0.01
                input_data=input_data+noise.to(input_data.device)

                mean=input_data.mean(dim=-2).unsqueeze(1).expand(-1,input_data.shape[1],-1)
                std=input_data.std(dim=-2).unsqueeze(1).expand(-1,input_data.shape[1],-1)+1e-6
                input_data=(input_data-mean)/std

                all_data.append(input_data)
                input_data=self.model[i*2+1](input_data)
            else:
                input_data=self.model[i*2+0](input_data)
                all_data.append(input_data)
        
        all_data=torch.concat(all_data,dim=-1)
        return all_data

    def get_scm_data(self):
        batch=self.args['batch']
        sample=random.randint(self.args['min_sample'],self.args['max_sample'])

        values = {}
        topo_order = list(nx.topological_sort(self.model))
        for node in topo_order:
            preds = list(self.model.predecessors(node))#找到所有前驱节点
            if not preds:
                values[node] = self.node_functions[node](batch,sample)
            else:
                input_vals = [values[p] for p in preds]
                input_vals=torch.concat(input_vals,dim=-1)
                values[node] = self.node_functions[node](input_vals)
                values[node]=values[node]+(torch.randn((values[node].shape[0],values[node].shape[1],1)).float())*0.01
                mean=values[node].mean(dim=-2).unsqueeze(1).expand(-1,values[node].shape[1],-1)
                std=values[node].std(dim=-2).unsqueeze(1).expand(-1,values[node].shape[1],-1)+1e-6
                values[node]=(values[node]-mean)/std

        all_data=[]
        for key in values.keys():
            all_data.append(values[key])  
        all_data=torch.concat(all_data,dim=-1)
        return all_data


    

    def _init_mlp_model_parameter(self):
        for i in range(len(self.model)):
            if type(self.model[i])==type(nn.Linear(1,1)):
                weight,bias=self.get_model_value(self.model[i].weight.shape)
                if random.uniform(0,1)<0.5:
                    p=random.uniform(0.1,0.3)
                else:
                    p=0
                weight=nn.functional.dropout(weight,p=p)
                self.model[i].weight=nn.Parameter(weight)
                if p>=0:
                    self.model[i].weight *= 1 / (1. - p)**0.5
                self.model[i].bias=nn.Parameter(bias)
            

            
    def get_model_value(self,input_shape):
        
        assert len(input_shape)==2 
        random_temp=random.uniform(0,1)
        sample,feature=input_shape
        
        
        bias=torch.randn((sample))

        if random_temp<0.25:
            mean = torch.randn((1,1)).expand(sample,feature)
            std=torch.abs(torch.randn((1,1)).expand(sample,feature))
            return std*torch.randn((sample,feature))+mean,bias

        elif random_temp<0.5:
            return torch.randn((sample,feature)),bias

        else:
            gass_number=random.randint(2,5)
            mean=torch.randn((gass_number))
            std=torch.abs(torch.randn((gass_number)))#标准差
        
            output=[torch.normal(mean=mean[i].item(),std=std[i].item(),size=[sample,feature,1]) for i in range(gass_number)]
            output=torch.concat(output,dim=-1).mean(dim=-1)
            
            return output,bias


    def get_input_value(self,input_shape):

        if len(input_shape)==3:
            batch,sample,feature=input_shape
        else:
            sample,feature=input_shape
            batch=1

        if random.uniform(0,1.)<0.5:
            random_temp=random.uniform(0,1)
            if random_temp<0.5:
                return torch.randn((batch,sample,feature))

            else:
                gass_number=random.randint(2,5)
                mean=torch.randn((gass_number))
                std=torch.abs(torch.randn((gass_number)))
                output=[torch.normal(mean[i].item(),std[i].item(),size=(batch,sample,feature,1)) for i in range(gass_number)]
                output=torch.concat(output,dim=-1).mean(dim=-1)/gass_number
                return output


        else:
            random_temp=random.uniform(0,1.)
            if random_temp<0.3:
                output=[torch.multinomial(
                            torch.rand((random.randint(2, 10))),
                            sample * feature,
                            replacement=True
                        ).reshape(1,sample,feature) for _ in range(batch)]

                output=torch.concat(output,dim=0)
                return output.float()

            elif random_temp<0.6:
                output=torch.rand((batch,sample,feature))
                return output

            else:
                output= torch.minimum(
                    torch.tensor(np.random.zipf(
                        2.0 + random.random() * 2,
                        size=(batch, sample, feature)
                        )).float(),
                        torch.tensor(10.0)
                    )
                return output.float()


class MyDataLoader(Dataset):
    def __init__(self,train_step,args):
        super(MyDataLoader,self).__init__()
        self.train_step=train_step
        self.args=args
        self.data_provide=Data_Provide(self.args)
    def __getitem__(self,index):
        self.data_provide._init_model()
        value=self.data_provide.get_data().detach().to('cpu')
        mean=value.mean(dim=-2).unsqueeze(1).expand(-1,value.shape[1],-1)
        std=value.std(dim=-2).unsqueeze(1).expand(-1,value.shape[1],-1)+1e-6
        value=(value-mean)/std
        value=torch.where(value<self.args['min_value'],self.args["min_value"],value)
        value=torch.where(value>self.args['max_value'],self.args["max_value"],value)
        return value

    def __len__(self):
        return self.train_step

                    

