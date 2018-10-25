#!/usr/bin/python
# -*- coding: utf-8 -*-

author = 'BinZhou'
nick_name = '发送小信号'
mtime = '2018/10/19'

import torch as t
import torch.nn as nn
import torch.nn.functional as F


USE_CUDA = True
embedding_dim = 300
embedding_path = '../save/embedding_matrix.npy' # or False, not use pre-trained-matrix
use_pretrained_embedding = True
BATCH_SIZE = 64
gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28
LR = 0.001
T_epsilon=1e-7
num_classes = 30


class Embed_Layer(nn.Module):
    
    def __init__(self, embedding_matrix=None, vocab_size=None, embedding_dim=300):
        super(Embed_Layer, self).__init__()
        self.encoder = nn.Embedding(vocab_size+1,embedding_dim)
        if use_pretrained_embedding:
            # self.encoder.weight.data.copy_(t.from_numpy(np.load(embedding_path))) # 方法一，加载np.save的npy文件
            self.encoder.weight.data.copy_(t.from_numpy(embedding_matrix)) # 方法二
    def forward(self, x, rate_drop_dense=0.28):
        return nn.Dropout(p=rate_drop_dense)(self.encoder(x))

class GRU_Layer(nn.Module):
    
    def __init__(self):
        super(GRU_Layer, self).__init__()
        self.gru = nn.GRU(input_size=embedding_dim,
                         hidden_size=gru_len,
                         bidirectional=True)     
    def forward(self, x):
        return self.gru(x)

# core caps_layer with squash func
class Caps_Layer(nn.Module):
    def __init__(self, input_dim_capsule=gru_len*2, num_capsule=Num_capsule, dim_capsule=Dim_capsule, \
                 routings=Routings, kernel_size=(9, 1), share_weights=True,
                activation='default',**kwargs):
        super(Caps_Layer, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size # 暂时没用到
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)
        
        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(t.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(nn.init.xavier_normal_(t.empty(64, input_dim_capsule,self.num_capsule * self.dim_capsule))) #64即batch_size

    def forward(self, x):
        
        if self.share_weights:
            u_hat_vecs = t.matmul(x, self.W)
        else:
            print('add later')
            
        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3) # 转成(batch_size,num_capsule,input_num_capsule,dim_capsule)
        b = t.zeros_like(u_hat_vecs[:, :, :, 0]) # (batch_size,num_capsule,input_num_capsule)
        
        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.activation(t.einsum('bij,bijk->bik', (c, u_hat_vecs))) # batch matrix multiplication
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = t.einsum('bik,bijk->bij', (outputs, u_hat_vecs)) # batch matrix multiplication
        return outputs # (batch_size, num_capsule, dim_capsule)
    
    # text version of squash, slight different from original one
    def squash(self, x, axis=-1):
        s_squared_norm  = (x ** 2).sum(axis, keepdim=True)
        scale = t.sqrt(s_squared_norm + T_epsilon)
        return x / scale
    
#     # original one for image decoder
#     def squash(self, input_tensor):
#         squared_norm = (input_tensor ** 2).sum(-1, keepdim=True) + T_epsilon
#         output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * t.sqrt(squared_norm))
#         return output_tensor

class Dense_Layer(nn.Module):
    def __init__(self):
        super(Dense_Layer, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p,inplace=True),    
            nn.Linear(Num_capsule*Dim_capsule, num_classes), # num_capsule*dim_capsule -> num_classes
            nn.Sigmoid()
        )
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc(x)

# capsule如果单纯做分类则不需要重构(reconstruction)
# 如果就用在分类里面，decoder用不到，不需要reconstruction

class Capsule_Main(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None):
        super(Capsule_Main, self).__init__()
        self.embed_layer = Embed_Layer(embedding_matrix, vocab_size)
        self.gru_layer = GRU_Layer()
        self.caps_layer = Caps_Layer()
        self.dense_layer = Dense_Layer()
    
    def forward(self, content):
        content = self.embed_layer(content)
        content = self.gru_layer(content)[0] # 这个输出是个tuple，一个output(seq_len, batch_size, num_directions * hidden_size)，一个hn
        content = self.caps_layer(content)
        output = self.dense_layer(content)
        return output