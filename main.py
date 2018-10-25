#!/usr/bin/python
# -*- coding: utf-8 -*-

author = 'BinZhou'
nick_name = '发送小信号'
mtime = '2018/10/19'

import torch
import torch.utils.data as Data
import torch as t
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import pandas as pd
import jieba
import gensim
from gensim.models import Word2Vec, FastText
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# tfidf or countvec for lr or svm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import copy

from m import f1_for_car, BOW, BasicModule
from capsule_layer import *

# 以训练数据为例
data = pd.read_csv('data/train.csv')
data['content'] = data.content.map(lambda x: ''.join(x.strip().split()))

# 把主题和情感拼接起来，一共10*3类
data['label'] = data['subject'] + data['sentiment_value'].astype(str)
subj_lst = list(filter(lambda x : x is not np.nan, list(set(data.label))))
subj_lst_dic = {value:key for key, value in enumerate(subj_lst)}
data['label'] = data['label'].apply(lambda x : subj_lst_dic.get(x))

# 处理同一个句子对应对标签的情况，然后进行MLB处理
data_tmp = data.groupby('content').agg({'label':lambda x : set(x)}).reset_index()
# [[1,0,0],[0,1,0],[0,0,1]]
# 可能有多标签则[[1,1,0],[0,1,0],[0,0,1]]
mlb = MultiLabelBinarizer()
data_tmp['hh'] = mlb.fit_transform(data_tmp.label).tolist()
y_train = np.array(data_tmp.hh.tolist())

# 构造embedding字典

bow = BOW(data_tmp.content.apply(jieba.lcut).tolist(), min_count=1, maxlen=100) # 长度补齐或截断固定长度100

# word2vec = Word2Vec(data_tmp.content.apply(jieba.lcut).tolist(),size=300,min_count=1)
word2vec = gensim.models.KeyedVectors.load_word2vec_format('data/ft_wv.txt') # 读取txt文件的预训练词向量

vocab_size = len(bow.word2idx)
embedding_matrix = np.zeros((vocab_size+1, 300))
for key, value in bow.word2idx.items():
    if key in word2vec.vocab: # Word2Vec训练得到的的实例需要word2vec.wv.vocab
        embedding_matrix[value] = word2vec.get_vector(key)
    else:
        embedding_matrix[value] = [0] * embedding_dim

X_train = copy.deepcopy(bow.doc2num)
y_train = copy.deepcopy(y_train)
#--------------------------------------------

# 数据处理成tensor
BATCH_SIZE = 64
label_tensor = torch.from_numpy(np.array(y_train)).float()
content_tensor = torch.from_numpy(np.array(X_train)).long()

torch_dataset = Data.TensorDataset(content_tensor, label_tensor)
train_loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=8,              # subprocesses for loading data
    )

# 网络结构、损失函数、优化器初始化
capnet = Capsule_Main(embedding_matrix) # 加载预训练embedding matrix
loss_func = nn.BCELoss() # 用二分类方法预测是否属于该类，而非多分类
if USE_CUDA:
    capnet = capnet.cuda() # 把搭建的网络载入GPU
    loss_func.cuda() # 把损失函数载入GPU
optimizer = Adam(capnet.parameters(),lr=LR) # 默认lr

# 开始跑模型
it = 1
EPOCH = 30
for epoch in tqdm_notebook(range(EPOCH)):
    for batch_id, (data, target) in enumerate(train_loader):
        if USE_CUDA:
            data, target = data.cuda(), target.cuda() # 数据载入GPU
        output = capnet(data)
        loss = loss_func(output, target)
        if it % 50 == 0:
            print('training loss: ', loss.cpu().data.numpy().tolist())
            print('training acc: ', accuracy_score(np.argmax(target.cpu().data.numpy(),axis=1), np.argmax(output.cpu().data.numpy(),axis=1)))
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        it += 1

