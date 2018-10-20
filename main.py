#!/usr/bin/python
# -*- coding: utf-8 -*-

author = 'BinZhou'
nick_name = '发送小信号'
mtime = '2018/10/19'

import torch
import torch.utils.data as Data
import torch as t
from torch import nn
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
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from keras.preprocessing import text, sequence

from m import f1_for_car, BOW, BasicModule
from capsule_layer import *

# 以训练数据为例
data = pd.read_csv('data/train.csv')

# 把主题和情感拼接起来，一共10*3类
data['label'] = data['subject'] + data['sentiment_value'].astype(str)
subj_lst = list(filter(lambda x : x is not np.nan, list(set(data.label))))
subj_lst_dic = {value:key for key, value in enumerate(subj_lst)}
data['label'] = data['label'].apply(lambda x : subj_lst_dic.get(x))

# 构造embedding字典
train_dict = {}
for ind, row in train_df.iterrows():
    content, label = row['content'], row['label']
    if train_dict.get(content) is None:
        train_dict[content] = set([label])
    else:
        train_dict[content].add(label)
        
conts = []
labels = []
for k, v in train_dict.items():
    conts.append(k)
    labels.append(v)

# [[1,0,0],[0,1,0],[0,0,1]]
# 可能有多标签则[[1,1,0],[0,1,0],[0,0,1]]
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(labels)
content_list = [jieba.lcut(str(c)) for c in conts]
word_set = set([word for row in list(content_list) for word in row])

word2index = {w: i + 1 for i, w in enumerate(word_set)}
seqs = [[word2index[w] for w in l] for l in content_list]

word2vec = Word2Vec(data.content.apply(jieba.lcut).tolist(),size=300,min_count=1)

word_embed_dict = {}
def get_word_embed_dict():
    for i in word2vec.wv.vocab:
        word_embed_dict[i] = word2vec.wv.get_vector(i).tolist()
    return word_embed_dict
word_embed_dict = get_word_embed_dict()

EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(word2index) + 1, EMBEDDING_DIM))
for word, i in word2index.items():
    embedding_vector = word_embed_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
max_features = len(word_set) + 1

# # 保存成embedding_matrix.npy，后面跑神经网络的时候会读取，如果用另一种加载方式则不需要先存成文件
# np.save('../save/embedding_matrix',arr=embedding_matrix)

# 长度补齐，这里用的是keras里面的方法，也可以用我自己写的BOW里面的方法，0补在后面，跑双向RNN不影响
def get_padding_data(maxlen=100):
    x_train = sequence.pad_sequences(seqs, maxlen=maxlen)
    return x_train
maxlen = 100
X_train = get_padding_data(maxlen).astype(int)
print(X_train.shape, y_train.shape)

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

