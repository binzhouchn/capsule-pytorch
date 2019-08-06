# CapsNet pytorch实现（文本多分类）

CapsNet based on Geoffrey Hinton's original paper 
[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

[先读懂CapsNet架构然后用TensorFlow实现：全面解析Hinton提出的Capsule](https://www.jiqizhixin.com/articles/2017-11-05)

## Requirements

 - python 3.6+
 - pytorch 0.4.1+
 - gensim
 - tqdm
   
## Run

```bash
python main.py
```
Train and test dataset should be included in data folder

## DIY

If you need hard_sigmoid for GRU gate, just uncomment
```python
from rnn_revised import *
```
in capsule_layer.py. You can also use whatever activation func 
or dropout/recurrent_dropout ratio you want and revise in rnn_revised.py doc. 
One more thing, the revise version is non-cuda, if you find a way 
out for cuda version please let me know.

注：<br>
1. PrimaryCapsLayer中的squash压缩的是向量size是[batch_size, 1152, 8]，在最后一个维度上进行压缩即维度8
压缩率|Sj|2/(1+|Sj|2)/|Sj|大小为[batch_size, 1152]，然后与原来的输入向量相乘即可

2. 如果reconstruction为True，则loss由两部分组成margin_loss和reconstruction_loss<br>
```python
output, probs = model(data, target)
reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
margin_loss = loss_fn(probs, target)
# 如果reconstruction为True，则loss由两部分组成margin_loss和reconstruction_loss
loss = reconstruction_alpha * reconstruction_loss + margin_loss
```