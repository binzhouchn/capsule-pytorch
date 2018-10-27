# CapsNet pytorch实现（文本多分类）

CapsNet based on Geoffrey Hinton's original paper 
[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

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
or recurrent_drop ratio you want and revise in rnn_revised.py doc. 
One more thing, the revise version is non-cuda, if you find a way 
out for cuda version please let me know.