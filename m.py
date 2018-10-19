author = 'binzhou'
mdtime = '2018/10/15'

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch as t
import time

def f1_for_car(df_true:pd.DataFrame, df_pred:pd.DataFrame):
	
	'''
	f1评分标准 for 汽车行业用户观点主题及情感识别(DF竞赛)
	'''

	
	Tp, Fp, Fn = 0, 0, 0
	for cnt_id in set(df_true.content_id):
		y_true = df_true[df_true.content_id==cnt_id].copy(deep=True)
		y_pred = df_pred[df_pred.content_id==cnt_id].copy(deep=True)
		if len(y_true) > len(y_pred):
			Fn += len(y_true) - len(y_pred)
			tp = y_pred.merge(y_true,on=['subject','sentiment_value'],how='left').content_id_y.notnull().sum()
			Tp += tp
			Fp += len(y_pred) - tp
		elif len(y_true) < len(y_pred):
			Fp += len(y_pred) - len(y_true)
			tp = y_true.merge(y_pred,on=['subject','sentiment_value'],how='left').content_id_y.notnull().sum()
			Tp += tp
			Fp += len(y_true) - tp
		else:
			tp = y_true.merge(y_pred,on=['subject','sentiment_value'],how='left').content_id_y.notnull().sum()
			Tp += tp
			Fp += len(y_true) - tp
	P = Tp*1.0/(Tp+Fp)
	R = Tp*1.0/(Tp+Fn)
	return 2*P*R/(P+R)
	
	
	
class BOW(object):
	def __init__(self, X, min_count=10, maxlen=100):
		"""
		X: [[w1, w2],]]
		"""
		self.X = X
		self.min_count = min_count
		self.maxlen = maxlen
		self.__word_count()
		self.__idx()
		self.__doc2num()

	def __word_count(self):
		wc = {}
		for ws in tqdm(self.X, desc='   Word Count'):
			for w in ws:
				if w in wc:
					wc[w] += 1
				else:
					wc[w] = 1
		self.word_count = {i: j for i, j in wc.items() if j >= self.min_count}

	def __idx(self):
		self.idx2word = {i + 1: j for i, j in enumerate(self.word_count)}
		self.word2idx = {j: i for i, j in self.idx2word.items()}

	def __doc2num(self):
		doc2num = []
		for text in tqdm(self.X, desc='Doc To Number'):
			s = [self.word2idx.get(i, 0) for i in text[:self.maxlen]]
			doc2num.append(s + [0]*(self.maxlen-len(s)))  # 未登录词全部用0表示
		self.doc2num = np.asarray(doc2num)
		
class BasicModule(t.nn.Module):
	'''
	封装了nn.Module,主要是提供了save和load两个方法
	'''

	def __init__(self):
		super(BasicModule,self).__init__()
		self.model_name=str(type(self))# 默认名字

	def load(self, path,change_opt=True):
		print(path)
		data = t.load(path)
		if 'opt' in data:
			# old_opt_stats = self.opt.state_dict() 
			if change_opt:
				
				self.opt.parse(data['opt'],print_=False)
				self.opt.embedding_path=None
				self.__init__(self.opt)
			# self.opt.parse(old_opt_stats,print_=False)
			self.load_state_dict(data['d'])
		else:
			self.load_state_dict(data)
		return self.cuda()

	def save(self, name=None,new=False):
		prefix = 'checkpoints/' + self.model_name + '_' +self.opt.type_+'_'
		if name is None:
			name = time.strftime('%m%d_%H:%M:%S.pth')
		path = prefix+name

		if new:
			data = {'opt':self.opt.state_dict(),'d':self.state_dict()}
		else:
			data=self.state_dict()

		t.save(data, path)
		return path

	def get_optimizer(self,lr1,lr2=0,weight_decay = 0):
		ignored_params = list(map(id, self.encoder.parameters()))
		base_params = filter(lambda p: id(p) not in ignored_params,
						self.parameters())
		if lr2 is None: lr2 = lr1*0.5 
		optimizer = t.optim.Adam([
				dict(params=base_params,weight_decay = weight_decay,lr=lr1),
				{'params': self.encoder.parameters(), 'lr': lr2}
			])
		return optimizer