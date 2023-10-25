# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features,mom=0.9):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features

		# Parameters
		self.weight = Parameter(torch.Tensor(num_features))
		self.bias = Parameter(torch.Tensor(num_features))
		self.momentum = mom
		self.register_buffer('running_mean',torch.zeros(num_features) )
		self.register_buffer('running_var',torch.ones(num_features) )
		
		# Initialize your parameter
		init.ones_(self.weight)
		init.zeros_(self.bias)

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			mu = input.mean(dim=0)
			var = input.var(dim=0)
			self.running_mean = self.running_mean * self.momentum + mu * (1 - self.momentum)
			self.runnin_var = self.running_var * self.momentum + var * (1 - self.momentum)
		else:
			mu = self.running_mean
			var = self.running_var
		# import pdb; pdb.set_trace()
		input = self.weight * ((input - mu) / torch.sqrt(var + 1e-5))+ self.bias
		return input
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			return (input * torch.bernoulli(torch.ones(input.shape).to(input.device)*(1-self.p)) / (1-self.p))
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5,hidden_features = 1024):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		in_features = 3072
		hidden_features = hidden_features
		self.linear_layer1 = nn.Linear(in_features,hidden_features)
		self.batch_norm_layer = BatchNorm1d(hidden_features)
		self.dropout_layer = Dropout(drop_rate)
		self.linear_layer2 = nn.Linear(hidden_features, 10)
		print(f"drop_rate:{drop_rate}")
		print(f"hidden_features:{hidden_features}")
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		x = self.linear_layer1(x)
		x = (self.batch_norm_layer(x))
		x = nn.ReLU()(x)
		x = self.dropout_layer(x)
		x = self.linear_layer2(x)
		logits = x
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc


