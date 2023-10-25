# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features, mom=0.9):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features

		# Parameters
		self.weight = Parameter(torch.Tensor(num_features))
		self.bias = Parameter(torch.Tensor(num_features)) 
		self.momentum = mom
		# Store the average mean and variance
		self.register_buffer('running_mean',torch.zeros(num_features) )
		self.register_buffer('running_var',torch.ones(num_features) )

		
		# Initialize your parameter
		init.ones_(self.weight)
		init.zeros_(self.bias)
		self.eps = 1e-5

	def forward(self, input):
		if self.training:
			mu = input.mean(dim=(0,2,3))
			var = input.var(dim=(0,2,3))
			self.running_mean = self.running_mean * self.momentum + mu * (1 - self.momentum)
			self.running_var = self.running_var * self.momentum + var * (1 - self.momentum)
		else:
			mu = self.running_mean
			var = self.running_var
		# import pdb; pdb.set_trace()
		input = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * ((input - mu.unsqueeze(0).unsqueeze(2).unsqueeze(3)) / torch.sqrt(var.unsqueeze(0).unsqueeze(2).unsqueeze(3) + 1e-5))+ self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
		return input
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			# import pdb; pdb.set_trace()
			return (input * (torch.bernoulli(torch.ones(input.shape[:2]).to(input.device)*(1-self.p)).unsqueeze(2).unsqueeze(3)) / (1-self.p))
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, layer1,layer2,drop_rate):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		drop_rate1 = drop_rate[0]
		drop_rate2 = drop_rate[1]
		layer1_config = layer1
		layer2_config = layer2
		self.layer1 = nn.Sequential(
			nn.Conv2d(layer1_config["conv"]["in_channel"], layer1_config["conv"]["out_channel"], layer1_config["conv"]["kernel"], layer1_config["conv"]["stride"], layer1_config["conv"]["padding"]),
			BatchNorm2d(layer1_config["conv"]["out_channel"]),
			nn.ReLU(),
			Dropout(drop_rate1),
			nn.MaxPool2d(layer1_config["maxpool"]["kernel"],layer1_config["maxpool"]["stride"],layer1_config["maxpool"]["padding"])
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(layer2_config["conv"]["in_channel"], layer2_config["conv"]["out_channel"], layer2_config["conv"]["kernel"], layer2_config["conv"]["stride"], layer2_config["conv"]["padding"]),
			BatchNorm2d(layer2_config["conv"]["out_channel"]),
			nn.ReLU(),
			Dropout(drop_rate2),
			nn.MaxPool2d(layer2_config["maxpool"]["kernel"],layer2_config["maxpool"]["stride"],layer2_config["maxpool"]["padding"])
		)
		N = 32
		for layer_config in (layer1_config["conv"],layer1_config["maxpool"],layer2_config["conv"],layer2_config["maxpool"]):
			# assert((N - layer_config["kernel"] + 2 * layer_config["padding"]) % layer_config["stride"] == 0)
			N = (N - layer_config["kernel"] + 2 * layer_config["padding"]) // layer_config["stride"] + 1
			print(N)
		self.fc = nn.Linear(layer2_config["conv"]["out_channel"] * N * N, 10) 
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		x = self.layer1(x)
		x = self.layer2(x)
		x = x.flatten(start_dim=1)
		# import pdb; pdb.set_trace()
		logits = self.fc(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc




