import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn import metrics
import sklearn
from matplotlib import pyplot as plt
from torchvision import transforms,datasets
from torchvision.datasets import MNIST

class BinarizeAct(torch.autograd.Function):
	@staticmethod
	def forward(cls, input):
		cls.save_for_backward(input)
		return (input >= 0).float()*2-1

	@staticmethod
	def backward(cls, grad_output):
		input = cls.saved_tensors[0]
		status = (torch.abs(input)<=1).float()
		return grad_output*status

	def __call__(self,x):
		return self.apply(x)


class BinarizeWeight(torch.autograd.Function):
	@staticmethod
	def forward(cls, input):
		cls.save_for_backward(input)
		return (input >= 0).float()*2-1

	@staticmethod
	def backward(cls, grad_output):
		return grad_output

	def __call__(self,x):
		return self.apply(x)

class Block(nn.Module):
	def __init__(self,in_num,out_num,kernel_size,binarized=True):
		super().__init__()
		self.binarized = binarized
		self.norm = nn.BatchNorm2d(out_num,affine=False)
		self.net = nn.Conv2d(in_num,out_num,kernel_size)
		self.binarize_act = BinarizeAct()
		self.binarize_weight = BinarizeWeight()
		self.clip_weights()
				
	def forward(self,x,act=True,norm=True):
		if self.binarized:
			w,b = self.get_binarize_net()
		else:
			w,b = self.net.weight,self.net.bias
		x = F.conv2d(x, w, b)
		if norm:
			x = self.norm(x)
		if act:
			if self.binarized:
				x = self.binarize_act(x)
			else:
				x = x.relu()
		return x
				
	def get_binarize_net(self):
		w = self.binarize_weight(self.net.weight)
		b = self.binarize_weight(self.net.bias)
		return w,b

	def clip_weights(self):
		if self.binarized:
			self.net.weight.values = torch.clip(self.net.weight,-1,1)
			self.net.bias.values = torch.clip(self.net.bias,-1,1)

class Model(nn.Module):
	def __init__(self,in_num,out_num,binarized=True):
		super().__init__()
		base = 8
		self.b1 = Block(in_num,base*4,9,binarized)
		self.b2 = Block(base*4,base*2,9,binarized)
		self.b3 = Block(base*2,base,9,binarized)
		self.binarized = binarized
		self.binarize_act = BinarizeAct()
		self.last = Block(base,out_num,4)
			
	def forward(self,x):
		if self.binarized:
			x = self.binarize_act(x)
		x = self.b3(self.b2(self.b1(x)))
		return self.last(x,False,False).reshape(-1,10)

	def clip_weights(self):
		for child in self.children():
			if hasattr(child,'clip_weights'):
				child.clip_weights()

def train_step(model,loader,optim,device):
	loss_func = nn.CrossEntropyLoss()
	predicts = []
	answers = []
	loss_sum = 0
	for x in loader:
		model.train(True)
		optim.zero_grad()
		y = model(x[0].to(device))
		loss = loss_func(y,x[1].to(device))
		predicts += y.argmax(1).detach().cpu().numpy().tolist()
		answers += x[1].detach().cpu().numpy().tolist()
		loss.backward()
		loss_sum += loss
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1,'Inf')
		optim.step()
		model.clip_weights()
	f1 = metrics.f1_score(answers,predicts,average='macro')
	loss = loss_sum/len(loader)
	return f1,loss,answers,predicts

def test_step(model,loader,device):
	loss_func = nn.CrossEntropyLoss()
	loss_sum = 0
	predicts = []
	answers = []
	for x in loader:
		model.train(False)
		with torch.no_grad():
			y = model(x[0].to(device))
			loss = loss_func(y,x[1].to(device))
			loss_sum += loss
			predicts += y.argmax(1).detach().cpu().numpy().tolist()
			answers += x[1].detach().cpu().numpy().tolist()
	f1 = sklearn.metrics.f1_score(answers,predicts,average='macro')
	loss = loss_sum/len(loader)
	return f1,loss,answers,predicts
	
def main(binarized=True):
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),])
	data = datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
	test_data = datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)
	num = int(np.floor(len(data)*0.2))
	generator = torch.Generator().manual_seed(0)
	train_data, val_data = torch.utils.data.random_split(data, [num*4, num],generator=generator)
	train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128, shuffle=True,drop_last=True)
	val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=128, shuffle=False,drop_last=False)
	test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=False,drop_last=False)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = Model(1,10,binarized).to(device)
	optim = torch.optim.Adam(model.parameters(),lr=0.01)
	lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=0.5,patience=2)
	losses = []
	val_losses = []
	f1s = []
	val_f1s = []
	min_loss = None
	patience = 0
	for i in range(100):
		print(i)
		f1,loss,answers,predicts = train_step(model,train_loader,optim,device)
		f1s.append(f1)
		losses.append(loss)
		val_f1,val_loss,val_answers,val_predicts = test_step(model,val_loader,device)
		lr_scheduler.step(val_loss)
		val_f1s.append(val_f1)
		val_losses.append(val_loss)
		print(f1,val_f1)
		if min_loss is None or val_loss<=min_loss:
			patience = 0
			min_loss = val_loss
		else:
			patience += 1
		if patience==5:
			break
	test_f1,test_loss,test_answers,test_predicts = test_step(model,test_loader,device)
	return model,optim,lr_scheduler,losses,val_losses,f1s,val_f1s,test_f1,test_loss,test_answers,test_predicts
