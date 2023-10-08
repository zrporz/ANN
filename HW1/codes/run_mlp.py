from network import Network
from utils import LOG_INFO
from layers import Selu, Swish, Linear, Gelu, Relu, Sigmoid
from loss import MSELoss, SoftmaxCrossEntropyLoss, HingeLoss, FocalLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import json
import time
from datetime import datetime
import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--act', type=str,default="RELU")
parser.add_argument('--loss', type=str,default="SoftmaxCrossEntropy")
parser.add_argument('--bs', type=int, default=100)
parser.add_argument('--hs', type=int, default=784)
parser.add_argument('--lr', type=float,default=1e-4)
parser.add_argument('--wd', type=float,default=1e-3)
parser.add_argument('--mom', type=float,default=0.9)
opt = parser.parse_args()
train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Linear('fc1', 784, opt.hs, 0.01,False))
if opt.act=="RELU":
    model.add(Relu('relu1'))
elif opt.act=="SELU":
    model.add(Selu('selu1'))
elif opt.act=="Swish":
    model.add(Swish('swish1'))
elif opt.act=="GELU":
    model.add(Gelu('gelu1'))
elif opt.act=="Sigmoid":
    model.add(Sigmoid('sigmoid1'))
else:
    raise ValueError("Unknown activation function: "+opt.act)

# model.add(Linear('fc2', 784, 784, 0.01))
# if opt.act=="RELU":
#     model.add(Relu('relu2'))
# elif opt.act=="SELU":
#     model.add(Selu('selu2'))
# elif opt.act=="Swish":
#     model.add(Swish('swish2'))
# elif opt.act=="GELU":
#     model.add(Gelu('gelu2'))
# elif opt.act=="Sigmoid":
#     model.add(Sigmoid('sigmoid2'))
# else:
#     raise ValueError("Unknown activation function: "+opt.act)

model.add(Linear('fc2', opt.hs, 10, 0.01))

if opt.loss=="MSE":
    loss = MSELoss(name="mseloss")
elif opt.loss=="SoftmaxCrossEntropy":
    loss = SoftmaxCrossEntropyLoss(name='SoftmaxCrossEntropyLoss')
elif opt.loss=="Hinge":
    loss = HingeLoss(name="HingeLoss")
elif opt.loss=="Focal":
    loss = FocalLoss(name="FocalLoss")
else:
    raise ValueError("Unknown loss function: "+opt.loss)

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': opt.lr,
    'weight_decay': opt.wd,
    'momentum': opt.mom,
    'batch_size': opt.bs,
    'max_epoch': 100,
    'disp_freq': 50,
    'test_epoch': 5
}

class Log:
    def __init__(self,log):
        self.log = log
    def append(self,key,value):
        self.log[key].append(value)
    def save(self):
        name = ""
        for layer_name in self.log["model"]:
            name += layer_name+"_"
        name += self.log["loss"] +"_"
        name += time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        with open(f"./log/hyperparam/{name}.json","w") as f:
            f.write(json.dumps(self.log,indent=4))

print(config)
for layer in model.layer_list:
    print(layer.name)
print(loss.name)
model_print_list = [layer.name for layer in model.layer_list]
log = Log({"model":model_print_list,"time":0,"loss":loss.name,"config":config,"train_loss":[],"train_acc":[],"test_loss":[],"test_acc":[]})
time1 = datetime.now()
for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'],log)

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'], log)
time2 = datetime.now()
log.log["time"] = (time2-time1).total_seconds()
log.save()
