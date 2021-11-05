import torch
import  sys
from torch import nn
from d2l import torch as d2l

sys.path.append('F:\\web_auto\\Dive_into_deeplearning\\d2lutil')  # 加入路径，添加目录
import common
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = common.load_fashion_mnist(batch_size)
common.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)