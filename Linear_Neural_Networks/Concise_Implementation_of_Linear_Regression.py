import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# `nn` 是神经网络的缩写
from torch import nn

#生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train,num_workers=0)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

#首先定义一个模型变量net，它是一个Sequential类的实例。
# Sequential类为串联在一起的多个层定义了一个容器。
#当给定输入数据，Sequential实例将数据传入到第一层，然后将第一层的输出作为第二层的输入，依此类推
net = nn.Sequential(nn.Linear(2, 1))
#直接访问参数以设定初始值。通过net[0]选择网络中的第一个图层，然后使用weight.data和bias.data方法访问参数。
#使用替换方法normal_和fill_来重写参数值。
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

#计算均方误差使用的是MSELoss类，也称为平方 L2 范数。默认情况下，它返回所有样本损失的平均值。
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

#训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
# index 1 is out of range
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
