import torch
import  sys
from torch import nn
from d2l import torch as d2l

sys.path.append('F:\\web_auto\\Dive_into_deeplearning\\d2lutil')  # 加入路径，添加目录
import common
batch_size = 256
train_iter, test_iter = common.load_fashion_mnist(batch_size)

#Fashion-MNIST中的每个图像由 28×28=784 个灰度像素值组成。所有图像共分为10个类别。忽略像素之间的空间结构，我们可以将每个图像视为具有784个输入特征和10个类的简单分类数据集。首先，我们将实现一个具有单隐藏层的多层感知机，它包含256个隐藏单元。
#  注意，我们可以将这两个量都视为超参数。
#  通常，我们选择2的若干次幂作为层的宽度。
#  因为内存在硬件中的分配和寻址方式，这么做往往可以在计算上更高效
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

loss = nn.CrossEntropyLoss()

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
common.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)