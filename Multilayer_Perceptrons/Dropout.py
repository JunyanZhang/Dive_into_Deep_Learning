import torch
from torch import nn
from d2l import torch as d2l
from d2lutil import common

#要实现单层的dropout函数，我们必须从伯努利（二元）随机变量中提取与我们的层的维度一样多的样本，
# 其中随机变量以概率 1−p 取值 1 （保持），以概率 p 取值 0 （丢弃）。
# 实现这一点的一种简单方式是首先从均匀分布 U[0,1] 中抽取样本。
# 那么我们可以保留那些对应样本大于 p 的节点，把剩下的丢弃。
#在下面的代码中，我们实现 dropout_layer 函数，该函数以dropout的概率丢弃张量输入X中的元素，如上所述重新缩放剩余部分：将剩余部分除以1.0-dropout
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃。
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留。
    if dropout == 0:
        return X
    mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()
    return mask * X / (1.0 - dropout)
#可以通过几个例子来测试dropout_layer函数。
# 在下面的代码行中，我们将输入X通过dropout操作，丢弃概率分别为0、0.5和1
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))

#引入的Fashion-MNIST数据集。我们定义具有两个隐藏层的多层感知机，每个隐藏层包含256个单元
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

#定义模型
#模型将dropout应用于每个隐藏层的输出（在激活函数之后）。我们可以分别为每一层设置丢弃概率。
#一种常见的技巧是在靠近输入层的地方设置较低的丢弃概率。
# 下面，我们将第一个和第二个隐藏层的丢弃概率分别设置为0.2和0.5。
# 我们确保dropout只在训练期间有效
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
#类似于前面描述的多层感知机训练和测试
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = common.load_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
common.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)