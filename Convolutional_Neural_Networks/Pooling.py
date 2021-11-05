import torch
from torch import nn
from d2l import torch as d2l


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
pool2d(X, (2, 2), 'avg')
#填充和步幅
#构造了一个输入张量 X，它有四个维度，其中样本数和通道数都是 1
X = torch.arange(16, dtype=d2l.float32).reshape((1, 1, 4, 4))
X
#深度学习框架中的步幅与池化窗口的大小相同。
# 因此，如果我们使用形状为 (3, 3) 的池化窗口，那么默认情况下，我们得到的步幅形状为 (3, 3)
pool2d = nn.MaxPool2d(3)
pool2d(X)

#填充和步幅可以手动设定。
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
#可以设定一个任意大小的矩形池化窗口，并分别设定填充和步幅的高度和宽度
pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
pool2d(X)

#处理多通道输入数据时，汇聚层在每个输入通道上单独运算，而不是像卷积层一样在通道上对输入进行汇总。
# 这意味着汇聚层的输出通道数与输入通道数相同。
# 下面，我们将在通道维度上连结张量 X 和 X + 1，以构建具有 2 个通道的输入。
X = torch.cat((X, X + 1), 1)
X

#池化后输出通道的数量仍然是 2。
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
