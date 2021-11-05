#每个卷积块中的基本单元是一个卷积层、一个 sigmoid 激活函数和平均汇聚层。
# 请注意，虽然 ReLU 和最大汇聚层更有效，但它们在20世纪90年代还没有出现。
# 每个卷积层使用  5×5  卷积核和一个 sigmoid 激活函数。
# 这些层将输入映射到多个二维特征输出，通常同时增加通道的数量。
# 第一卷积层有 6 个输出通道，而第二个卷积层有 16 个输出通道。
# 每个  2×2  池操作（步骤2）通过空间下采样将维数减少 4 倍。
# 卷积的输出形状由批量大小、通道数、高度、宽度决定。
#为了将卷积块的输出传递给稠密块，我们必须在小批量中展平每个样本。
#换言之，我们将这个四维输入转换成全连接层所期望的二维输入。
# 这里的二维表示的第一个维度索引小批量中的样本，第二个维度给出每个样本的平面向量表示。
# LeNet 的稠密块有三个全连接层，分别有 120、84 和 10 个输出。
# 因为我们仍在执行分类，所以输出层的 10 维对应于最后输出结果的数量。

import torch
from torch import nn
from d2l import torch as d2l
from d2lutil import common


class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
#在整个卷积块中，与上一层相比，每一层特征的高度和宽度都减小了。
# 第一个卷积层使用 2 个像素的填充，来补偿  5×5  卷积核导致的特征减少。
# 相反，第二个卷积层没有填充，因此高度和宽度都减少了 4 个像素。
# 随着层叠的上升，通道的数量从输入时的 1 个，增加到第一个卷积层之后的 6 个，再到第二个卷积层之后的 16 个。
# 同时，每个汇聚层的高度和宽度都减半。
# 最后，每个全连接层减少维数，最终输出一个维数与结果分类数相匹配的输出。
batch_size = 256
train_iter, test_iter = common.load_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # BERT微调所需的（之后将介绍）
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)。"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = common.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，范例数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())