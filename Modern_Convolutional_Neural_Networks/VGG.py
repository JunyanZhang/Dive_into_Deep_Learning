#经典卷积神经网络的基本组成部分是下面的这个序列： 1. 带填充以保持分辨率的卷积层； 1. 非线性激活函数，如ReLU； 1. 汇聚层，如最大汇聚层。

#而一个 VGG 块与之类似，由一系列卷积层组成，后面再加上用于空间下采样的最大汇聚层。
#在最初的 VGG 论文 [Simonyan & Zisserman, 2014] 中，作者使用了带有  3×3  卷积核、填充为 1（保持高度和宽度）的卷积层，和带有  2×2  池化窗口、步幅为 2（每个块后的分辨率减半）的最大汇聚层。
# 在下面的代码中，我们定义了一个名为 vgg_block 的函数来实现一个 VGG 块
import torch
from torch import nn
from d2l import torch as d2l
from d2lutil import common

#该函数有三个参数，分别对应于卷积层的数量 num_convs、输入通道的数量 in_channels 和输出通道的数量 out_channels.
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
#实现了 VGG-11。可以通过在 conv_arch 上执行 for 循环来简单实现
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
#将构建一个高度和宽度为 224 的单通道数据样本，以观察每个层输出的形状
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

#VGG-11比AlexNet计算量更大，因此我们构建了一个通道数较少的网络，足够用于训练Fashion-MNIST数据集。
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
#除了使用略高的学习率外，模型训练过程与 AlexNet 类似。
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = common.load_fashion_mnist(batch_size, resize=224)
common.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
