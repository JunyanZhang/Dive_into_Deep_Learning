import torch
import torchvision
import torchvision.transforms as transforms
from d2l import torch as d2l
from torch.utils import data
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import extract_archive
from IPython import display
import matplotlib.pyplot as plt
from torch import nn


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    d2l.plt.show()


def hello():
    print("semilogy_HELLO")

# 李沐课件中采用的是远程获取的方式，因为公司网络的限制，远程获取会报错。
# 运行远程获取的方式，c:\users\lwx898760\miniconda3\envs\d2l\lib\site-packages\torchvision\datasets\mnist.py会报错
# 这里采用本地下载的方式先将数据集下载到本地，放在D://d2l-data//下面
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
def load_fashion_mnist(batch_size,resize=None):
    extract_archive('F://web_auto//Dive_into_deeplearning//data//t10k-images-idx3-ubyte.gz', 'F://web_auto//Dive_into_deeplearning//data//FashionMNIST//raw', False)
    extract_archive('F://web_auto//Dive_into_deeplearning//data//train-images-idx3-ubyte.gz', 'F://web_auto//Dive_into_deeplearning//data//FashionMNIST//raw', False)
    extract_archive('F://web_auto//Dive_into_deeplearning//data//t10k-labels-idx1-ubyte.gz', 'F://web_auto//Dive_into_deeplearning//data//FashionMNIST//raw', False)
    extract_archive('F://web_auto//Dive_into_deeplearning//data//train-labels-idx1-ubyte.gz', 'F://web_auto//Dive_into_deeplearning//data//FashionMNIST//raw', False)

    training_set = (
        read_image_file('F://web_auto//Dive_into_deeplearning//data//FashionMNIST//raw//train-images-idx3-ubyte'),
        read_label_file('F://web_auto//Dive_into_deeplearning//data//FashionMNIST//raw//train-labels-idx1-ubyte')
    )
    test_set = (
        read_image_file('F://web_auto//Dive_into_deeplearning//data//FashionMNIST//raw//t10k-images-idx3-ubyte'),
        read_label_file('F://web_auto//Dive_into_deeplearning//data//FashionMNIST//raw//t10k-labels-idx1-ubyte')
    )
    with open('F://web_auto//Dive_into_deeplearning//data//FashionMNIST//processed//training.pt', 'wb') as f:
        torch.save(training_set, f)
    with open('F://web_auto//Dive_into_deeplearning//data//FashionMNIST//processed//test.pt', 'wb') as f:
        torch.save(test_set, f)
    print('Done!')
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    #train_data, train_targets = torch.load('D://d2l-data//FashionMNIST//processed//training.pt')
    #test_data, test_targets = torch.load('D://d2l-data//FashionMNIST//processed//test.pt')

    mnist_train = torchvision.datasets.FashionMNIST(root="F:/web_auto/Dive_into_deeplearning/data", train=True, transform=trans,
                                                    download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="F:/web_auto/Dive_into_deeplearning/data", train=False, transform=trans,
                                                   download=False)

    # 这里有个坑 如果线程数num_workers设置大于0会报错  An attempt has been made to start a new process before the current process has finished its bootstrapping
    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return (train_iter, test_iter)
#可以调用框架中现有的API来读取数据。
# 我们将features和labels作为API的参数传递，并在实例化数据迭代器对象时指定batch_size。
# 此外，布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train,num_workers=0)
#如果y_hat是矩阵，那么假定第二个维度存储每个类的预测分数。
# 我们使用argmax获得每行中最大元素的索引来获得预测类别。
# 然后我们将预测类别与真实y元素进行比较。
# 由于等式运算符“==”对数据类型很敏感，因此我们将y_hat的数据类型转换为与y的数据类型一致。
# 结果是一个包含0（错）和1（对）的张量。进行求和会得到正确预测的数量
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

#对于任意数据迭代器data_iter可访问的数据集，我们可以评估在任意模型net的准确率
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

#Accumulator是一个实用程序类，用于对多个变量进行累加。
#在上面的evaluate_accuracy函数中，我们在Accumulator实例中创建了2个变量，用于分别存储正确预测的数量和预测的总数量。
#当我们遍历数据集时，两者都将随着时间的推移而累加
class Accumulator:  #@save
    """在`n`个变量上累加。"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
#定义一个在动画中绘制数据的实用程序类
class Animator:  #@save
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        plt.draw()
        plt.pause(0.001)
        display.clear_output(wait=True)
#我们定义一个函数来训练一个迭代周期。请注意，updater是更新模型参数的常用函数，它接受批量大小作为参数。
# 它可以是封装的d2l.sgd函数，也可以是框架的内置优化函数
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）。"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y),
                       y.size().numel())
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]
#实现一个训练函数，它会在train_iter访问到的训练数据集上训练一个模型net。
# 该训练函数将会运行多个迭代周期（由num_epochs指定）。
# 在每个迭代周期结束时，利用test_iter访问到的测试数据集对模型进行评估。
# 我们将利用Animator类来可视化训练进度
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）。"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）。"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

#让我们实现一个函数来评估模型在给定数据集上的损失
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失。"""
    metric = d2l.Accumulator(2)  # 损失的总和, 样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


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
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
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
    print(timer.sum())
