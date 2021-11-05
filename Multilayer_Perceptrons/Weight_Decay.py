import torch
from torch import nn
from d2l import torch as d2l

#选择标签是关于输入的线性函数。
# 标签同时被均值为0，标准差为0.01高斯噪声破坏。
# 为了使过拟合的效果更加明显，我们可以将问题的维数增加到 d=200 ，并使用一个只包含20个样本的小训练集
from d2lutil import common

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

#将从头开始实现权重衰减，只需将 L2 的平方惩罚添加到原始目标函数中
#将定义一个函数来随机初始化我们的模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
#实现这一惩罚最方便的方法是对所有项求平方后并将它们求和。
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

#线性网络和平方损失没有变化，所以我们通过d2l.linreg和d2l.squared_loss导入它们。
#唯一的变化是损失现在包括了惩罚项。
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = common.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                # 增加了L2范数惩罚项，广播机制使l2_penalty(w)成为一个长度为`batch_size`的向量。
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())
#现在用lambd = 0禁用权重衰减后运行这个代码。
# 注意，这里训练误差有了减少，但测试误差没有减少。
#  这意味着出现了严重的过拟合。这是过拟合的一个典型例子
train(lambd=0)

#使用权重衰减来运行代码。
# 注意，在这里训练误差增大，但测试误差减小。
# 这正是我们期望从正则化中得到的效果
train(lambd=3)

#简洁实现
#在实例化优化器时直接通过weight_decay指定weight decay超参数。
# 默认情况下，PyTorch同时衰减权重和偏移。
# 这里我们只为权重设置了weight_decay，所以bias参数 b 不会衰减
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减。
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = common.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                trainer.zero_grad()
                l = loss(net(X), y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (common.evaluate_loss(net, train_iter, loss),
                                     common.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
