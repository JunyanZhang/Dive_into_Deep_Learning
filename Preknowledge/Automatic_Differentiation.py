import torch

x = torch.arange(4.0)

#在我们计算 y 关于 x 的梯度之前，我们需要一个地方来存储梯度。
# 重要的是，我们不会在每次对一个参数求导时都分配新的内存。
# 因为我们经常会成千上万次地更新相同的参数，每次都分配新的内存可能很快就会将内存耗尽。
# 注意，标量函数关于向量 x 的梯度是向量，并且与 x 具有相同的形状。
x.requires_grad_(True)  # 等价于 `x = torch.arange(4.0, requires_grad=True)`
x.grad  # 默认值是None
#计算 y
y = 2 * torch.dot(x, x)

#x是一个长度为4的向量，计算x和x的内积，得到了我们赋值给y的标量输出。
# 接下来，我们可以通过调用反向传播函数来自动计算y关于x每个分量的梯度，并打印这些梯度
y.sum().backward()
x.grad
#x.grad == 4 * x

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad

#可以分离y来返回一个新变量u，该变量与y具有相同的值，但丢弃计算图中如何计算y的任何信息。
# 换句话说，梯度不会向后流经u到x。
# 因此，下面的反向传播函数计算z=u*x关于x的偏导数，同时将u作为常数处理，而不是z=x*x*x关于x的偏导数
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

a.grad == d / a