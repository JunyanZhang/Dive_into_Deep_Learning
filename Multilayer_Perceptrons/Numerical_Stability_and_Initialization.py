import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
plt.show()

#正如你所看到的，当它的输入很大或是很小时，sigmoid函数的梯度都会消失。
# 此外，当反向传播通过许多层时，除非我们在刚刚好的地方，这些地方sigmoid函数的输入接近于零，否则整个乘积的梯度可能会消失。
# 当我们的网络有很多层时，除非我们很小心，否则在某一层可能会切断梯度。
# 事实上，这个问题曾经困扰着深度网络的训练。因此，更稳定（但在神经科学的角度看起来不太合理）的ReLU系列函数已经成为从业者的默认选择

#当梯度爆炸时，可能同样令人烦恼。为了更好地说明这一点，我们生成100个高斯随机矩阵，并将它们与某个初始矩阵相乘。
# 对于我们选择的尺度（方差 σ2=1 ），矩阵乘积发生爆炸。
# 当这种情况是由于深度网络的初始化所导致时，我们没有机会让梯度下降优化器收敛
M = torch.normal(0, 1, size=(4,4))
print('一个矩阵 \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('乘以100个矩阵后\n', M)