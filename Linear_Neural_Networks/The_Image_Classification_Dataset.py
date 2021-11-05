import torch
import sys
import os
from d2l import torch as d2l
import matplotlib.pyplot as plt

sys.path.append('F:\\web_auto\\Dive_into_deeplearning\\d2lutil')  # 加入路径，添加目录
import common
#def use_svg_display():  #@save
   # """使用svg格式在Jupyter中显示绘图。"""
    #display.set_matplotlib_formats('svg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
# 并除以255使得所有像素的数值均在0到1之间
#通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中
batch_size = 256
resize=64
mnist_train,mnist_test = common.load_fashion_mnist(batch_size)

#print(len(mnist_train), len(mnist_test))
#print(mnist_train[0])
for X, y in mnist_train:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break


## 展示部分数据
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

#创建一个函数来可视化这些样本
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
#训练数据集中前几个样本的图像及其相应的标签（文本形式）
#展示部分训练数据
train_data, train_targets = iter(mnist_train).next()
show_fashion_mnist(train_data[0:10], train_targets[0:10])
