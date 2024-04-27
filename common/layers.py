import numpy as np
from .functions import *


# 定义一个ReLU激活函数类
class Relu:
    # 初始化方法，创建一个ReLU对象时调用
    def __init__(self):
        # 初始化mask属性，用于存储前向传播时输入值小于等于0的位置信息
        self.mask = None

        # 前向传播方法，输入数据x，返回经过ReLU激活后的数据

    def forward(self, x):
        # 创建一个与x相同形状的布尔数组，其中小于等于0的位置为True，其余为False
        self.mask = (x <= 0)
        # 复制输入数据x
        out = x.copy()
        # 将输入中小于等于0的部分置为0，实现ReLU激活功能
        out[self.mask] = 0
        # 返回激活后的数据
        return out

        # 反向传播方法，根据传入的上游梯度dout，计算并返回对输入的梯度dx

    def backward(self, dout):
        # 将上游梯度中小于等于0对应位置的梯度置为0，因为ReLU在输入小于等于0时导数为0
        dout[self.mask] = 0
        # 直接返回处理后的上游梯度作为对输入的梯度
        dx = dout
        return dx

    # 定义一个Sigmoid激活函数类


class Sigmoid:
    # 初始化方法
    def __init__(self):
        # 初始化输出值out，用于存储前向传播时的输出
        self.out = None

        # 前向传播方法，输入数据x，返回经过Sigmoid激活后的数据

    def forward(self, x):
        # 调用sigmoid函数计算输出（注意：此代码段未提供sigmoid函数的实现）
        out = sigmoid(x)
        # 存储输出值
        self.out = out
        # 返回激活后的数据
        return out

        # 反向传播方法，根据传入的上游梯度dout，计算并返回对输入的梯度dx

    def backward(self, dout):
        # 根据Sigmoid的导数公式计算对输入的梯度dx
        dx = dout * (1.0 - self.out) * self.out
        return dx

    # 定义一个全连接层（Affine层）类


class Affine:
    # 初始化方法，创建Affine对象时调用，需要传入权重W和偏置b
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None  # 存储输入数据x
        self.original_x_shape = None  # 存储输入数据的原始形状
        self.dW = None  # 存储权重W的梯度
        self.db = None  # 存储偏置b的梯度

    # 前向传播方法，输入数据x，返回全连接层的输出数据
    def forward(self, x):
        # 存储输入数据的原始形状并将其改变为2D形状以进行矩阵乘法
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        # 进行全连接层的计算并返回输出数据
        out = np.dot(self.x, self.W) + self.b
        return out

        # 反向传播方法，根据传入的上游梯度dout，计算并返回对输入的梯度dx，同时计算权重W和偏置b的梯度

    def backward(self, dout):
        # 根据全连接层的反向传播公式计算对输入的梯度dx、权重W的梯度dW和偏置b的梯度db
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        # 将输入的梯度dx的形状还原为原始形状并返回
        dx = dx.reshape(*self.original_x_shape)
        return dx

    # 定义一个结合了Softmax与交叉熵损失的类


class SoftmaxWithLoss:
    # 类的初始化方法
    def __init__(self):
        # 初始化损失值为None，将在forward方法中被计算
        self.loss = None
        # softmax函数的输出，初始化为None，将在forward方法中被计算
        self.y = None
        # 监督数据（真实标签），初始化为None，将在forward方法中被赋值
        self.t = None

        # 前向传播方法，计算softmax输出和交叉熵损失

    def forward(self, x, t):
        # 将监督数据（真实标签）保存到类的实例变量中
        self.t = t
        # 通过softmax函数计算输入x的softmax输出，并保存到类的实例变量中
        self.y = softmax(x)  # 注意：此处的softmax函数需要在外部定义或导入
        # 使用softmax输出和监督数据计算交叉熵误差，并保存到类的实例变量中
        self.loss = cross_entropy_error(self.y, self.t)  # 注意：此处的cross_entropy_error函数也需要在外部定义或导入

        # 返回计算的损失值
        return self.loss

        # 反向传播方法，根据传入的上游梯度（dout，默认为1）计算并返回本层的梯度dx

    def backward(self, dout=1):
        # 获取batch的大小
        batch_size = self.t.shape[0]

        # 判断监督数据是否是one-hot编码
        if self.t.size == self.y.size:  # 如果监督数据是one-hot向量
            # 直接计算梯度dx，即softmax输出与监督数据的差，然后除以batch的大小
            dx = (self.y - self.t) / batch_size
        else:
            # 如果监督数据不是one-hot编码，则进行如下处理：
            # 1. 复制softmax的输出到dx
            dx = self.y.copy()
            # 2. 在dx中，将对应监督数据中标签位置的元素减去1
            dx[np.arange(batch_size), self.t] -= 1
            # 3. 将dx除以batch的大小，得到最终的梯度
            dx = dx / batch_size

            # 返回计算的梯度dx
        return dx
