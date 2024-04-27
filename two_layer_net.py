from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    # 初始化函数，设定网络结构和权重等参数
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重W和偏置b，其中权重使用标准差为weight_init_std的正态分布初始化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 初始化网络层，使用OrderedDict保持层的顺序
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        # 初始化输出层，使用SoftmaxWithLoss作为损失函数
        self.lastLayer = SoftmaxWithLoss()

    # 预测函数，给定输入x，返回网络的输出
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # 计算损失函数值，给定输入x和监督标签t
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    # 计算准确率，给定输入x和监督标签t
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

        # 使用数值微分计算梯度，给定输入x和监督标签t

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

        # 计算梯度，给定输入x和监督标签t，使用反向传播算法

    def gradient(self, x, t):
        # forward传播计算损失值（虽然此值在此处并未直接使用，但它是反向传播的基础）
        self.loss(x, t)

        # backward传播开始，初始化dout为1（对应于损失函数对输出层的导数）
        dout = 1
        dout = self.lastLayer.backward(dout)

        # 反转层的顺序，从输出层开始反向传播
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 收集各层的梯度信息
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
