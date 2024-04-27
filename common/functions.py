import numpy as np  # 导入NumPy库，一个强大的N维数组对象、复杂的函数、用于整合C/C++和Fortran代码的工具等


# 定义一个恒等函数，输入什么就输出什么
def identity_function(x):
    return x


# 定义一个阶跃函数，输入大于0返回1，否则返回0
def step_function(x):
    return np.array(x > 0, dtype=np.int)


# 定义一个Sigmoid函数，用于将输入映射到(0,1)区间，常用于二分类问题的输出层激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义Sigmoid函数的导数，用于反向传播时计算梯度
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


# 定义一个ReLU函数，输入小于0时返回0，否则返回输入值，常用于神经网络的激活函数
def relu(x):
    return np.maximum(0, x)


# 定义ReLU函数的导数，用于反向传播时计算梯度
def relu_grad(x):
    grad = np.zeros(x)  # 创建一个和x形状相同，元素全为0的数组
    grad[x >= 0] = 1  # 将x中大于等于0的元素对应的grad位置设为1
    return grad


# 定义一个Softmax函数，将输入映射为概率分布，常用于多分类问题的输出层激活函数
def softmax(x):
    if x.ndim == 2:  # 如果输入是二维数组（例如：一个batch的数据）
        x = x.T  # 转置，使得每一列代表一个样本
        x = x - np.max(x, axis=0)  # 减去每列的最大值，防止指数爆炸
        y = np.exp(x) / np.sum(np.exp(x), axis=0)  # 计算Softmax值
        return y.T  # 转置回来，使得每一行代表一个样本的概率分布
    x = x - np.max(x)  # 对于一维输入，同样需要防止指数爆炸
    return np.exp(x) / np.sum(np.exp(x))  # 计算Softmax值


# 定义一个均方误差函数，常用于回归问题的损失函数
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 定义一个交叉熵误差函数，常用于分类问题的损失函数
def cross_entropy_error(y, t):
    if y.ndim == 1:  # 如果输入是一维数组，则将其转换为二维数组，以便于后续处理
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:  # 如果监督数据是标签形式而非one-hot编码，则转换为one-hot编码
        t = t.argmax(axis=1)
    batch_size = y.shape[0]  # 获取batch大小
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  # 计算交叉熵误差，并添加一个小值防止log(0)的情况


# 定义一个结合了Softmax函数和交叉熵误差的损失函数，常用于多分类问题的输出层损失函数
def softmax_loss(X, t):
    y = softmax(X)  # 先通过Softmax函数得到概率分布
    return cross_entropy_error(y, t)  # 再计算交叉熵误差作为损失值
