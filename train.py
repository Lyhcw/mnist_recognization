import pickle
# 导入numpy库，并简写为np
import numpy as np
# 从initial_mnist模块中导入load_mnist函数，用于加载MNIST数据集
from initial_mnist import load_mnist
# 从two_layer_net模块中导入TwoLayerNet类，这是一个两层神经网络模型
from two_layer_net import TwoLayerNet

# 使用load_mnist函数读入MNIST数据集，并进行归一化和one-hot编码
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 创建一个两层神经网络实例，指定输入、隐藏层和输出层的大小
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 设置训练迭代次数
iters_num = 6000

# 获取训练数据的数量
train_size = x_train.shape[0]

# 设置每个batch的大小
batch_size = 100

# 设置学习率
learning_rate = 0.1

# 初始化用于记录训练过程中损失和准确率的列表
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 计算每个epoch的迭代次数
iter_per_epoch = max(train_size / batch_size, 1)

# 开始训练循环
for i in range(iters_num):
    # 从训练数据中随机选择一个batch的数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度（这里可以选择使用数值梯度或解析梯度，但解析梯度效率更高）
    # grad = network.numerical_gradient(x_batch, t_batch)  # 数值梯度，较慢，主要用于验证解析梯度的正确性
    grad = network.gradient(x_batch, t_batch)  # 解析梯度，较快

    # 更新网络参数（权重和偏置）
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 计算并记录当前batch的损失值
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 每个epoch结束后，计算并记录训练集和测试集的准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("Epoch {}, Train accuracy: {}, Test accuracy: {}".format(i // iter_per_epoch, train_acc, test_acc))

file_path = 'wight.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(network.params, f)
