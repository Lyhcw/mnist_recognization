import numpy as np
import pickle
from initial_mnist import load_mnist
from two_layer_net import TwoLayerNet
from common.functions import *
import matplotlib.pyplot as plt


# 加载MNIST数据
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("wight.pkl", 'rb') as f:
        network = pickle.load(f)
    W1 = network['W1']
    b1 = network['b1']
    W2 = network['W2']
    b2 = network['b2']
    return W1, b1, W2, b2


def predicrt(x, W1, b1, W2, b2):
    a1 = np.dot(x, W1) + b1
    z1 = relu(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)
    return y


x, t = get_data()
W1, b1, W2, b2 = init_network()
acc = np.argmax(predicrt(x, W1, b1, W2, b2), axis=1)
print(np.sum(acc == t) / t.shape[0])

for i in range(5):
    random_idx = np.random.randint(len(x))
    slut = predicrt(x[random_idx], W1, b1, W2, b2)
    image = x[random_idx].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    print(t[random_idx], np.argmax(slut))
    plt.show()
