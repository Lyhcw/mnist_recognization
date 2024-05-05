# 环境安装

---

> - python>=3.8
> - pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
> - pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

# 怎么运行

---

- 没有下载`data`训练集和已经训练好的`wight.pkl`权重数据
  - 首先，运行` initial_mnist.py `初始化mnist训练集主程序，从网上下载mnist训练集
  - 其次，运行`train.py` 神经网络权重和偏置的训练主程序，训练出两层神经网络的权重
  - 最后，运行`NbRecog_two.py` 数字识别主程序
- 下载了`data`训练集和已经训练好的`wight.pkl`权重数据
  - 直接运行`NbRecog_two.py` 	数字识别主程序

# 目录说明

---

​	`NbRecog_two.py` 	数字识别主程序

​	`train.py` 	神经网络权重和偏置的训练主程序

​	` initial_mnist.py `	初始化mnist训练集主程序

​	`two_layer_net.py`	设定两层神将网络的框架

​	`wight.pkl`	通过`train.py`训练出的权重和偏置数据

​	`datas`	` initial_mnist.py `下载的mnist训练集的保存位置

​	`common\function.py`	两层神经网络中会用到的激活函数和损失函数的实现

​	`common\gradient.py`	梯度计算的实现

​	`common\layers.py`	一些激活函数，损失函数和映射层的正向传播和反向传播的实现

