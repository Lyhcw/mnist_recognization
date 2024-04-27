import numpy as np


# 定义一个计算一维数值梯度的函数，输入是一个函数f和一维数组x
def _numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001，h是一个很小的正数，用于计算函数f在x处的导数的近似值
    grad = np.zeros_like(x)
    # 创建一个和x形状相同，数据类型相同，但所有元素都是0的数组grad

    for idx in range(x.size):
        # 遍历x中的每个元素
        tmp_val = x[idx]
        # 临时保存当前位置的原始值
        x[idx] = float(tmp_val) + h
        # 将当前位置的值增加h
        fxh1 = f(x)  # f(x+h)，计算函数在x+h处的值

        x[idx] = tmp_val - h
        # 将当前位置的值减少h
        fxh2 = f(x)  # f(x-h)，计算函数在x-h处的值
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 使用中心差分公式计算当前位置的梯度，并保存到grad数组中

        x[idx] = tmp_val  # 还原值，将当前位置的值还原为原始值

    return grad
    # 返回计算得到的梯度数组


# 定义一个计算二维数值梯度的函数，输入是一个函数f和二维数组X
def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        # 如果输入的是一维数组，则直接调用_numerical_gradient_1d函数计算梯度
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        # 创建一个和X形状相同，数据类型相同，但所有元素都是0的数组grad

        for idx, x in enumerate(X):
            # 遍历X中的每一行x
            grad[idx] = _numerical_gradient_1d(f, x)
            # 对每一行x调用_numerical_gradient_1d函数计算梯度，并保存到grad数组中

        return grad
        # 返回计算得到的梯度数组


# 定义一个计算任意维度数值梯度的函数，输入是一个函数f和数组x
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001，h是一个很小的正数，用于计算函数f在x处的导数的近似值
    grad = np.zeros_like(x)
    # 创建一个和x形状相同，数据类型相同，但所有元素都是0的数组grad

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    # 创建一个NumPy的nditer对象，用于遍历x中的每个元素，支持多维数组
    while not it.finished:
        # 当遍历没有完成时
        idx = it.multi_index
        # 获取当前元素的索引
        tmp_val = x[idx]
        # 临时保存当前位置的原始值
        x[idx] = float(tmp_val) + h
        # 将当前位置的值增加h
        fxh1 = f(x)  # f(x+h)，计算函数在x+h处的值

        x[idx] = tmp_val - h
        # 将当前位置的值减少h
        fxh2 = f(x)  # f(x-h)，计算函数在x-h处的值
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 使用中心差分公式计算当前位置的梯度，并保存到grad数组中

        x[idx] = tmp_val  # 还原值，将当前位置的值还原为原始值
        it.iternext()
        # 移动到下一个元素

    return grad
    # 返回计算得到的梯度数组
