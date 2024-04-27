import urllib.request  # 引入urllib.request模块
import os.path  # 导入os.path模块，用于文件路径操作
import gzip  # 导入gzip模块，用于处理gzip压缩文件
import pickle  # 导入pickle模块，用于对象序列化和反序列化
import os  # 导入os模块，用于与操作系统交互
import numpy as np  # 导入numpy模块，用于数组操作

url_base = 'http://yann.lecun.com/exdb/mnist/'  # 定义基础URL
key_file = {  # 定义字典，存储MNIST数据集的文件名
    'train_img': 'train-images-idx3-ubyte.gz',  # 训练集图像文件名
    'train_label': 'train-labels-idx1-ubyte.gz',  # 训练集标签文件名
    'test_img': 't10k-images-idx3-ubyte.gz',  # 测试集图像文件名
    'test_label': 't10k-labels-idx1-ubyte.gz'  # 测试集标签文件名
}
dataset_dir = os.path.dirname(os.path.abspath(__file__)) + '\\datas'  # 获取当前脚本的目录
save_file = dataset_dir + "/mnist.pkl"  # 定义保存数据集的pickle文件路径
# 定义训练集和测试集的数量
train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)  # 定义图像的维度
img_size = 784  # 定义图像的像素数


# 定义下载文件的函数
def _download(file_name):
    file_path = dataset_dir + "/" + file_name  # 定义文件完整路径
    if os.path.exists(file_path):  # 如果文件已存在，则直接返回
        return
    print("Downloading " + file_name + " ... ")  # 打印下载信息
    # 使用urllib.request从网络下载文件
    urllib.request.urlretrieve(url_base + file_name, file_path)
    # 打印下载完成信息
    print("Done")


# 定义下载MNIST数据集的函数
def download_mnist():
    # 遍历字典中所有文件，逐个下载
    for v in key_file.values():
        _download(v)


# 定义加载标签文件的函数
def _load_label(file_name):
    # 定义文件完整路径
    file_path = dataset_dir + "/" + file_name
    # 打印转换信息
    print("Converting " + file_name + " to NumPy Array ...")
    # 使用gzip模块打开文件，并读取数据
    with gzip.open(file_path, 'rb') as f:
        # 从文件缓冲区中读取数据，并转换为numpy数组，从第8个字节开始读取（跳过文件头）
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
        # 打印转换完成信息
    print("Done")
    # 返回转换后的标签数组
    return labels


# 定义加载图像文件的函数
def _load_img(file_name):
    # 定义文件完整路径
    file_path = dataset_dir + "/" + file_name
    # 打印转换信息
    print("Converting " + file_name + " to NumPy Array ...")
    # 使用gzip模块打开文件，并读取数据
    with gzip.open(file_path, 'rb') as f:
        # 从文件缓冲区中读取数据，并转换为numpy数组，从第16个字节开始读取（跳过文件头）
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # 将一维数组重塑为二维数组，其中每行代表一个图像
    data = data.reshape(-1, img_size)
    # 打印转换完成信息
    print("Done")
    # 返回转换后的图像数组
    return data


# 定义将MNIST数据集转换为numpy数组的函数
def _convert_numpy():
    # 创建一个空字典，用于存储数据集
    dataset = {}
    # 调用_load_img函数加载训练集图像数据，并存储到dataset字典中
    dataset['train_img'] = _load_img(key_file['train_img'])
    # 调用_load_label函数加载训练集标签数据，并存储到dataset字典中
    dataset['train_label'] = _load_label(key_file['train_label'])
    # 调用_load_img函数加载测试集图像数据，并存储到dataset字典中
    dataset['test_img'] = _load_img(key_file['test_img'])
    # 调用_load_label函数加载测试集标签数据，并存储到dataset字典中
    dataset['test_label'] = _load_label(key_file['test_label'])
    # 返回存储数据集信息的字典
    return dataset


# 定义初始化MNIST数据集的函数
def init_mnist():
    # 调用download_mnist函数下载MNIST数据集
    download_mnist()
    # 调用_convert_numpy函数将MNIST数据集转换为numpy数组
    dataset = _convert_numpy()
    # 打印正在创建pickle文件的信息
    print("Creating pickle file ...")
    # 使用pickle模块将数据集字典写入到文件中
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    # 打印pickle文件创建完成的信息
    print("Done!")


# 定义将标签转换为one-hot编码的函数
def _change_one_hot_label(X):
    # 创建一个全零矩阵，行数为X的长度，列数为10
    T = np.zeros((X.size, 10))
    # 遍历矩阵的每一行
    for idx, row in enumerate(T):
        # 将对应X[idx]索引位置的元素设为1，完成one-hot编码
        row[X[idx]] = 1
        # 返回one-hot编码后的矩阵
    return T


# 定义加载MNIST数据集的函数
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """
    读入MNIST数据集

    参数:
    normalize : 是否将图像的像素值正规化为0.0~1.0
    one_hot_label : 是否将标签转换为one-hot编码
    flatten : 是否将图像数据展开为一维数组

    返回:
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    # 如果pickle文件不存在
    if not os.path.exists(save_file):
        # 调用init_mnist函数初始化MNIST数据集
        init_mnist()

        # 使用pickle模块从文件中加载数据集字典
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    # 如果normalize为True，则将图像数据正规化到0.0~1.0
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    # 如果one_hot_label为True，则将标签转换为one-hot编码
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    # 如果flatten为False，则将图像数据重新reshape为四维数组
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    # 返回训练集和测试集的图像与标签数据
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])
