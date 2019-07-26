"""
预处理数据，包含正则化、one-hot编码
"""
import numpy as np
import pickle as pk
from deep_neural_network.convolution_neural_network.data_load import loaddataset as cifar


class Preprocess(object):
    def __init__(self):
        self.data = None  # 处理的数据
        self.save_data_path = None  # 保存预处理后的文件路径
        self.data_path = None

    def normalize(self):
        """
        使用min-max归一化技术将原始图像数据转化0-1的范围内
        self.data:输入的图像数据，(32,32,3)形状的numpy数组形式
        :return: 处理后的数据
        """
        # 获得3-D数组中的最小值和最大值
        min_value = np.min(self.data)
        max_value = np.max(self.data)
        # 进行归一化的处理,3-D数组中的每个元素都参与减、除操作
        data = (self.data-min_value)/(max_value-min_value)
        return data

    def one_hot_encode(self):
        """
        使用one-hot编码将每个图像的类别标签变成one-hot向量
        :return: 编码后的标签矩阵，（标签数量，类别数（one-hot向量））
        """
        encoded_label = np.zeros((len(self.data), 10))  # 初始化相应数量的one-hot向量（每一行全为0）
        for index, value in enumerate(self.data):
            encoded_label[index][value] = 1  # 设置每个类的one-hot向量的值（特定类别位置为1）
        return encoded_label

    def preprocess_and_save_data(self):
        """

        :return:
        """
