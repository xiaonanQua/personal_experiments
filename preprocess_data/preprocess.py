"""
预处理数据，包含正则化、one-hot编码
"""
import numpy as np
import pickle as pk
import os
from config import config as cfg
from load_data import loaddataset as cifar
# 调用相应的类
cfg = cfg.Config()
cifar = cifar.DataSetOp(cfg.data_path)


class Preprocess(object):
    def __init__(self):
        # 保存训练集、验证集、测试集预处理后的文件路径
        self.save_train_path = cfg.prepro_train_path
        self.save_valida_path = cfg.prepro_valid_path
        self.save_test_path = cfg.prepro_test_path
        # 数据集路径
        self.data_path = cfg.data_path

    def _normalize(self, data):
        """
        使用min-max归一化技术将原始图像数据转化0-1的范围内
        :param data:输入的图像数据，(32,32,3)形状的numpy数组形式
        :return: 处理后的数据
        """
        # 获得3-D数组中的最小值和最大值
        min_value = np.min(data)
        max_value = np.max(data)
        # 进行归一化的处理,3-D数组中的每个元素都参与减、除操作
        data = (data-min_value)/(max_value-min_value)
        return data

    def _one_hot_encode(self, data):
        """
        使用one-hot编码将每个图像的类别标签变成one-hot向量
        :param data: 数据
        :return: 编码后的标签矩阵，（标签数量，类别数（one-hot向量））
        """
        try:
            encoded_label = np.zeros((len(data), 10))  # 初始化相应数量的one-hot向量（每一行全为0）
            for index, value in enumerate(data):
                encoded_label[index][value] = 1  # 设置每个类的one-hot向量的值（特定类别位置为1）
        except TypeError:  # 捕捉数据为None的时候
            print('标签是：{}'.format(data))
            return None
        return encoded_label

    def _preprocess_and_save_data(self, features_data, save_file_path, labels_data=None):
        """
        归一化训练特征数据、对标签数据进行one-hot编码，并将预处理后的数据进行保存。
        :param features_data: 训练的特征数据
        :param labels_data: 标签数据
        :param save_file_path: 保存文件路径
        :return:
        """
        # 进行归一化和one-hot编码
        features_data = self._normalize(features_data)
        labels_data = self._one_hot_encode(labels_data)
        print(save_file_path)
        # 使用pickle进行数据的保存
        if os.path.isfile(save_file_path) is False:
            # 打开不存在的文件
            with open(save_file_path, mode='wb') as save_file:
                if labels_data is None:  # 若标签是None，则保存的是测试集
                    pk.dump(features_data, save_file)
                else:  # 保存训练、验证集
                    pk.dump((features_data, labels_data), save_file)

    def preprocess_and_save_data(self):
        """
        预处理训练数据并保存数据
        :return:
        """
        # 定义数据批次、验证数据、标签
        n_batches = 5
        valid_data = []
        valid_labels = []

        # 读取五个批次的训练数据的图像数据、标签，进行归一化和one-hot编码
        for i in range(0, n_batches):
            train_data, train_labels = cifar.load_cifar_10_single_data(data_type=cfg.data_type[i])
            # 划分出训练数据中的百分之十用作验证集，计算出相应的长度
            valid_len = int(len(train_data)*0.1)
            # 单个批次中预处理90%的数据
            # 分别进行特征数据的归一化、标签的one-hot向量的编码，保存为一个新文件
            self._preprocess_and_save_data(train_data[:-valid_len],
                                           self.save_train_path + 'preprocess_batch_' + str(i + 1) + '.p',
                                           train_labels[:-valid_len])
            # 不像训练集，验证集需要从每个训练集批次中选取10%一起组合成验证集
            valid_data.extend(train_data[-valid_len:])
            valid_labels.extend(train_labels[-valid_len:])
            print(np.array(valid_data).shape, np.array(valid_labels).shape)

            # 对验证集进行预处理并保存
            self._preprocess_and_save_data(np.array(valid_data),
                                           self.save_valida_path+'preprocess_validation.p',
                                           np.array(valid_labels),)

        # 获取测试集数据
        test_data, _ = cifar.load_cifar_10_single_data(data_type=cfg.data_type[5])
        print(test_data.shape)
        # 对测试集数据进行预处理并保存
        self._preprocess_and_save_data(test_data,
                                       self.save_test_path+'preprocess_test.p')


if __name__ == '__main__':
    preprocess_data = Preprocess()
    preprocess_data.preprocess_and_save_data()

