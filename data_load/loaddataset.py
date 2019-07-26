"""
实现数据集的加载
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os

class DataSetOp(object):
    """
    操作数据集类
    """
    def __init__(self, data_path):
        self.data_path = data_path  # 数据存储路径

    def load_cifar_10_data(self):
        """
        装载CIFAR-10数据集
        :return: 训练特征、标签、标签名称
        """
        # 定义数据类型名称数组
        data_type = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                       'data_batch_4', 'data_batch_5', 'test_batch', 'batches.meta']
        # 定义训练数据集、标签,测试集，标签名称
        train_data = []
        train_label = []
        test_data = []
        label_name = []
        # 打开数据集文件
        for i in range(len(data_type)):
            file_path = os.path.join(self.data_path, data_type[i])  # 数据文件路径
            # 使用上下文环境打开文件
            with open(file_path, mode='rb') as file:
                # 使用‘latin1’编码格式进行编码
                batch = pk.load(file, encoding='latin1')
            if i < 5:  # 处理训练数据批次
                # 将特征行向量重塑成（10000，3，32，32）再转置成（10000,32,32,3）
                train_batch = batch['data'].reshape(len(batch['data']), 3, 32, 32).transpose(0, 3, 2, 1)
                label_batch = batch['labels']
                # 保存所有批次的特征、标签
                train_data.append(train_batch)
                train_label.append(label_batch)
            elif i == 5: # 处理测试数据批次
                test_data = batch['data'].reshape(len(batch['data']), 3, 32, 32).transpose(0, 3, 2, 1)
            else:
                label_name = batch['label_names']
        return train_data, train_label, test_data, label_name


if __name__ == '__main__':
    cfiar = DataSetOp(data_path='/home/xiaonan/Dataset/cifar-10/')
    train_data, train_label, test_data, label_name = cfiar.load_cifar_10_data()
    print(train_data[0].shape)
    print(train_label[0])
    print(test_data.shape)
    print(label_name)


