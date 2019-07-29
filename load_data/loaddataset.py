"""
实现数据集的加载
"""
import pickle as pk
import os
import numpy as np
import matplotlib.pyplot as plt
from config import config as cfg


class DataSetOp(object):
    """
    操作数据集类
    """
    def __init__(self, data_path=None):
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

    def load_cifar_10_single_data(self, data_type):
        """
        获取单个批次文件的训练数据
        :param data_type: 批次文件名称
        :return: 单个批次的训练数据、标签数据
        """
        # 定义训练数据、标签
        train_data = []
        label_data = []

        # 文件路径
        file_path = os.path.join(self.data_path, data_type)
        # 使用上下文环境打开文件
        with open(file_path, mode='rb') as file:
            batch_file = pk.load(file, encoding='latin1')
        # 将获取的训练数据进行转化
        train_data = batch_file['data'].reshape(len(batch_file['data']), 3, 32, 32).transpose(0, 3, 2, 1)
        label_data = batch_file['labels']

        return train_data, label_data

    def load_preprocess_data(self, data_path, data_name, batch_size=None, index=None):
        """
        加载预处理后的数据，包括训练集、验证集、测试集
        :param data_path: 数据路径
        :param data_name: 数据文件名称
        :param batch_size: 随机批次大小，每次进行数据的加载，即随机选取batch_size数量的样本
        :param index: 索引位置
        :return:
        """
        # 定义训练数据、标签
        train_data = []
        label_data = []
        # 加载的文件路径
        file_path = os.path.join(data_path, data_name)
        # 判断文件是否存在
        if os.path.isfile(file_path) is False:
            print('{}:不存在'.format(file_path))
            return None

        # 打开需要加载的文件
        with open(file_path, 'rb') as file:
            # 编码形式是'latin1',使得读出的数据为data形式，用于numpy使用
            batch = pk.load(file, encoding='latin1')

        if batch_size is not None:  # 处理训练数据
            # 定义索引的low和high,[low, high]。若超出批次数据总长度，则设置high为最大值
            low = index*batch_size
            high = (index+1)*batch_size
            if high >= len(batch[0]):
                high = len(batch[0])
            index_list = np.array(range(low, high))
            np.random.shuffle(index_list)  # 对读取数据的索引进行洗牌
            print('fetch data min and max index :{}, {}'.format(np.min(index_list), np.max(index_list)))
            # 抽取batch_size数量的训练数据并进行封装
            for i in index_list:
                #print(batch[0][i])
                # 将数据附加在训练数据中
                train_data.append(batch[0][i])
                label_data.append(batch[1][i])
            # 将list数据转化成数组
            train_data = np.array(train_data)
            label_data = np.array(label_data)

            return train_data, label_data

        return batch


if __name__ == '__main__':

    cfiar = DataSetOp(data_path=cfg.Config().data_path)
    # train_data, train_label, test_data, label_name = cfiar.load_cifar_10_data()
    # print(train_data[0].shape)
    # print(train_label[0])
    # print(test_data.shape)
    # print(label_name)
    # cfg = cfg.Config()
    # train_data, train_label = cfiar.load_preprocess_data(cfg.prepro_train_path, 'preprocess_batch_1.p', batch_size=1)
    # cfiar_data, cfiar_label = cfiar.load_cifar_10_single_data('data_batch_1')
    # print(train_data[0])
    # print(cfiar_data[0])
    # #plt.imshow(cfiar_data[0])
    # plt.imshow(train_data[0])
    # plt.show()
    index = set()
    while len(index) != 128:
        index.add(np.random.shuffle())


