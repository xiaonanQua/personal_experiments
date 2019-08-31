"""
加载cifar-10数据集
"""
import pickle as pk
import numpy as np
import os
from skimage.transform import resize
from config import config
cfg = config.Config()


class DataSet(object):
    def __init__(self):
        # 定义根目录、文件类型、文件存在的路径
        self.root = cfg.root_path
        self.file_type = cfg.data_type
        self.file_path = cfg.data_path

    def _unpickle(self, file):
        """
        读取pickle封装的文件
        :param file: 文件
        :return: 读取的文件
        """
        with open(file, mode='rb') as file:
            dict = pk.load(file, encoding='latin1')
        return dict

    def load_cfiar_10(self, image_width, image_height):
        """
        加载cifar-10数据集
        :param image_width: 图像宽度
        :param image_height: 图像高度
        :return:
        """
        # 读取cfiar-10数据集批次数据
        train_batch_1 = self._unpickle(os.path.join(self.file_path, self.file_type[0]))
        train_batch_2 = self._unpickle(os.path.join(self.file_path, self.file_type[1]))
        train_batch_3 = self._unpickle(os.path.join(self.file_path, self.file_type[2]))
        train_batch_4 = self._unpickle(os.path.join(self.file_path, self.file_type[3]))
        train_batch_5 = self._unpickle(os.path.join(self.file_path, self.file_type[4]))
        test_batch = self._unpickle(os.path.join(self.file_path, self.file_type[5]))

        # 数据类别
        classes = self._unpickle(os.path.join(self.file_path, 'batches.meta'))['label_names']
        # 整体的训练样本
        total_train_samples = len(train_batch_1['labels']) + \
        len(train_batch_2['labels'])+len(train_batch_3['labels']) + \
        len(train_batch_4['labels'])+len(train_batch_5['labels'])
        # 初始化训练数据和标签
        x_train = np.zeros(shape=[total_train_samples, image_width, image_height, 3], dtype=np.uint8)
        y_train = np.zeros(shape=[total_train_samples, len(classes)], dtype=np.float32)
        # 将训练批次集成到数组中
        train_batches = [train_batch_1, train_batch_2, train_batch_3, train_batch_4, train_batch_5]

        index = 0
        for train_batch in train_batches:
            for i in range(len(train_batch['labels'])):
                # 图像数据的行向量形式重塑成(3,32,32)的形状再转置成（32,32,3）
                image = train_batch['data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
                label = train_batch['labels'][i]
                # 对获取的图像数据重新设置高度、宽度，标签转化成one-hot
                x = resize(image, output_shape=(image_height, image_width))
                y = np.zeros(shape=[len(classes)], dtype=np.int)
                y[label] = 1

                x_train[index+i] = x
                y_train[index+i] = y

            index += len(train_batch['labels'])
        # 加载测试样本
        # 获得整体测试样本数量、初始化测试数据、标签
        total_test_samples = len(test_batch['labels'])
        x_test = np.zeros(shape=[total_test_samples, image_width, image_height, 3], dtype=np.uint8)
        y_test = np.zeros(shape=[total_test_samples, len(classes)], dtype=np.float32)

        # 迭代测试集重塑图像的形状、大小
        for i in range(test_batch['labels']):
            image = test_batch['data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
            label = test_batch['labels'][i]

            x = resize(image, (image_height, image_width))
            y = np.zeros(shape=[len(classes)], dtype=np.int)
            y[label] = 1

            x_test[i] = x
            y_test[i] = y
        return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    cifar = DataSet()
    cifar.load_cfiar_10()