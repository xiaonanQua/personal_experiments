"""
保存实验的设置
"""
import os


class Config(object):
    """
    配置实验参数
    """
    def __init__(self):
        # 数据集根目录、项目根目录
        self.root_dataset = '/home/xiaonan/Dataset/'
        self.root_project = '/home/xiaonan/python_project/tensorflow_learn/'

        # cifar-10数据集目录、文件名称
        self.cifar_10_dir = self.root_dataset + 'cifar-10/'
        self.cifar_file_name = {'meta': 'batches.meta',
                                'train': ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
                                'test': 'test_batch'}

        # svhn数据集目录、文件名称
        self.svhn_dir = self.root_dataset + 'svhn/'
        self.svhn_file_name = ['train_32.mat', 'test_32.mat', 'extra_32.mat']

        # mnist数据集目录,文件名称
        self.mnist_dir = self.root_dataset + 'mnist/'
        self.mnist_file_name = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                                't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

        # 数据保存根目录、模型保存目录、日志文件保存目录
        self.data_save_dir = self.root_project + 'data_save/'
        self.model_dir = self.data_save_dir + 'model'
        self.log_dir = self.data_save_dir + 'log'

        # 初始化文件夹
        self._init()

    def _init(self):
        # 若文件夹不存在，则创建
        if os.path.exists(self.data_save_dir) is False:
            os.mkdir(self.data_save_dir)
        elif os.path.exists(self.model_dir) is False:
            os.mkdir(self.model_dir)
        elif not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)


if __name__ == '__main__':
    cfg = Config()