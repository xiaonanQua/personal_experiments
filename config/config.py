"""
保存实验的设置
"""
import os


class Config(object):
    """
    配置实验参数
    """
    def __init__(self):
        self.data_path = '/home/xiaonan/Dataset/cifar-10/'  # 数据集路径
        self.root_path = '/home/xiaonan'  # 项目根目录
        self.save_data_path = self.root_path + '/personal_experiments/save_data/'  # 保存文件路径
        # 数据类型
        self.data_type = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

        # 预处理后的训练、验证数据集、训练模型的保存路径
        self.prepro_train_path = self.save_data_path + 'preprocess_train_files/'
        self.prepro_valid_path = self.save_data_path + 'preprocess_valid_files/'
        self.prepro_test_path = self.save_data_path + 'preprocess_test_files/'
        self.train_model_path = self.save_data_path + 'train_model/'

        # 实验的超参数配置
        self.hyperparam = {'epochs': 10, 'batch_size': 2048, 'keep_prob': 0.7, 'learning_rate': 0.001}

        # 初始化文件夹
        self.init_dir()

    def init_dir(self):
        """
        检测是否存在保存训练、验证数据的文件夹
        :return:
        """
        # 若保存预处理文件的文件夹、训练的模型文件夹不存在则创建相应的文件夹
        if os.path.exists(os.path.join(self.root_path, self.save_data_path)) is False:
            os.mkdir(os.path.join(self.root_path, self.save_data_path))
            print('create file or directory:{}'.format(os.path.join(self.root_path, self.save_data_path)))
        elif os.path.exists(self.prepro_train_path) is False:
            os.mkdir(self.prepro_train_path)
            print('create file or directory:{}'.format(self.prepro_train_path))
        elif os.path.exists(self.prepro_valid_path) is False:
            os.mkdir(self.prepro_valid_path)
            print('create file or directory:{}'.format(self.prepro_valid_path))
        elif os.path.exists(self.prepro_test_path) is False:
            os.mkdir(self.prepro_test_path)
            print('create file or directory:{}'.format(self.prepro_test_path))
        elif os.path.exists(self.train_model_path) is False:
            os.mkdir(self.train_model_path)
            print('create file or directory:{}'.format(self.train_model_path))
