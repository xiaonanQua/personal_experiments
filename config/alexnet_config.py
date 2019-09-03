from config.config import Config
import logging as log


class AlexNetConf(Config):
    def __init__(self):
        # 继承父类构造函数
        super(AlexNetConf, self).__init__()
        # 图像宽度、高度、通道
        self.image_width = 70
        self.image_height = 70
        self.image_channels = 3
        # 类别数量
        self.num_classes = 10
        # 实验的超参数配置
        self.epochs = 100
        self.batch_size = 128
        self.learning_rate = 0.001  # 原始是0.01
        self.momentum = 0.9
        self.keep_prob = 0.5
        # 配置日志文件
        log.basicConfig(filename=self.log_dir+'alexnet.log', level=log.INFO)
        self.log = log


