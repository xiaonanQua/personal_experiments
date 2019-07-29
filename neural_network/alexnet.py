"""
AlexNet网络结构
"""

import tensorflow as tf


class AlexNet(object):
    def __init__(self):
        """
        初始化输入数据、标签
        """
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name='input_x')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='label_y')
        self._logging = []  # 用于保存参数等数据,私有属性

    def alex_net(self):
        """
        AlexNet网络结构
        :return:
        """
        # 在命名空间下conv1实现第一个卷积层
        with tf.name_scope('conv_1'):
            conv_filter