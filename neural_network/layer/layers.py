"""
实现层一些方法
"""
import tensorflow as tf


class Layers(object):
    def __init__(self):
        # 定义初始化变量时高斯分布的参数：均值、标准差
        self.random_mean = 0
        self.random_stddev = 0.01

    def conv2d_and_activation(self, features, filter_width, filter_height, filters_count,
                              stride_x, stride_y, padding='VALID', init_biases=False, name='conv'):
        """
        卷积运算和激活函数激活
        :param input: 输入的特征数据
        :param filter_width: 卷积核的宽度
        :param filter_height: 卷积核的高度
        :param filters_count: 卷积核数量
        :param stride_x: 步长的x轴
        :param stride_y: 步长的y轴
        :param padding: 零填充方式
        :param init_biases: boolean，True：对偏置值进行1初始化，False：对偏置值进行0初始化
        :param name:名称
        :return:进行卷积运算和激活函数后的特征数据
        """
        with tf.name_scope(name):
            # 通过输入数据获取通道数量
            input_channels = features.get_shape()[-1].value
            # 初始化偏置值biases
            if init_biases:
                biases = tf.Variable(tf.ones(shape=[filters_count], dtype=tf.float32), name='biases')
            else:
                biases = tf.Variable(tf.zeros(shape=[filters_count], dtype=tf.float32), name='biases')
            # 根据传入的卷积核参数，定义卷积核变量，使用高斯分布进行初始化
            filters = tf.Variable(
                self.random_value(shape=[filter_width, filter_height, input_channels, filters_count]),
                name='filters')
            # 执行卷积运算,添加偏置值，使用ReLU激活函数进行激活
            convs = tf.nn.conv2d(features, filters, strides=[1, stride_y, stride_x, 1], padding=padding, name='convs')
            preactivations = tf.nn.bias_add(convs, biases, name='preactivations')
            activations = tf.nn.relu(preactivations, name='activations')

            # 将变量值(卷积核参数，偏置值参数)，未激活的卷积数据和激活后的卷积数据添加到日志中
            with tf.name_scope('filter_summaries'):
                self.variable_summaries(filters)
            with tf.name_scope('biases_summaries'):
                self.variable_summaries(biases)
            with tf.name_scope('preactivation_histogram'):
                tf.compat.v1.summary.histogram('preactivations', preactivations)
            with tf.name_scope('activation_histogram'):
                tf.compat.v1.summary.histogram('activations', activations)

            return activations

    def max_pool(self, features, filters_width, filters_height, stride_x, stride_y, padding='VALID', name='pool'):
        """
        最大汇聚层
        :param features: 输入的特征
        :param filters_width: 核的宽度
        :param filters_height: 核的高度
        :param stride_x: 步长x
        :param stride_y: 步长y
        :param padding: 零填充方式
        :param name: 名称
        :return: 汇聚后的特征
        """
        with tf.name_scope(name):
            pool = tf.nn.max_pool2d(features, ksize=[1, filters_width, filters_height, 1],
                                    strides=[1, stride_y, stride_x, 1], padding=padding, name='max_pool')
        return pool

    def fully_connected(self, features, input_num, output_num, relu=True,
                         init_biases=False, name='fully_connected'):
        """
        全连接层
        :param features: 特征数据
        :param input_num: 输入的神经元数量
        :param output_num: 输出的神经元数量
        :param relu: boolean：是否使用ReLU激活函数进行激活
        :param init_biases: boolean：偏置值初始化
        :param name: 名称
        :return:
        """
        with tf.name_scope(name):
            # 初始化权重、偏置值参数
            weights = tf.Variable(self.random_value(shape=[input_num, output_num]), name='weights')
            if init_biases:
                biases = tf.Variable(tf.ones(shape=[output_num], dtype=tf.float32), name='biases')
            else:
                biases = tf.Variable(tf.zeros(shape=[output_num], dtype=tf.float32), name='biases')
            # 线性值
            preactivations = tf.nn.bias_add(tf.matmul(features, weights), biases, name='preactivations')
            # 使用激活函数进行激活
            if relu:
                activations = tf.nn.relu(preactivations, name='activations')

            # 将参数变量、线性值、激活值添加到日志中，用于tensorboard查看
            with tf.name_scope('weight_summaries'):
                self.variable_summaries(weights)
            with tf.name_scope('biases_summaries'):
                self.variable_summaries(biases)
            with tf.name_scope('preactivations_histogram'):
                tf.compat.v1.summary.histogram('preactivations', preactivations)

            if relu:
                with tf.name_scope('activation_histogram'):
                    tf.compat.v1.summary.histogram('activations', activations)
                return activations
            else:
                return preactivations

    def dropout(self, features, keep_prob, name='dropout'):
        """
        对神经元进行一定比例丢弃
        :param features: 特征数据
        :param: keep_prob: 保存的概率值
        :param name: 名称
        :return: 丢弃后的特征
        """
        with tf.name_scope(name):
            return tf.nn.dropout(features, keep_prob=keep_prob, name='dropout')

    def local_reponse_normalization(self, features, name='lrn'):
        """
        局部响应归一化
        :param features: 特征数据
        :param name: 名称
        :return:
        """
        with tf.name_scope(name):
            lrn = tf.nn.local_response_normalization(features, depth_radius=2, alpha=10**-4,
                                                      beta=0.75, name='local_response_normalization')
        return lrn

    def random_value(self, shape, ):
        """
        获取随机值，且随机值满足高斯分布
        :param shape: 数据的形状
        :return: 随机值
        """
        return tf.random.normal(shape=shape,
                                mean=self.random_mean,
                                stddev=self.random_stddev,
                                dtype=tf.float32)

    def variable_summaries(self, var):
        """
        :param var: 变量
        :return:
        """
        mean = tf.reduce_mean(var)  # 平均值
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))  # 标准差
        # 输出包含单个标量值的摘要协议缓冲区,用于tensorboard显示数据
        tf.compat.v1.summary.scalar('min', tf.reduce_min(var))  # 变量中最小值
        tf.compat.v1.summary.scalar('max', tf.reduce_max(var))  # 变量中最大值
        tf.compat.v1.summary.scalar('mean', mean)  # --平均值
        tf.compat.v1.summary.scalar('stddev', stddev)  # --标准差
        tf.compat.v1.summary.histogram('histogram', var)  # 使用直方图可视化变量数据