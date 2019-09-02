"""
AlexNet网络结构
"""

import tensorflow as tf
from neural_network.layer import layers
# 实例化layers对象
layer = layers.Layers()


class AlexNet(object):
    def __init__(self, input_width=227, input_height=227, input_channels=3,
                 num_classes=10, learning_rate=0.01, momentum=0.9, keep_prob=0.5):
        """
        初始化参数
        :param input_width: 输入图像数据的宽度
        :param input_height: 输入图像数据的高度
        :param input_channels: 输入图像数据的通道
        :param num_classes: 识别(recognize)的类别数量
        :param learning_rate: 学习率，优化算法中梯度下降速率
        :param momentum: 动量，提高梯度优化速度
        :param keep_prob: Dropout中的参数，保留一定比例的神经元
        """
        # 初始化参数
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.keep_prob = keep_prob

        # 定义占位符:输入数据、标签、keep_prob
        with tf.name_scope('input'):
            self.x = tf.compat.v1.placeholder(dtype=tf.float32,
                                    shape=[None, self.input_width, self.input_height, self.input_channels],
                                    name='x')
        with tf.name_scope('label'):
            self.y = tf.compat.v1.placeholder(dtype=tf.float32,
                                    shape=[None, self.num_classes],
                                    name='y')
        with tf.name_scope('dropout'):
            self.dropout_keep_prob = tf.compat.v1.placeholder(dtype=tf.float32,
                                                    shape=(),
                                                    name='dropout_keep_prob')

        # 卷积层1
        # input：227*227*3-->卷积后(filter:[11,11,3,96],strides=4,padding=Valid)：55*55*96(55=(227-11+2*0)/4+1)-->
        # (ReLU):55*55*96-->(Local Response Normalization):55*55*96-->
        # 最大汇聚(ksize=3*3,strides=2,padding=Valid):27*27*96-->output:27*27*96(27=(55-3+2*0)/2+1)
        with tf.name_scope('layer_1'):
            layer_1_conv = layer.conv2d_and_activation(self.x, filter_width=11, filter_height=11,
                                                     filters_count=96, stride_x=4, stride_y=4,
                                                     padding='VALID', init_biases=False)
            layer_1_lrn = layer.local_reponse_normalization(features=layer_1_conv)
            layer_1_pool = layer.max_pool(features=layer_1_lrn, filters_width=3, filters_height=3,
                                         padding='VALID', stride_x=2, stride_y=2)

        # 卷积层2
        # input：27*27*96-->卷积后(filter:[5,5,96,256],strides=1,padding=SAME)：27*27*256（27=(27-5+2*(5-1)/2)/1+1）-->
        # (ReLU):27*27*256-->(Local Response Normalization):27*27*256-->
        # 最大汇聚(ksize=3*3,strides=2,padding=Valid):13*13*256-->output:13*13*256(13=(27-3+2*0)/2+1)
        with tf.name_scope('layer_2'):
            layer_2_conv = layer.conv2d_and_activation(layer_1_pool, filter_width=5, filter_height=5,
                                                     filters_count=256, stride_x=1, stride_y=1,
                                                     padding='SAME', init_biases=True)
            layer_2_lrn = layer.local_reponse_normalization(features=layer_2_conv)
            layer_2_pool = layer.max_pool(features=layer_2_lrn, filters_width=3, filters_height=3,
                                         padding='VALID', stride_x=2, stride_y=2)

        # 卷积层3
        # input：13*13*256-->卷积后(filter:[3,3,256,384],strides=1,padding=SAME)：13*13*384（13=(13-3+2*(3-1)/2)/1+1）-->
        # (ReLU):13*13*384
        with tf.name_scope('layer_3'):
            layer_3_conv = layer.conv2d_and_activation(layer_2_pool, filter_width=3, filter_height=3,
                                                     filters_count=384, stride_x=1, stride_y=1,
                                                     padding='SAME', init_biases=False)

        # 卷积层4
        # input：13*13*384-->卷积后(filter:[3,3,384,384],strides=1,padding=SAME)：13*13*384（13=(13-3+2*(3-1)/2)/1+1）-->
        # (ReLU):13*13*384
        with tf.name_scope('layer_4'):
            layer_4_conv = layer.conv2d_and_activation(layer_3_conv, filter_width=3, filter_height=3,
                                                     filters_count=384, stride_x=1, stride_y=1,
                                                     padding='SAME', init_biases=True)

        # 卷积层5
        # input：13*13*384-->卷积后(filter:[3,3,384,256],strides=1,padding=SAME)：13*13*256（13=(13-3+2*(3-1)/2)/1+1）-->
        # (ReLU):13*13*256-->
        # 最大汇聚(ksize=3*3,strides=2,padding=Valid):6*6*256-->output:6*6*256(6=(13-3+2*0)/2+1)
        with tf.name_scope('layer_5'):
            layer_5_conv = layer.conv2d_and_activation(layer_4_conv, filter_width=3,
                                                       filter_height=3,filters_count=256,
                                                       stride_x=1, stride_y=1,padding='SAME',
                                                       init_biases=True)
            layer_5_pool = layer.max_pool(features=layer_5_conv, filters_width=3,
                                          filters_height=3, padding='VALID',
                                          stride_x=2, stride_y=2)

        # 6：全连接层1
        # input:6*6*256=9216-->fully connected:neurons=4096-->4096-->
        # ReLU-->4096-->Dropout-->4096-->output:4096
        with tf.name_scope('layer_6'):
            # 获得汇聚层输出值的形状
            pool_5_shape = layer_5_pool.get_shape().as_list()
            print(pool_5_shape)
            # 输入大小
            flattened_input_size = pool_5_shape[1]*pool_5_shape[2]*pool_5_shape[3]
            print(flattened_input_size)
            layer_6_fc = layer.fully_connected(features=tf.reshape(layer_5_pool, shape=[-1, flattened_input_size]),
                                               input_num=flattened_input_size, output_num=4096,
                                               relu=True, init_biases=True)
            layer_6_dropout = layer.dropout(layer_6_fc, self.dropout_keep_prob)

        # 7:全连接层2
        # input:4096-->fully connected:neurons=4096-->4096
        # 4096-->ReLU-->4096-->Dropout-->4096
        with tf.name_scope('layer_7'):
            layer_7_fc = layer.fully_connected(layer_6_dropout, 4096, 4096,
                                               relu=True, init_biases=True)
            layer_7_dropout = layer.dropout(layer_7_fc, self.dropout_keep_prob)

        # 8:输出层
        # input:4096-->fully connected:neurons=10-->10
        with tf.name_scope('layer_8'):
            logits = layer.fully_connected(layer_7_dropout, input_num=4096, output_num=self.num_classes,
                                           relu=False, name='logits')

        # 交叉熵损失函数
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,
                                                                       logits=logits,
                                                                       name='cross_entropy')
        layer.variable_summaries(cross_entropy)

        # 训练操作
        with tf.name_scope('training'):
            # 计算损失函数的平均值，添加到日志中
            self.loss_operation = tf.reduce_mean(cross_entropy, name='loss_operation')
            tf.compat.v1.summary.scalar(name='loss', tensor=self.loss_operation)
            # 优化器
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
            # 获取梯度和变量
            grads_and_vars = optimizer.compute_gradients(self.loss_operation)
            # 训练操作
            self.training_operation = optimizer.apply_gradients(grads_and_vars, name='training_operation')

            # 若梯度存在，则将该变量的梯度保存在日志中
            for grad, var in grads_and_vars:
                with tf.name_scope(var.op.name + '/gradients'):
                    layer.variable_summaries(grad)

        # 计算准确值
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
            self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')
            tf.compat.v1.summary.scalar(name='accuracy', tensor=self.accuracy_operation)


if __name__ == '__main__':
    alexnet = AlexNet()
