"""
AlexNet网络结构
"""

import tensorflow as tf


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

        # 定义初始化变量时高斯分布的参数：均值、标准差
        self.random_mean = 0
        self.random_stddev = 0.01

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
            layer_1_conv = self._conv2d_and_activation(self.x, filter_width=11, filter_height=11,
                                                     filters_count=96, stride_x=4, stride_y=4,
                                                     padding='VALID', init_biases=False)
            layer_1_lrn = self._local_reponse_normalization(features=layer_1_conv)
            layer_1_pool = self._max_pool(features=layer_1_lrn, filters_width=3, filters_height=3,
                                         padding='VALID', stride_x=2, stride_y=2)

        # 卷积层2
        # input：27*27*96-->卷积后(filter:[5,5,96,256],strides=1,padding=SAME)：27*27*256（27=(27-5+2*(5-1)/2)/1+1）-->
        # (ReLU):27*27*256-->(Local Response Normalization):27*27*256-->
        # 最大汇聚(ksize=3*3,strides=2,padding=Valid):13*13*256-->output:13*13*256(13=(27-3+2*0)/2+1)
        with tf.name_scope('layer_2'):
            layer_2_conv = self._conv2d_and_activation(layer_1_pool, filter_width=5, filter_height=5,
                                                     filters_count=256, stride_x=1, stride_y=1,
                                                     padding='SAME', init_biases=True)
            layer_2_lrn = self._local_reponse_normalization(features=layer_2_conv)
            layer_2_pool = self._max_pool(features=layer_2_lrn, filters_width=3, filters_height=3,
                                         padding='VALID', stride_x=2, stride_y=2)

        # 卷积层3
        # input：13*13*256-->卷积后(filter:[3,3,256,384],strides=1,padding=SAME)：13*13*384（13=(13-3+2*(3-1)/2)/1+1）-->
        # (ReLU):13*13*384
        with tf.name_scope('layer_3'):
            layer_3_conv = self._conv2d_and_activation(layer_2_pool, filter_width=3, filter_height=3,
                                                     filters_count=384, stride_x=1, stride_y=1,
                                                     padding='SAME', init_biases=False)

        # 卷积层4
        # input：13*13*384-->卷积后(filter:[3,3,384,384],strides=1,padding=SAME)：13*13*384（13=(13-3+2*(3-1)/2)/1+1）-->
        # (ReLU):13*13*384
        with tf.name_scope('layer_4'):
            layer_4_conv = self._conv2d_and_activation(layer_3_conv, filter_width=3, filter_height=3,
                                                     filters_count=384, stride_x=1, stride_y=1,
                                                     padding='SAME', init_biases=True)

        # 卷积层5
        # input：13*13*384-->卷积后(filter:[3,3,384,256],strides=1,padding=SAME)：13*13*256（13=(13-3+2*(3-1)/2)/1+1）-->
        # (ReLU):13*13*256-->
        # 最大汇聚(ksize=3*3,strides=2,padding=Valid):6*6*256-->output:6*6*256(6=(13-3+2*0)/2+1)
        with tf.name_scope('layer_5'):
            layer_5_conv = self._conv2d_and_activation(layer_4_conv, filter_width=3,
                                                       filter_height=3,filters_count=256,
                                                       stride_x=1, stride_y=1,padding='SAME',
                                                       init_biases=True)
            layer_5_pool = self._max_pool(features=layer_5_conv, filters_width=3,
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
            layer_6_fc = self._fully_connected(features=tf.reshape(layer_5_pool, shape=[-1, flattened_input_size]),
                                               input_num=flattened_input_size, output_num=4096,
                                               relu=True, init_biases=True)
            layer_6_dropout = self._dropout(layer_6_fc)

        # 7:全连接层2
        # input:4096-->fully connected:neurons=4096-->4096
        # 4096-->ReLU-->4096-->Dropout-->4096
        with tf.name_scope('layer_7'):
            layer_7_fc = self._fully_connected(layer_6_dropout, 4096, 4096,
                                               relu=True, init_biases=True)
            layer_7_dropout = self._dropout(layer_7_fc)

        # 8:输出层
        # input:4096-->fully connected:neurons=10-->10
        with tf.name_scope('layer_8'):
            logits = self._fully_connected(layer_7_dropout, 4096, self.num_classes,
                                           relu=False, name='logits')

        # 交叉熵损失函数
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,
                                                                       logits=logits,
                                                                       name='cross_entropy')
        self._variable_summaries(cross_entropy)

        # 训练
        with tf.name_scope('training'):
            # 计算损失函数的平均值，添加到日志中
            loss_operation = tf.reduce_mean(cross_entropy, name='loss_operation')
            tf.compat.v1.summary.scalar(name='loss', tensor=loss_operation)
            # 优化器
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                   momentum=self.momentum)
            # 获取梯度和变量
            grads_and_vars = optimizer.compute_gradients(loss_operation)
            # 训练操作
            self.training_operation = optimizer.apply_gradients(grads_and_vars, name='training_operation')

            # 若梯度存在，则将该变量的梯度保存在日志中
            for grad, var in grads_and_vars:
                with tf.name_scope(var.op.name + '/gradients'):
                    self._variable_summaries(grad)

        # 准确值
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
            self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')
            tf.compat.v1.summary.scalar(name='accuracy', tensor=self.accuracy_operation)

    def train_epoch(self, sess, x_data, y_data, batch_size=128,
                    file_writer=None, summary_operation=None, epoch_number=None):
        """
        使用一个128批次数量的样本并随机梯度下降训练我们的模型
        :param sess: 用于计算的会话
        :param x_data: 输入特征数据
        :param y_data: 标签数据
        :param batch_size: 批次大小
        :param file_writer:
        :param summary_operation: 日志操作
        :param epoch_number: 周期数量
        :return:
        """
        # 样本数量, 训练步骤
        num_example = len(x_data)
        step = 0
        for offset in (0, num_example, batch_size):
            end = offset + batch_size
            batch_x, batch_y = x_data[offset:end], y_data[offset, end]
            if file_writer is not None and summary_operation is not None:
                _, summary = sess.run([self.training_operation, summary_operation],
                                      feed_dict={self.x: batch_x, self.y: batch_y,
                                                 self.dropout_keep_prob: self.keep_prob})
                file_writer.add_summary(summary, epoch_number*(num_example//batch_size+1)+step)
                step += 1
            else:
                sess.run(self.training_operation, feed_dict={self.x: batch_x, self.y: batch_y,
                                                             self.dropout_keep_prob: self.keep_prob})

    def evaluate(self, sess, x_data, y_data, batch_size=128):
        """
        评估整体数据的准确率
        :param sess: 会话
        :param x_data: 输入数据
        :param y_data: 标签数据
        :param batch_size: 批次大小
        :return:
        """
        # 样本数量、整体准确率
        num_examples = len(x_data)
        total_accuracy = 0
        for offset in range(0, num_examples, batch_size):
            end = offset+batch_size
            batch_x, batch_y = x_data[offset:end], y_data[offset:end]
            # 每批次数据的准确率
            batch_accuracy = sess.run(self.accuracy_operation,
                                      feed_dict={self.x: batch_x,
                                                 self.y: batch_y,
                                                 self.dropout_keep_prob: 1.0})
            total_accuracy += (batch_accuracy*len(batch_x))
        return total_accuracy/num_examples

    def save(self, sess, file_name):
        """
        对训练的模型进行保存
        :param sess: 会话
        :param file_name:保存的文件名称
        :return:
        """
        saver = tf.train.Saver()
        saver.save(sess, file_name)

    def restore(self, sess, checkpoint_dir):
        """
        恢复训练好的模型参数
        :param sess: 会话
        :param checkpoint_dir: 检查点目录
        :return:
        """
        saver = tf.train.Saver()
        saver.save(sess, tf.train.latest_checkpoint(checkpoint_dir))

    def _conv2d_and_activation(self, features, filter_width, filter_height, filters_count,
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
            filters = tf.Variable(self._random_value(shape=[filter_width, filter_height, input_channels, filters_count]),
                                  name='filters')
            # 执行卷积运算,添加偏置值，使用ReLU激活函数进行激活
            convs = tf.nn.conv2d(features, filters, strides=[1, stride_x, stride_y, 1], padding=padding, name='convs')
            preactivations = tf.nn.bias_add(convs, biases, name='preactivations')
            activations = tf.nn.relu(preactivations, name='activations')

            # 将变量值(卷积核参数，偏置值参数)，未激活的卷积数据和激活后的卷积数据添加到日志中
            with tf.name_scope('filter_summaries'):
                self._variable_summaries(filters)
            with tf.name_scope('biases_summaries'):
                self._variable_summaries(biases)
            with tf.name_scope('preactivation_histogram'):
                tf.compat.v1.summary.histogram('preactivations', preactivations)
            with tf.name_scope('activation_histogram'):
                tf.compat.v1.summary.histogram('activations', activations)

            return activations

    def _max_pool(self, features, filters_width, filters_height, stride_x, stride_y, padding='VALID', name='pool'):
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
            return tf.nn.max_pool2d(features, ksize=[1, filters_width, filters_height, 1],
                                    strides=[1, stride_x, stride_y, 1], padding=padding, name='max_pool')

    def _fully_connected(self, features, input_num, output_num, relu=True,
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
            weights = tf.Variable(self._random_value(shape=[input_num, output_num]), name='weights')
            if init_biases:
                biases = tf.Variable(tf.ones(shape=[output_num], dtype=tf.float32), name='biases')
            else:
                biases = tf.Variable(tf.zeros(shape=[output_num], dtype=tf.float32), name='biases')
            # 线性值
            preactivations = tf.nn.bias_add(tf.matmul(features, weights), biases, name='preactivations')
            # 使用激活函数进行激活
            if relu:
                activations = tf.nn.relu(preactivations)

            # 将参数变量、线性值、激活值添加到日志中，用于tensorboard查看
            with tf.name_scope('weight_summaries'):
                self._variable_summaries(weights)
            with tf.name_scope('biases_summaries'):
                self._variable_summaries(biases)
            with tf.name_scope('preactivations_histogram'):
                tf.compat.v1.summary.histogram('preactivations', preactivations)

            if relu:
                with tf.name_scope('activation_histogram'):
                    tf.compat.v1.summary.histogram('activations', activations)
                return activations
            else:
                return preactivations

    def _dropout(self, features, name='dropout'):
        """
        对神经元进行一定比例丢弃
        :param features: 特征数据
        :param name: 名称
        :return: 丢弃后的特征
        """
        with tf.name_scope(name):
            return tf.nn.dropout(features, rate=1-self.dropout_keep_prob, name='dropout')

    def _local_reponse_normalization(self, features, name='lrn'):
        """
        局部响应归一化
        :param features: 特征数据
        :param name: 名称
        :return:
        """
        with tf.name_scope(name):
            return tf.nn.local_response_normalization(features, depth_radius=2, alpha=10**-4,
                                                      beta=0.75, name='local_response_normalization')

    def _random_value(self, shape):
        """
        获取随机值，且随机值满足高斯分布
        :param shape: 数据的形状
        :return: 随机值
        """
        return tf.random.normal(shape=shape,
                                mean=self.random_mean,
                                stddev=self.random_stddev,
                                dtype=tf.float32)

    def _variable_summaries(self, var):
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


if __name__ == '__main__':
    alexnet = AlexNet()
