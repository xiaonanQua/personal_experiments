"""
构建神经网络架构
"""
import tensorflow as tf
import logging as log

# 配置信息日志
log.basicConfig(level=log.INFO)


class ConvNet(object):
    """
    构建神经网络架构
    """
    def __init__(self):
        # 设置输入数据、标签占位符
        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='input_x')
        self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 10), name='label_y')
        # 设置Dropout函数的参数keep_prob占位符,保证每一层应该保存多少神经元，减少神经元，防止过拟合
        self.keep_prob = tf.compat.v1.placeholder(dtype=tf.float32, name='keep_prob')

    def conv_net(self):
        """
        构建神经网络层
        :return:最后的类别输出结果
        """
        # 定义四层卷积层的过滤器变量,全都使用截断高斯分布进行初始化
        conv1_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 3, 64], mean=0.0, stddev=0.08))
        conv2_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 64, 128], mean=0.0, stddev=0.08))
        conv3_filter = tf.Variable(tf.random.truncated_normal(shape=[5, 5, 128, 256], mean=0.0, stddev=0.08))
        conv4_filter = tf.Variable(tf.random.truncated_normal(shape=[5, 5, 256, 512], mean=0.0, stddev=0.08))

        # 卷积层1
        # 进行卷积运算，步长为1，用0进行填充。激活函数使用ReLU，使用批次正则化
        conv1 = tf.nn.conv2d(self.x, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(conv1)
        # 对卷积后的值进行最大池化进行降维，核尺寸是2*2，步长为1*1，进行0填充
        conv1_pool = tf.nn.max_pool2d(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        # conv1_bn = tf.layers.batch_normalization(conv1_pool)

        # 卷积层2
        conv2 = tf.nn.conv2d(conv1_pool, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(conv2)
        conv2_pool = tf.nn.max_pool2d(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # conv2_bn = tf.layers.batch_normalization(conv2_pool)

        # 卷积层3
        conv3 = tf.nn.conv2d(conv2_pool, conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.nn.relu(conv3)
        conv3_pool = tf.nn.max_pool2d(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # conv3_bn = tf.layers.batch_normalization(conv3_pool)

        # 卷积层4
        conv4 = tf.nn.conv2d(conv3_pool, conv4_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = tf.nn.relu(conv4)
        conv4_pool = tf.nn.max_pool2d(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # conv4_bn = tf.compat.v1.layers.batch_normalization(conv4_pool)

        # 将经过卷积层卷积后的数据展平为行向量
        flat = tf.compat.v1.layers.Flatten()(conv4_pool)

        # 全连接层1
        fc1_weights = tf.Variable(tf.random.truncated_normal([8192, 128], dtype=tf.float32, stddev=0.08))
        fc1_bias = tf.Variable(tf.constant(0.0, tf.float32, [128]), trainable=True)
        fc1 = tf.nn.relu(tf.matmul(flat, fc1_weights) + fc1_bias)
        fc1 = tf.nn.dropout(fc1, rate=1-self.keep_prob)
        # fc1 = tf.compat.v1.layers.batch_normalization(fc1)

        # 全连接层2
        fc2_weights = tf.Variable(tf.random.truncated_normal([128, 256], dtype=tf.float32, stddev=0.08))
        fc2_bias = tf.Variable(tf.constant(0.0, tf.float32, [256]), trainable=True)
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_bias)
        fc2 = tf.nn.dropout(fc2, rate=1-self.keep_prob)
        # fc2 = tf.compat.v1.layers.batch_normalization(fc2)

        # 全连接层3
        fc3_weights = tf.Variable(tf.random.truncated_normal([256, 512], dtype=tf.float32, stddev=0.08))
        fc3_bias = tf.Variable(tf.constant(0.0, tf.float32, [512]), trainable=True)
        fc3 = tf.nn.relu(tf.matmul(fc2, fc3_weights) + fc3_bias)
        fc3 = tf.nn.dropout(fc3, rate=1 - self.keep_prob)
        # fc3 = tf.compat.v1.layers.batch_normalization(fc3)

        # 全连接层4
        fc4_weights = tf.Variable(tf.random.truncated_normal([512, 1024], dtype=tf.float32, stddev=0.08))
        fc4_bias = tf.Variable(tf.constant(0.0, tf.float32, [1024]), trainable=True)
        fc4 = tf.nn.relu(tf.matmul(fc3, fc4_weights) + fc4_bias)
        fc4 = tf.nn.dropout(fc4, rate=1 - self.keep_prob)
        # fc4 = tf.compat.v1.layers.batch_normalization(fc4)

        # 输出层
        out_weights = tf.Variable(tf.random.truncated_normal([1024, 10], stddev=0.08, dtype=tf.float32))
        out_bias = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), trainable=True)
        out = tf.matmul(fc4, out_weights) + out_bias

        return out


if __name__ == '__main__':
    conv_net = ConvNet()
    conv_net.conv_net()