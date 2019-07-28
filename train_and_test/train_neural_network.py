"""
对构建好的神经网络进行训练
"""
import tensorflow as tf
from config import config as cfg
from load_data import loaddataset
from neural_network import conv_net
cfg = cfg.Config()
cifar = loaddataset.DataSetOp()


class TrainNeuralNetwork(object):

    def __init__(self):
        # 定义损失函数、优化器、准确度
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.x = None
        self.y = None
        self.keep_prob = None
        self.batch_size = None

        # 定义学习率，训练周期,数据x和标签y
        self.learning_rate = cfg.hyperparam['learning_rate']
        self.epochs = cfg.hyperparam['epochs']

        # 预处理的训练、验证、测试、保存路径，训练后模型保存路径,读取的数据名称
        self.prepro_train_path = cfg.prepro_train_path
        self.prepro_valid_path = cfg.prepro_valid_path
        self.train_model_path = cfg.train_model_path

        # 初始化训练的组件
        self.init_component()

    def init_component(self):
        """
        初始化损失函数、优化器、准确度
        :return:
        """
        # 声明神经网络结构对象
        net = conv_net.ConvNet()
        self.x = net.x
        self.y = net.y
        self.keep_prob = net.keep_prob
        self.batch_size = cfg.hyperparam['batch_size']
        # 获得卷积神经网络结构输出特征
        logits = net.conv_net()
        # 定义损失函数
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=net.y))
        # 定义优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # 准确率
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(net.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    def train_nerual_network(self):
        """
        进行神经网络的训练
        :return:
        """
        # 开启会话进行训练
        with tf.Session() as sess:
            # 初始化全局变量
            tf.global_variables_initializer().run()
            valid_data = cifar.load_preprocess_data(self.prepro_valid_path, 'preprocess_validation.p')
            j = 0
            # 训练循环
            for epoch in range(self.epochs):
                # 循环所有批次
                for i in range(1, 6):

                    data_name = 'preprocess_batch_{}.p'.format(i)
                    batch_data = cifar.load_preprocess_data(self.prepro_train_path,
                                                            data_name,
                                                            batch_size=self.batch_size,
                                                            index=j)
                    if i % 5 == 0:
                        j = j + 1
                    features, labels = batch_data
                    sess.run(self.optimizer, feed_dict={self.x: features,
                                                        self.y: labels,
                                                        self.keep_prob: cfg.hyperparam['keep_prob']})

                    loss = sess.run(self.loss, feed_dict={self.x: features,
                                                          self.y: labels,
                                                          self.keep_prob: 1})
                    valid_acc = sess.run(self.accuracy, feed_dict={self.x: valid_data[0],
                                                                   self.y: valid_data[1],
                                                                   self.keep_prob: 1})
                    print('Epoch:{}, CIFAR-10_Batch_{}, '.format(epoch+1, i), end='')
                    print('loss:{:>10.4f}, valid_acc:{:.6f}'.format(loss, valid_acc))


if __name__ == '__main__':
    train = TrainNeuralNetwork()
    train.train_nerual_network()


