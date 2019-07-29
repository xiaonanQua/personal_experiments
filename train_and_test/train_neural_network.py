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
        # 定义损失函数、优化器、准确度、模型
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.model = None

        # 定义训练的数据x和标签y、保留率、批次大小、学习率，训练周期
        self.x = None
        self.y = None
        self.keep_prob = None
        self.batch_size = cfg.hyperparam['batch_size']
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
        初始化损失函数、优化器、准确度、模型、超参数等数据
        :return:
        """
        # 声明神经网络结构对象、配置文件对象、数据集加载对象
        net = conv_net.ConvNet()

        # 获取网络结构中的占位符，用于训练时进行数据输送
        self.x = net.x
        self.y = net.y
        self.keep_prob = net.keep_prob

        # 获得卷积神经网络结构输出特征，并定义模型
        logits = net.conv_net()
        self.model = tf.identity(logits, name='logits')  # 用于训练结束后加载训练模型

        # 初始化损失函数、优化器、准确率
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
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
            index = 0
            # 训练循环
            for epoch in range(self.epochs):
                # 循环所有批次
                for i in range(1, 6):
                    # 加载批次数据
                    data_name = 'preprocess_batch_{}.p'.format(i)
                    batch_data = cifar.load_preprocess_data(self.prepro_train_path,
                                                            data_name,
                                                            batch_size=self.batch_size,
                                                            index=index)
                    # 处理索引值
                    if i % 5 == 0:
                        index = index + 1
                    if index >= int(9000/self.batch_size):
                        index = 0

                    # 获取批次中的特征、标签数据
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

            # 保存模型
            saver = tf.train.Saver()
            saver.save(sess, cfg.train_model_path)


if __name__ == '__main__':
    train = TrainNeuralNetwork()
    train.train_nerual_network()


