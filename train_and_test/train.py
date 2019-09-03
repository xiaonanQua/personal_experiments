"""
训练操作
"""
import tensorflow as tf
from utils.queue_linked_list import QueueLink
from data_preprocess.preprocessdata import PreprocessData


def train_epoch(sess, net, train_data, train_labels, batch_size=128, valid_size=None,
                file_writer=None, summary_operation=None, epoch_number=None, log=None):
    """
    使用一个128批次数量的样本并随机梯度下降训练我们的模型
    :param sess: 用于计算的会话
    :param net: 用于训练的网络结构
    :param train_data: 训练的特征数据
    :param train_labels: 训练的标签数据
    :param batch_size: 批次大小
    :param valid_size: 若值不为空，则对训练集按比例值划分出验证集
    :param file_writer: 日志文件写入对象
    :param summary_operation: 日志操作
    :param epoch_number: 训练周期数
    :param log: 日志对象
    :return:
    """
    # 样本数量,定义验证集
    num_example = len(train_data)
    valid_data, valid_labels = [], []
    # 实例化队列对象，用于批次数据的读取
    queue = QueueLink()
    # 实例化数据预处理对象
    preprocess = PreprocessData()

    # 若valid_size不为空，则对训练集按一定比例划分数据
    if valid_size is not None:
        train_data, train_labels, valid_data, valid_labels = preprocess.divide_valid_data(train_data,
                                                                                          train_labels,
                                                                                          valid_size)
    # print(train_data.shape, train_labels.shape, valid_data.shape, valid_labels.shape)
    # 周期循环进行训练
    for epoch in range(1, epoch_number+1):
        # 每周期训练步骤等于总样本//批次大小
        # for step in range(0, num_example, batch_size):
        for step in range(num_example//batch_size):
            end = step + batch_size
            # 获取批次训练数据
            batch_data, batch_labels = preprocess.batch_and_shuffle_data(train_data, train_labels, batch_size, queue)
            # batch_data, batch_labels = train_data[step:end], train_labels[step:end]
            # 进行训练并保存日志文件
            if file_writer is not None and summary_operation is not None:
                _, summary = sess.run([net.training_operation, summary_operation],
                                      feed_dict={net.x: batch_data, net.y: batch_labels,
                                                 net.dropout_keep_prob: net.keep_prob})
                file_writer.add_summary(summary, epoch_number * (num_example // batch_size + 1) + step)
            else:
                sess.run(net.training_operation, feed_dict={net.x: batch_data, net.y: batch_labels,
                                                            net.dropout_keep_prob: net.keep_prob})

            # 获得loss值和批次准确率
            loss = sess.run(net.loss_operation, feed_dict={net.x: batch_data, net.y: batch_labels,
                                                           net.dropout_keep_prob: 1.0})
            batch_accuracy = sess.run(net.accuracy_operation, feed_dict={net.x: batch_data,
                                                                         net.y: batch_labels,
                                                                         net.dropout_keep_prob: 1.0})
            # 计算验证准确率
            if valid_size is not None:
                valid_accuracy = sess.run(net.accuracy_operation, feed_dict={net.x: valid_data,
                                                                             net.y: valid_labels,
                                                                             net.dropout_keep_prob: 1.0})
            else:
                valid_accuracy = None

            # 输出得到的值
            print("epoch：{}，step:{}/{},loss:{},  batch_accuracy:{},  valid_accuracy:{}".format(epoch,
                                                                                           step+1,
                                                                                           num_example//batch_size,
                                                                                           loss,
                                                                                           batch_accuracy,
                                                                                           valid_accuracy))
            # 用日志文件保存控制输出的值
            if log is not None:
                log.info("epoch：%s，step:%d/%d,loss:%.4f,  batch_accuracy:%.4f,  valid_accuracy:%.4f", epoch, step+1,
                        num_example//batch_size, loss, batch_accuracy, valid_accuracy)
            print()


def save(sess, file_name):
    """
    对训练的模型进行保存
    :param sess: 会话
    :param file_name:保存的文件名称
    :return:
    """
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, file_name)
