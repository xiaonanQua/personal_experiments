# -*- coding: utf-8 -*-

import tensorflow as tf


def evaluate(sess, net, train_data, train_labels, test_data, test_labels, model_dir, log=None):
    """
    对训练好的模型进行测试评估
    :param sess: 会话
    :param net: 网络对象
    :param train_data: 训练数据
    :param train_labels: 训练标签
    :param test_data: 测试数据
    :param test_labels: 测试标签
    :param model_dir: 模型保存的目录
    :param log: 日志对象
    :return:
    """
    print('评估数据集...')
    print()

    print("加载训练好的模型...")
    print()
    saver = tf.train.Saver()
    saver.save(sess, tf.train.latest_checkpoint(model_dir))

    print('评估...')

    train_accuracy = sess.run(net.accuracy_operation, feed_dict={net.x: train_data,
                                                                 net.y: train_labels,
                                                                 net.dropout_keep_prob: 1.0})
    test_accuracy = sess.run(net.accuracy_operation, feed_dict={net.x: test_data,
                                                                 net.y: test_labels,
                                                                 net.dropout_keep_prob: 1.0})

    print("训练精度:{:.3f}".format(train_accuracy))
    print("测试精度：{:.3f}".format(test_accuracy))
    print()

    # 将输出数据保存到日志文件中
    log.info('训练精度:%.3f, 测试精度：%.3f', train_accuracy, test_accuracy)