"""
训练数据
"""
import tensorflow as tf
from neural_network.alexnet import AlexNet
from data_load.loaddata import LoadDataSet
from config.alexnet_config import AlexNetConf
from data_preprocess.preprocessdata import PreprocessData
from data_preprocess.imageprocess import ImageProcess
from train_and_test import train, evaluate
import logging as log
log.basicConfig(filename='alexnet.txt', level=log.INFO)
import os
from original.dataset_helper import read_cifar_10
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 实例化对象
cfg = AlexNetConf()
data = LoadDataSet()
preprocess = PreprocessData()
image_process = ImageProcess()

print('读取cifar-10数据集...')
# 加载cifar-10训练、测试数据
train_data, train_labels, test_data, test_labels = data.load_cifar_10(file_dir=cfg.cifar_10_dir,
                                                                      train_file_name=cfg.cifar_file_name['train'],
                                                                      test_file_name=cfg.cifar_file_name['test'],
                                                                      )
# train_data, train_labels, test_data, test_labels = read_cifar_10(image_width=cfg.image_width,
#                                                                  image_height=cfg.image_height)

# 将读取的训练数据转化成数组, 并和测试数据一起重塑图像大小
train_data = image_process.image_resize(train_data, resize_shape=[cfg.image_height,
                                                                            cfg.image_width,
                                                                            cfg.image_channels])
test_data = image_process.image_resize(test_data, resize_shape=[cfg.image_height,
                                                                cfg.image_height,
                                                                cfg.image_channels])
# # 对训练、测试数据进行数据归一化操作
# train_data = preprocess.normalize(train_data)
# test_data = preprocess.normalize(test_data)

# 将标签列表转化成one-hot向量
train_labels = preprocess.one_hot_encode(train_labels, num_classes=cfg.num_classes)
test_labels = preprocess.one_hot_encode(test_labels, num_classes=cfg.num_classes)

print("读取的cifar-10数据集的训练和测试形状:")
print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)

# 实例化网络
alexnet = AlexNet(input_width=cfg.image_width, input_height=cfg.image_height, input_channels=cfg.image_channels,
                  num_classes=cfg.num_classes, learning_rate=cfg.learning_rate,
                  momentum=cfg.momentum, keep_prob=cfg.keep_prob)

# 开启会话，进行模型的训练
with tf.compat.v1.Session(config=tf.ConfigProto(log_device_placement=True
                                                )) as sess:
    print("进行模型的训练...")
    print()
    # 创建日志文件写入对象
    log_writer = tf.compat.v1.summary.FileWriter(logdir=cfg.log_dir, graph=sess.graph)
    # 合并所有日志
    summary_operation = tf.compat.v1.summary.merge_all()
    # 初始化所有变量
    sess.run(tf.compat.v1.global_variables_initializer())

    # 进行模型的训练
    # with tf.device("/gpu:0"):
    train.train_epoch(sess, alexnet, train_data, train_labels, batch_size=cfg.batch_size,
                      file_writer=log_writer, summary_operation=summary_operation,
                      epoch_number=cfg.epochs, valid_size=0.1)

    # 保存模型
    train.save(sess, cfg.model_dir)
    print('保存模型...')
    print()
    print('训练结束...')

    # 评估模型
    evaluate.evaluate(sess, alexnet, train_data, train_labels, test_data, test_labels, cfg.model_dir)





