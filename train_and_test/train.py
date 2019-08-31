"""
训练数据
"""
import tensorflow as tf
import numpy as np
from neural_network.alexnet import AlexNet
from data_load.loaddata import LoadDataSet
from config.alexnet_config import AlexNetConf
from data_preprocess.preprocessdata import PreprocessData
from data_preprocess.imageprocess import ImageProcess

# 实例化对象
cfg = AlexNetConf()
data = LoadDataSet()
preprocess = PreprocessData()
image_process = ImageProcess()

print('读取cifar-10数据集...')
# cifar-10训练、测试数据名称列表
train_list = cfg.cifar_file_name['train']
test_list = cfg.cifar_file_name['test']

# 初始化训练、测试数据列表
train_data, train_labels = [], []
# 读取pickle中所有数据
for batch_name in train_list:
    print(batch_name)
    # 获取批次数据
    batch_data, batch_labels = data.load_cifar_10(file_dir=cfg.cifar_10_dir,
                                                     file_name=batch_name)
    # 将数据和标签附加进列表中
    train_data.extend(batch_data)
    train_labels.extend(batch_labels)
test_data, test_labels = data.load_cifar_10(file_dir=cfg.cifar_10_dir,
                                               file_name=test_list)
# 将读取的训练数据转化成数组, 并和测试数据一起重塑图像大小
train_data = image_process.image_resize(np.array(train_data), resize_shape=[cfg.image_height,
                                                                            cfg.image_width,
                                                                            cfg.image_channels])
test_data = image_process.image_resize(test_data, resize_shape=[cfg.image_height,
                                                                cfg.image_height,
                                                                cfg.image_channels])
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
with tf.compat.v1.Session() as sess:
    print("进行模型的训练...")
    print()
    # 创建日志文件写入对象
    log_writer = tf.compat.v1.summary.FileWriter(logdir=cfg.log_dir, graph=sess.graph)
    # 合并所有日志
    summary_operation = tf.compat.v1.summary.merge_all()
    # 初始化所有变量
    sess.run(tf.compat.v1.global_variables_initializer())

    # 进行模型的训练
    for i in range(cfg.epochs):
        # 计算训练和测试的准确度,并输出
        print('计算准确度..')
        train_accuracy = alexnet.evaluate(sess, train_data, train_labels, cfg.batch_size)
        test_accuracy = alexnet.evaluate(sess, test_data, test_labels, cfg.batch_size)
        print('训练准确度：{:.3f}'.format(train_accuracy))
        print('测试准确度：{:.3f}'.format(test_accuracy))
        print()

        # 进行训练
        print('训练批次{}...'.format(i+1))
        alexnet.train_epoch(sess, train_data, train_labels, cfg.batch_size,
                            log_writer, summary_operation, i)
        print()

    # 计算最终训练和测试的准确度
    final_train_accuracy = alexnet.evaluate(sess, train_data, train_labels, cfg.batch_size)
    final_test_accuracy = alexnet.evaluate(sess, test_data, test_labels, cfg.batch_size)
    print('最终训练准确度：{:.3f}'.format(final_train_accuracy))
    print('最终测试准确度：{:.3f}'.format(final_test_accuracy))
    print()

    # 保存模型
    alexnet.save(sess, cfg.model_dir)
    print('保存模型...')
    print()

print('训练结束...')






