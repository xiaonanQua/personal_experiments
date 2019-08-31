import collections
import tensorflow as tf
from tensorflow.contrib import slim


class Block(collections.namedtuple("block", ["name", "residual_unit", "args"])):
    """
    创建Block类，用来配置残差学习模型的大小。
    使用collections的namedtuple来初始化Block类，使其能够以元祖的形式为其提供需要传入的参数
    """


def conv2d_same(inputs, num_outputs,  kernel_size, stride, scope=None):
    """
    创建卷积层，根据不同stride进行不同的填充。
    :param inputs: 输入的单元
    :param num_outputs: 输出单元
    :param kernel_size: 卷积核大小
    :param stride: 步长，若步长为1，则进行SAME模型填充，否则显式调用pad（）函数进行填充0操作
    :param scope: 范围
    :return:
    """
    # 若步长为1，则直接使用padding=“SAME”的方式进行卷积操作，一般步长不为1的情况出现在残差学习模块的最后一个卷积操作中
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,padding="SAME", scope=scope)
    else:
        # 填充的开始和结束位置
        pad_begin = (kernel_size-1)//2
        pad_end = kernel_size-1-pad_begin

        # pad()函数用于对矩阵进行定制填充
        # 这里用于对inputs进行向上填充pad_begin行0，向左填充pad_begin列0，向下填充pad_end行0，向右填充pad_end列0
        inputs = tf.pad(inputs, [[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride, padding="VALID", scope=scope)


def residual_unit(inputs, depth, depth_residual, stride, outputs_collections=None, scope=None):
    """
    定义残差学习单元
    :param inputs: 输入图像数据
    :param depth: 一个残差学习单元中第三个卷积层的输出通道数
    :param depth_residual: 前两个卷积层的输出通道数
    :param stride: 中间卷积层的步长
    :param outputs_collections: 输出的集合
    :param scope: 范围
    :return:
    """
    with tf.variable_scope(scope, "residual_v2", [inputs]) as sc:
        # 输入的通道数，取inputs形状的最后一个元素
        depth_input = inputs.getshape()
