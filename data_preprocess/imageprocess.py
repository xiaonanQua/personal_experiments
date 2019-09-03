"""
提供图像数据处理方法:1.查看图像数据
"""
import matplotlib.pyplot as plt
import matplotlib.pylab as lab
import numpy as np
from skimage.transform import resize
import scipy.misc
import utils.tools as tool


class ImageProcess(object):
    def __init__(self):
        pass

    def show_image(self, image_matrix):
        """
        使用matplotlib显示图像矩阵,包括灰度图像和RGB图像
        :param image_matrix: 一张图像矩阵,格式是(高度, 宽度, 颜色通道)
        :param label: 标签数据,和图像相对应
        :return:
        """
        # 颜色通道
        num_channel = image_matrix.shape[2]
        # 若是灰度图像,则需要进行挤压
        if num_channel == 1:
            # 对灰度图像进行挤压,使得形状格式为(高度, 宽度)
            image_matrix = np.asarray(image_matrix).squeeze()
            plt.imshow(image_matrix, interpolation='none', cmap=lab.gray())
        else:
            # 显示图片
            plt.imshow(image_matrix)
        plt.show()

    def image_resize(self, images, resize_shape):
        """
        重塑一组图像大小
        :param images: 一组图像，数据格式：[图片数量，高度，宽度，通道]
        :param resize_shape: 重塑形状，形状格式：[高度， 宽度， 通道]
        :return: 重塑后的图像数据,数组形式
        """
        print("进行图像大小的重塑...")
        # 获得样本数量
        num_examples = images.shape[0]
        # 初始化重塑图像样本的形状
        resize_images = np.zeros(shape=[num_examples, resize_shape[0], resize_shape[1], resize_shape[2]], dtype=np.uint8)

        # 循环迭代所有图像
        for index in range(num_examples):
            # 重塑制定形状的图片并附加到列表中
            # image = resize(images[index], output_shape=resize_shape)
            image = scipy.misc.imresize(images[index], size=resize_shape, interp='bicubic')
            resize_images[index] = image
            tool.view_bar("重塑图像大小", index+1, num_examples)
        return resize_images
