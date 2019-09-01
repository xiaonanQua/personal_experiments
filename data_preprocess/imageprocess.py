"""
提供图像数据处理方法:1.查看图像数据
"""
import matplotlib.pyplot as plt
import matplotlib.pylab as lab
import numpy as np
from skimage.transform import resize
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
        # 获得样本数量,定义重塑图像列表
        num_examples = images.shape[0]
        image_list = []

        # 循环迭代所有图像
        for i in range(num_examples):
            # 重塑制定形状的图片并附加到列表中
            image = resize(images[i], output_shape=resize_shape)
            image_list.append(image)
            tool.view_bar("重塑图像大小", i+1, num_examples)
        return np.array(image_list)
