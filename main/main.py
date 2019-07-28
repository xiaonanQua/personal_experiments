"""
主函数
"""
# 导入必要的函数库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 导入自定义预处理、训练的函数库
from preprocess_data import preprocess as prepro
from train_and_test import train_neural_network as train
from config import config as cfg


class Main(object):

    def __init__(self):
        # 获得预处理、训练的对象
        self.prepro = prepro.Preprocess()
        self.train = train.TrainNeuralNetwork()
        self.cfg = cfg.Config()

    def main(self):
        self.prepro.preprocess_and_save_data()
        self.train.train_nerual_network()


if __name__ == '__main__':
    Main().main()