"""
Author : Hengtong Kang
E-mail : hengtongkang@ufl.edu
"""


from Utils import Data_util, Preprocess, Function
from Network import Network
import Config as config

import numpy as np
import matplotlib.pyplot as plt
import os

def load_image():
    """
    For loading data
    :return:
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))  # get path
    file_train_x = dir_path + "/data/train-images.idx3-ubyte"
    file_train_y = dir_path + "/data/train-labels.idx1-ubyte"
    file_test_x = dir_path + "/data/t10k-images.idx3-ubyte"
    file_test_y = dir_path + "/data/t10k-labels.idx1-ubyte"

    train_x = Data_util(file_train_x).get_image()
    train_y = Data_util(file_train_y).get_label()
    test_x = Data_util(file_test_x).get_image()
    test_y = Data_util(file_test_y).get_label()
    return train_x, train_y, test_x, test_y
    # return test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_image()

    # prep = Preprocess()
    func = Function()

    train_num = np.shape(train_x)[0]
    batch_size = config.batch_size

    net = Network()  # weight init














