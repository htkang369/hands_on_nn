from Utils import Data_util
from Network import Network

import numpy as np
import os

def load_image():
    """
    For loading data and create one-hot encoding labels
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

    # one hot encoding
    train_y_ = np.zeros((np.shape(train_y)[0], 10), dtype=float)
    test_y_ = np.zeros((np.shape(test_y)[0], 10), dtype=float)

    for i in range(len(train_y)):
        train_y_[i][train_y[i]] = 1.0

    for i in range(len(test_y)):
        test_y_[i][test_y[i]] = 1.0

    return train_x, train_y_, test_x, test_y_

if __name__ == '__main__':

    _, _, test_x, test_y = load_image()

    net = Network()  # weight init
    net.load_weight()
    net.eval(test_x, test_y, "test_data", save=False)
