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

    # one hot encoding
    train_y_ = np.zeros((np.shape(train_y)[0], 10), dtype=float)
    test_y_ = np.zeros((np.shape(test_y)[0], 10), dtype=float)

    for i in range(len(train_y)):
        train_y_[i][train_y[i]] = 1.0

    for i in range(len(test_y)):
        test_y_[i][test_y[i]] = 1.0

    return train_x, train_y_, test_x, test_y_


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_image()
    # train_x = np.random.random((64, 784))
    # train_y = np.ones((64, 10))
    # prep = Preprocess()
    func = Function()

    train_num = np.shape(train_x)[0]
    batch_size = config.batch_size
    # batch_size = 8
    max_training_time = config.max_training_time

    net = Network()  # weight init

    count = 1

    for epoch in range(1, max_training_time + 1):

        index_vector = np.array([i for i in range(train_num)])
        np.random.shuffle(index_vector)

        for j in range(1, train_num//batch_size + 1):
            batch_train_x = train_x[index_vector[(j-1):j * batch_size]][:]
            batch_train_y = train_y[index_vector[(j-1):j * batch_size]][:]
            batch_pred = net.forward(batch_train_x)
            batch_loss = net.back_propagation(batch_train_x, batch_train_y, batch_pred)
            count += 1

            if count % 25 == 0:
                print("After {0} mini batch training, the loss is {1}".format(count, batch_loss))
                net.eval(batch_train_x, batch_train_y, "train_data", save=False)
                net.eval(test_x, test_y, "test_data", save=True)
        if epoch % 2 == 0:
            net.lr_decay()



















