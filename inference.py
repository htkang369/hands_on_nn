from Utils import Data_util, Function
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

    F = Function()
    _, _, test_x, test_y = load_image()

    net = Network()  # weight init
    net.load_weight()
    max_num = np.shape(test_x)[0]
    choose_num = np.random.randint(0, max_num)  # randomly choose a image from test set
    input_image = test_x[choose_num][:]
    input_label = test_y[choose_num][:]

    pred = net.forward(input_image)
    pred_final = np.argmax(pred, axis=1)
    label = np.argmax(input_label, axis=0)

    print("Input label is {0}, predicted label is {1}".format(label, pred_final[0]))
    F.out_img(input_image)  # display image










