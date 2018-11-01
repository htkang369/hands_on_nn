import numpy as np
import copy
import os

import Config as config
from Utils import Function

F = Function()

dir_path = os.path.dirname(os.path.realpath(__file__))  # get path

class Network(object):
    """
    Network building
    """
    def __init__(self):
        self.num_input = 784
        self.num_h1 = config.num_h1
        self.num_h2 = config.num_h2
        self.num_output = 10

        mu, sigma = config.mu, config.sigma

        self.w_i_h1 = np.random.normal(mu, sigma, size=(self.num_input, self.num_h1))
        self.w_h1_h2 = np.random.normal(mu, sigma, size=(self.num_h1, self.num_h2))
        self.w_h2_o = np.random.normal(mu, sigma, size=(self.num_h2, self.num_output))

        self.b1 = np.random.normal(mu, sigma, size=(1, self.num_h1))
        self.b2 = np.random.normal(mu, sigma, size=(1, self.num_h2))
        self.b3 = np.random.normal(mu, sigma, size=(1, self.num_output))

        self.pred = None
        self.h1 = None
        self.h2 = None

        self.vb1 = self.vb2 = self.vb3 = self.vw_i_h1 = self.vw_h1_h2 = self.vw_h2_o = 0

        self.batch_size = config.batch_size
        self.lr = config.learning_rate
        self.mu = config.momentum
        self.dropout = config.dropout

        self.name = config.activate
        self.early_stop_time = 0
        self.pre_accu = 0  # previous accurate rate

    def forward(self, input_data):
        if self.name is "sigmoid":
            if self.dropout < 1:
                self.h1 = self.dropout(F.sigmoid(np.dot(input_data, self.w_i_h1) + self.b1))
                self.h2 = self.dropout(F.sigmoid(np.dot(self.h1, self.w_h1_h2) + self.b2))
            else:
                self.h1 = F.sigmoid(np.dot(input_data, self.w_i_h1) + self.b1)  # 64*784 784*300 64*300
                self.h2 = F.sigmoid(np.dot(self.h1, self.w_h1_h2) + self.b2)  # 64*300 300*100 64*100

        if self.name is "relu":
            self.h1 = F.relu(np.dot(input_data, self.w_i_h1) + self.b1)  # 64*784 784*300 64*300
            self.h2 = F.relu(np.dot(self.h1, self.w_h1_h2) + self.b2)  # 64*300 300*100 64*100

        o = np.dot(self.h2, self.w_h2_o) + self.b3  # 64*100 100*10 64*10

        rep = np.tile(np.max(o, axis=1), (self.num_output, 1))
        temp_o = o - rep.transpose()
        pred = np.exp(temp_o)/(np.tile(np.sum(np.exp(temp_o), axis=1), (self.num_output, 1)).transpose())

        return pred

    def back_propagation(self, x, l, pred):
        batch_size = self.batch_size
        lr = self.lr
        mu = self.mu

        loss = pred - l  # 64*10

        d_w_h2_o = (float(1) / batch_size) * np.dot(self.h2.transpose(), loss)  # 100*10
        d_b3 = (float(1) / batch_size) * np.sum(loss, 0).transpose()

        if self.name is "sigmoid":
            deri_h2 = np.multiply(self.h2, 1.0 - self.h2)
        if self.name is "relu":
            deri_h2 = copy.deepcopy(self.h2)
            deri_h2[deri_h2 > 0] = 1.0
            deri_h2[deri_h2 <= 0] = 0.0

        d_w_h1_h2 = (float(1) / batch_size) * np.dot(self.h1.transpose(),
                                              np.multiply(np.dot(loss, self.w_h2_o.transpose()), deri_h2))
        d_b2 = (float(1) / batch_size) * np.sum(np.multiply(np.dot(loss, self.w_h2_o.transpose()), deri_h2), 0).transpose()

        if self.name is "sigmoid":
            deri_h1 = np.multiply(self.h1, 1.0 - self.h1)
        if self.name is "relu":
            deri_h1 = copy.deepcopy(self.h1)
            deri_h1[deri_h1 > 0] = 1.0
            deri_h1[deri_h1 <= 0] = 0.0

        d_w_i_h1 = (float(1) / batch_size) * np.dot(x.transpose(), np.multiply(np.dot(np.multiply(np.dot(loss,
                    self.w_h2_o.transpose()), deri_h2), self.w_h1_h2.transpose()), deri_h1))

        d_b1 = (float(1) / batch_size) * np.sum(np.multiply(np.dot(np.multiply(np.dot(loss,
                    self.w_h2_o.transpose()), deri_h2), self.w_h1_h2.transpose()), deri_h1), 0).transpose()

        # update, SGD + Momentum
        self.vb1 = mu * self.vb1 - lr * d_b1
        self.b1 = self.b1 + self.vb1
        self.vb2 = mu * self.vb2 - lr * d_b2
        self.b2 = self.b2 + self.vb2
        self.vb3 = mu * self.vb3 - lr * d_b3
        self.b3 = self.b3 + self.vb3

        self.vw_i_h1 = mu * self.vw_i_h1 - lr * d_w_i_h1
        self.w_i_h1 += self.vw_i_h1
        self.vw_h1_h2 = mu * self.vw_h1_h2 - lr * d_w_h1_h2
        self.w_h1_h2 += self.vw_h1_h2
        self.vw_h2_o = mu * self.vw_h2_o - lr * d_w_h2_o
        self.w_h2_o += self.vw_h2_o

        return np.sum(np.sum(np.multiply(np.log(pred + config.eps), l))) * (float(1) / batch_size)

    def dropout(self, x):
        sample = np.random.binomial(n=1, p=config.dropout, size=x.shape)
        return np.multiply(x, sample)

    def lr_decay(self):
        """
        Learning rate decay
        :return: None
        """
        if self.lr > config.lr_threshold:
            self.lr *= 0.1

    def save_weight(self):
        """
        Save weights
        :return: None
        """
        if not os.path.exists(dir_path + '/weight'):
            os.mkdir(dir_path + '/weight')
        np.save(dir_path + '/weight/b1.npy', self.b1)
        np.save(dir_path + '/weight/b2.npy', self.b2)
        np.save(dir_path + '/weight/b3.npy', self.b3)
        np.save(dir_path + '/weight/w_i_h1.npy', self.w_i_h1)
        np.save(dir_path + '/weight/w_h1_h2.npy', self.w_h1_h2)
        np.save(dir_path + '/weight/w_h2_o.npy', self.w_h2_o)

    def load_weight(self):
        self.b1 = np.load(dir_path + '/weight/b1.npy')
        self.b2 = np.load(dir_path + '/weight/b2.npy')
        self.b3 = np.load(dir_path + '/weight/b3.npy')
        self.w_i_h1 = np.load(dir_path + '/weight/w_i_h1.npy')
        self.w_h1_h2 = np.load(dir_path + '/weight/w_h1_h2.npy')
        self.w_h2_o = np.load(dir_path + '/weight/w_h2_o.npy')

        # test mode
        self.name = config.activate
        self.dropout = 1

    def eval(self, x, y, data_type, save):
        """
        Evaluate performance
        x:input data
        y:input label
        data_type: train or test
        save: whether to save weight
        :return: None
        """
        pred = self.forward(x)
        pred_final = np.argmax(pred, axis=1)
        label = np.argmax(y, axis=1)
        t = (pred_final-label)
        num_true = len(t[t==0])

        if data_type is "train_data":
            print("Train data accuracy is {0}".format(float(num_true)/np.shape(x)[0]))

            if float(num_true)/np.shape(x)[0] >= 0.99:
                self.early_stop_time += 1

            if self.early_stop_time > config.early_stop_threshold:
                return True

        else:
            print("Test data accuracy is {0}".format(float(num_true)/np.shape(x)[0]))

            if save:
                cur_accu = float(num_true) / np.shape(x)[0]
                if cur_accu >= self.pre_accu:
                    self.save_weight()
                    print("Model saved!")
                    self.pre_accu = cur_accu  # keep best model
                    self.early_stop_time = 0

        return False



