import numpy as np
import Config as config
from Utils import Function


F = Function()

class Network(object):
    """
    Network
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
        # self.w_i_h1 = np.random.normal(size=(self.num_input, self.num_h1))
        # self.w_h1_h2 = np.random.normal(size=(self.num_h1, self.num_h2))
        # self.w_h2_o = np.random.normal(size=(self.num_h2, self.num_output))

        self.b1 = np.random.normal(mu, sigma, size=(1, self.num_h1))
        self.b2 = np.random.normal(mu, sigma, size=(1, self.num_h2))
        self.b3 = np.random.normal(mu, sigma, size=(1, self.num_output))

        self.pred = None
        self.h1 = None
        self.h2 = None

        self.batch_size = config.batch_size
        self.lr = config.learning_rate

    def forward(self, input_data):
        self.h1 = F.sigmoid(np.dot(input_data, self.w_i_h1) + self.b1)  # 64*784 784*300 64*300
        self.h2 = F.sigmoid(np.dot(self.h1, self.w_h1_h2) + self.b2)  # 64*300 300*100 64*100
        o = np.dot(self.h2, self.w_h2_o) + self.b3  # 64*100 100*10 64*10

        rep = np.tile(np.max(o, axis=1), (self.num_output, 1))
        temp_o = o - rep.transpose()
        pred = np.exp(temp_o)/(np.tile(np.sum(np.exp(temp_o), axis=1), (self.num_output, 1)).transpose())

        # pred = np.exp(o) / (np.tile(np.sum(np.exp(o), axis=1), (self.num_output, 1)).transpose())
        return pred

    def back_propagation(self, x, l, pred):
        batch_size = self.batch_size
        lr = self.lr

        loss = pred - l  # 64*10

        d_w_h2_o = (float(1) / batch_size) * np.dot(self.h2.transpose(), loss)  # 100*10
        d_b3 = (float(1) / batch_size) * np.sum(loss, 0).transpose()

        deri_h2 = np.multiply(self.h2, 1.0 - self.h2)
        d_w_h1_h2 = (float(1) / batch_size) * np.dot(self.h1.transpose(),
                                              np.multiply(np.dot(loss, self.w_h2_o.transpose()), deri_h2))
        d_b2 = (float(1) / batch_size) * np.sum(np.multiply(np.dot(loss, self.w_h2_o.transpose()), deri_h2), 0).transpose()

        deri_h1 = np.multiply(self.h1, 1.0 - self.h1)
        d_w_i_h1 = (float(1) / batch_size) * np.dot(x.transpose(), np.multiply(np.dot(np.multiply(np.dot(loss,
                    self.w_h2_o.transpose()), deri_h2), self.w_h1_h2.transpose()), deri_h1))

        d_b1 = (float(1) / batch_size) * np.sum(np.multiply(np.dot(np.multiply(np.dot(loss,
                    self.w_h2_o.transpose()), deri_h2), self.w_h1_h2.transpose()), deri_h1), 0).transpose()

        # update
        self.b1 = self.b1 - lr * d_b1
        self.b2 = self.b2 - lr * d_b2
        self.b3 = self.b3 - lr * d_b3

        self.w_h2_o = self.w_h2_o - lr * d_w_h2_o
        self.w_h1_h2 = self.w_h1_h2 - lr * d_w_h1_h2
        self.w_i_h1 = self.w_i_h1 - lr * d_w_i_h1

        return np.sum(np.sum(np.multiply(np.log(pred), l))) * (float(1) / batch_size)

    def lr_decay(self):
        if self.lr > config.lr_threshold:
            self.lr *= 0.1

    def eval(self, x, y, data_type):
        """
        Evaluate performance
        :return: None
        """
        pred = self.forward(x)
        pred_final = np.argmax(pred, axis=1)
        label = np.argmax(y, axis=1)
        t = (pred_final-label)
        num_true = len(t[t==0])
        if data_type is "train_data":
            print("Train data accuracy is {0}".format(float(num_true)/np.shape(x)[0]))
        else:
            print("Test data accuracy is {0}".format(float(num_true)/np.shape(x)[0]))



