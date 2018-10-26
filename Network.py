import numpy as np
import Config as config

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

        self.b1 = np.random.normal(mu, sigma, size=(1, self.num_h1))
        self.b2 = np.random.normal(mu, sigma, size=(1, self.num_h2))
        self.b3 = np.random.normal(mu, sigma, size=(1, self.num_output))


    def forward(self, input_data):
        pass

    def back_propagation(self):
        pass
    
    def eval(self):
        """
        Load weight
        :return: None
        """
        pass


