import numpy as np

class RegressionTrainer():
    def __init__(self, regression, error_function):
        self.regression = regression
        self.error_function = error_function

    def train(self, train_data, check_data):
        self.regression.weights = np.random.rand(len(train_data[0]))
        td = np.array(train_data)
        cd = np.array(check_data)
        self.regression.weights = self.error_function(td, cd, self.regression.weights)
