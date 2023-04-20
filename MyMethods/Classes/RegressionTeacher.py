import numpy as np
from .Interfaces.ITeacher import ITeacher


class RegressionTeacher(ITeacher):
    def __init__(self, regression, error_function):
        self.regression = regression
        self.error_function = error_function

    def teach(self, train_data, check_data, colums_spec: iter = []):
        self.regression.weights = np.random.rand(len(train_data[0]))
        td = np.array(train_data)
        cd = np.array(check_data)
        self.regression.weights = self.error_function(td, cd, self.regression.weights)
