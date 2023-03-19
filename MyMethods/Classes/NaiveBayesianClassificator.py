import numpy as np
from collections import Counter


class NaiveBayesianClassificator:
    def __init__(self, function_of_priority) -> None:
        self.function_of_priority = function_of_priority
        self.X_train = None
        self.y_train = None
        self.spreading_functions = None
    
    def predict(self, x) -> np.array:
        k_true = list()
        k_false = list()
        p_true = sum(self.y_train) / self.y_train.shape[0]
        p_false = 1 - p_true
        # print(p_true, p_false)
        for j in x:
            z_true = list()
            z_false = list()

            for value, func, train_values_true, train_values_false in zip(j, self.spreading_functions, self.X_train[self.y_train == True].T, self.X_train[self.y_train == False].T):
                z_true.append(func(value, train_values_true))
                z_false.append(func(value, train_values_false))
            k_false.append(np.prod(np.array(z_false).T) * p_false)
            k_true.append(np.prod(np.array(z_true).T) * p_true)
        return self.function_of_priority(np.array(k_true) - np.array(k_false))

