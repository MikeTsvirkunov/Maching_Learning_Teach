import numpy as np
from numpy import array
from .Interfaces.ITraining import ITraining
from .Interfaces.IPredictor import IPredictor
from Classes.KNumNeighborsClassifier import KNumNeighborsClassifier
from Classes.KNumNeighborsTeacher import KNumNeighborsTeacher
# Кап теорема


class DBSCANTrainer(ITraining):
    def __init__(self,
                 e: float,
                 metric: callable,
                 noise_border: int,
                 ) -> None:
        self.metric = metric
        self.e = e
        self.noise_border = noise_border
        self.clasters_dots = dict()
        self.clasters_dots['noise'] = []

    def train(self, X: array) -> any:
        x_train = np.array(X, copy=True)
        self.counter = np.ones(x_train.shape[0], dtype=bool)
        c = 0
        while x_train[self.counter].shape[0] > 0:
            # print('x_train[self.counter]', x_train[self.counter].shape, x_train[self.counter].tolist())
            self.clasters_dots[c] = x_train[self.counter][0:1]
            self.__search(x_train[self.counter][0], c, x_train)
            c+=1
            # print(self.counter)
        return self.clasters_dots

    def __search(self, dot, c, x):
        res = np.logical_and(self.metric(dot, x) <= self.e, self.counter)
        # print('self.clasters_dots', self.clasters_dots)
        # print('res', res.shape, res.sum(), res.tolist())
        self.clasters_dots[c] = np.append(self.clasters_dots[c], x[np.logical_and(res, self.counter)], axis=0)
        self.counter = np.logical_not(np.logical_or(res, np.logical_not(self.counter)))
        # print('self.counter', self.counter.shape, self.counter.sum(), self.counter.tolist())
        for d in x[np.logical_and(res, self.counter)]:
            return self.__search(d, c, x, self.counter)
        
# A -> B = A + !B
# 1 1 1 1 1 1 1 1 
# 0 1 1 0 1 0 1 0
# 0 1 1 0 1 0 1 0
# 1 0 0 1 0 1 0 1

# 1 0 0 1 0 1 0 1
# 1 0 1 0 1 0 1 1
# 1 0 0 0 0 0 0 1 
# 0 0 0 1 0 1 0 0

