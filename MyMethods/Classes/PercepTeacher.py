import numpy as np
from .Interfaces.ITeacher import ITeacher
from numpy import array
from .Layer import Layer
# from Perceptron import Perceptron


class PerceptronTeacher(ITeacher):

    def __init__(self,
                 layers: iter,
                 step: float,
                 batch_spliter: callable,
                 epochs: int,
                 error_function: callable,
                 derror_function: callable) -> None:
        super(ITeacher, self).__init__()
        self.__layers = layers
        self.__step = step
        self.__epochs = epochs
        self.__batch_spliter = batch_spliter
        self.__error_function = error_function
        self.__derror_function = derror_function


    def forward_propagation_m(self, X_batch: np.array):
        As = list()
        for x in X_batch:
            A = [np.array(x, copy=True)]
            for layer in self.__layers:
                A.append(layer.get_output(As[-1]))
            As.append(A)
        return np.array(As)
    

    def forward_propagation(self, x: np.array):
        A = [np.array(x, copy=True).reshape(1, x.shape[0])]
        for layer in self.__layers:
            A.append(layer.get_activation_function()(np.dot(A[-1], layer.get_weights().T) + layer.get_dias())/A[-1].shape[1])
            A[-1] = A[-1].reshape(1, A[-1].shape[1])
        return A

    def teach(self, X_train: array, y_train: array, colums_spec: iter = ...):
        for _ in range(self.__epochs):
            for X_batch, y_batch in self.__batch_spliter(X_train, y_train):
                for x, y in zip(X_batch, y_batch):
                    A = self.forward_propagation(x)
                    dL_dA = self.__derror_function(A[-1], y).reshape(1, len(A[-1]))
                    d_output = np.array(dL_dA, copy=True)
                    # print('A: ', A)
                    # print('dL/dA: ', dL_dA.shape, dL_dA.tolist())
                    for a, l in zip(reversed(A[:len(A)-1]), reversed(self.__layers)):
                        # x2 = np.dot(d_output.T, l.get_dactivation_function()(a))
                        print('d_output', d_output.shape)
                        print('l.weights', l.get_weights().shape)
                        print('l.dias', l.get_dias().shape)
                        print('l.get_dactivation_function()(a)',l.get_dactivation_function()(a).shape)
                        # print('x2', x2.shape, x2.tolist())
                        d_output = np.dot(d_output.T, l.get_dactivation_function()(a))
                        l.set_weights(l.get_weights() - d_output)

                        print('='*40)
        return self.__layers
    



                    

                    

                



        
