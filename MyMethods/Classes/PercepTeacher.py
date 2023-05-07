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
        A = [np.array(x, copy=True)]
        for layer in self.__layers:
            A.append(layer.get_output(A[-1]))
        return A



    def teach(self, X_train: array, y_train: array, colums_spec: iter = ...):
        for _ in range(self.__epochs):
            for X_batch, y_batch in self.__batch_spliter(X_train, y_train):
                for x, y in zip(X_batch, y_batch):
                    # print('\n\n' + '<'*20 + 'NEW_X' + '>'*20)
                    outputs = self.forward_propagation(x)
                    d_output = self.__derror_function(outputs[-1], y).reshape(1, len(outputs[-1]))
                    # print('All', outputs)
                    # print('err: ', self.__error_function(outputs[-1], y))
                    for a, l in zip(reversed(outputs[:len(outputs)-1]), reversed(self.__layers)):
                        # print('\toutput_i: ', a)
                        df = l.get_output(a)
                        # print('\tdf: ', df.tolist(), df.shape)
                        # print('\tnow d_output: ', d_output.tolist(), d_output.shape)
                        d_output = np.array([(d_output.T.dot(df.T) * a_i) for a_i in a])
                        # d_dias = d_output * df
                        # print('\td_output: ', d_output.tolist(), d_output.shape)
                        # print('\td_b: ', d_dias, d_dias.shape)
                        # print('\tw: ', l.get_weights().tolist(), l.get_weights().shape)
                        # print('\tb: ', l.get_dias().tolist(), l.get_dias().shape)
                        l.set_weights(
                            l.get_weights() - (self.__step * d_output).T / (l.get_weights().shape[0] * l.get_weights().shape[1]))
                        # l.set_dias(l.get_dias() - self.__step * d_output)
                        # print('='*30)




        return self.__layers


                    

                    

                



        
