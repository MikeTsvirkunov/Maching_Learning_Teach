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
                    for a, ap, l in zip(reversed(outputs[:len(outputs)-1]), outputs[1:len(outputs)], reversed(self.__layers)):
                        print('a: ', a)
                        df = l.get_doutput(ap.reshape(ap.shape[0], 1))
                        print('df: ', df.tolist(), df.shape)
                        # print('\tnow d_output: ', d_output.tolist(), d_output.shape)
                        # print(f'd_output: {d_output.shape}, \n df.shape: {df.shape}, \n a.shape: {a.shape}, \n l.get_weights().shape: {l.get_weights().shape}, \n ap: {(ap.reshape(ap.shape[0], 1)).T.shape}')
                        print('prev_d_output: ', d_output.tolist(), d_output.shape)
                        # print(l.get_doutput((ap.reshape(ap.shape[0], 1))).shape)
                        # d_output = np.array([(d_output.T.dot(df.T) * a_i) for a_i in a])
                        d_output = d_output.T.dot(df.reshape(df.shape[0], 1).T)
                        print('d_output: ', d_output.tolist(), d_output.shape)
                        res = d_output.T.dot(a.reshape(a.shape[0], 1).T)
                        # d_dias = d_output * df
                        print('res: ', res.tolist(), res.shape)
                        # print('\td_b: ', d_dias, d_dias.shape)
                        # print('\tw: ', l.get_weights().tolist(), l.get_weights().shape)
                        # print('\tb: ', l.get_dias().tolist(), l.get_dias().shape)
                        # print(((self.__step * d_output).T / (l.get_weights().shape[0] * l.get_weights().shape[1])).tolist(), ((self.__step * d_output).T / (l.get_weights().shape[0] * l.get_weights().shape[1])).shape)
                        print('w: ', l.get_weights().tolist(), l.get_weights().shape)
                        print()
                        l.set_weights(l.get_weights() - (self.__step * res) / (l.get_weights().shape[0] * l.get_weights().shape[1]))
                        # print('b', l.get_dias(), self.__step * d_output)
                        l.set_dias(l.get_dias() - self.__step * df / (l.get_dias().shape[0] * l.get_dias().shape[1]))
                        # print('='*30)
        return self.__layers
    



                    

                    

                



        
