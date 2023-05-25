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
        S = []
        for layer in self.__layers:
            # print('W: ', layer.get_weights().shape, layer.get_weights().tolist())
            # print('A: ', A[-1].shape, A[-1].tolist())
            S.append(layer.get_sumation(A[-1]))
            # print('S: ', S[-1].shape, S[-1].tolist())
            A.append(layer.activation_function(S[-1]))
            A[-1] = A[-1].reshape(1, A[-1].shape[0])
        return A, S

    def teach(self, X_train: array, y_train: array, colums_spec: iter = ...):
        for _ in range(self.__epochs):
            for X_batch, y_batch in self.__batch_spliter(X_train, y_train):
                for x, y in zip(X_batch, y_batch):
                    # print('\n\n' + '<'*20 + 'NEW_X' + '>'*20)
                    A, S = self.forward_propagation(x)
                    dL_dA = self.__derror_function(A[-1], y).reshape(1, len(A[-1]))
                    d_output = dL_dA
                    # print('A: ', A)
                    # print('S: ', S)
                    # print('dL/dA: ', self.__error_function(dL_dA[-1], y))
                    # print(len(S), len(A), len(A[:len(A)-1]), len(self.__layers))
                    for a, s, l in zip(reversed(A[:len(A)-1]), reversed(S), reversed(self.__layers)):
                        # print(f'shapes\ta: {a.shape}, s: {s.shape}')
                        df = l.dactivation_function(s)
                        print('df: ', df.shape, df.tolist())
                        print('d_output: ', d_output.shape, d_output.tolist())
                        print('a: ', a.shape, a.tolist())
                        print('s: ', s.shape, s.tolist())
                        print('W: ', l.get_weights().shape, l.get_weights().tolist())
                        print('B: ', l.get_dias().shape, l.get_dias().tolist())
                        print('\nres')
                        print('1) ', d_output.dot(df).shape, d_output.dot(df).tolist())
                        print('2.1) ', l.get_dsumation(a)[0].shape, l.get_dsumation(a)[0].tolist())
                        print('2.2) ', l.get_dsumation(a)[1].shape, l.get_dsumation(a)[1].tolist())
                    #     # print('\tnow d_output: ', d_output.tolist(), d_output.shape)
                    #     # print(f'd_output: {d_output.shape}, \n df.shape: {df.shape}, \n a.shape: {a.shape}, \n l.get_weights().shape: {l.get_weights().shape}, \n ap: {(ap.reshape(ap.shape[0], 1)).T.shape}')
                    #     print('prev_d_output: ', d_output.tolist(), d_output.shape)
                    #     # print(l.get_doutput((ap.reshape(ap.shape[0], 1))).shape)
                        # d_output = d_output.dot(df)
                        d_output = d_output.dot(df)
                        db = d_output.dot(l.get_dsumation(a)[1])
                        d_output = d_output.dot(l.get_dsumation(a)[0])
                        # print('3) ', d_output.shape, d_output.tolist())

                        # print('db: ', db.shape, db.tolist())
                    #     d_output = d_output.T.dot(df.reshape(df.shape[0], 1).T)
                    #     res = d_output.T.dot(a.reshape(a.shape[0], 1).T)
                    #     # d_dias = d_output * df
                    #     print('res: ', res.tolist(), res.shape)
                    #     # print('\td_b: ', d_dias, d_dias.shape)
                    #     # print('\tw: ', l.get_weights().tolist(), l.get_weights().shape)
                    #     # print('\tb: ', l.get_dias().tolist(), l.get_dias().shape)
                    #     # print(((self.__step * d_output).T / (l.get_weights().shape[0] * l.get_weights().shape[1])).tolist(), ((self.__step * d_output).T / (l.get_weights().shape[0] * l.get_weights().shape[1])).shape)
                    #     print('w: ', l.get_weights().tolist(), l.get_weights().shape)
                    #     print()
                        w1 = np.array([((self.__step * d_output) / (l.get_weights().shape[0] * l.get_weights().shape[1]))[0] for _i in l.get_weights()])
                        print('W_Step: ', w1.shape, w1.tolist())
                        l.set_weights(l.get_weights() - w1)
                        # print('='*40)
                        # l.set_weights(l.get_weights() - (self.__step * d_output) / (l.get_weights().shape[0] * l.get_weights().shape[1]))
                        # print('b', l.get_dias(), self.__step * d_output)
                        l.set_dias(l.get_dias() - self.__step * db.T / (l.get_dias().shape[0] * l.get_dias().shape[1]))
                        print('='*40)
        return self.__layers
    



                    

                    

                



        
