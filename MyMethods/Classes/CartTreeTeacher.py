import numpy as np
from numpy import array
from .Interfaces.ITeacher import ITeacher
from .CartTreeClassifier import CartTreeClassifier
from .Node import Node
from collections import Counter


class CartTreeTeacher(ITeacher):
    def __init__(self, 
                 cart_tree_classifier: CartTreeClassifier, 
                 function_of_split: callable, 
                 function_of_searching_of_split: callable,
                 function_of_appling_stop: callable,
                 tree_size: int=-1) -> None:
        super().__init__()
        self.__cart_tree_classifier = cart_tree_classifier
        self.__function_of_split = function_of_split
        self.__function_of_searching_of_split = function_of_searching_of_split
        self.__function_of_appling_stop = function_of_appling_stop
        self.__tree_size = tree_size

    def creation_of_tree(self, x: np.array, y: np.array, count: int):
        p = self.__function_of_searching_of_split(x, y)
        # f = lambda a :self.__function_of_split(a, p)
        def f(a):
            return self.__function_of_split(a, p)
        if self.__function_of_appling_stop(count, p[0]):
            split = f(x)
            return Node({True: self.creation_of_tree(x[self.__function_of_split(x, p)], y[self.__function_of_split(x, p)], count-1),
                         False: self.creation_of_tree(x[np.logical_not(self.__function_of_split(x, p))], y[np.logical_not(self.__function_of_split(x, p))], count-1)}, f)
        else:
            return Counter(y).most_common(1)[0][0]


    def teach(self, X_train: array, y_train: array, colums_spec: iter = ...):
        p = self.__function_of_searching_of_split(X_train, y_train)
        # f = lambda a: 
        def f(a):
            return self.__function_of_split(a, p)
        split = f(X_train)
        self.__cart_tree_classifier.tree = Node({True: self.creation_of_tree(X_train[self.__function_of_split(X_train, p)], y_train[self.__function_of_split(X_train, p)], self.__tree_size),
                                                 False: self.creation_of_tree(X_train[np.logical_not(self.__function_of_split(X_train, p))], 
                                                                              y_train[np.logical_not(self.__function_of_split(X_train, p))], self.__tree_size)},
                                                 f)

                
