import numpy as np
from numpy import array
from .Interfaces.ITeacher import ITeacher
from .CartTreeClassifier import CartTreeClassifier
from .Node import Node


class CartTreeTeacher(ITeacher):
    def __init__(self, cart_tree_classifier: CartTreeClassifier, 
                 function_of_split: callable) -> None:
        super().__init__()
        self.__cart_tree_classifier = cart_tree_classifier
        self.__function_of_split = function_of_split


    def teach(self, X_train: array, y_train: array, colums_spec: iter = ...):
        self.__cart_tree_classifier.tree = Node({True: None, False: None}, 
                                                lambda a: self.__function_of_split(a, X_train.T[0].mean()))
        
        for i in range(X_train.shape[0]):
            timed_node = self.__cart_tree_classifier.tree
            for j in range(X_train[i].shape[0]):
                if type(timed_node.next(X_train[i][j])) is Node:
                    print('Node:', timed_node.next(X_train[i][j]), i, j)
                    timed_node = timed_node.next(X_train[i][j])
                elif timed_node.next(X_train[i][j]) == None:
                    print(f'none: y={y_train[i]}, x={X_train[i][j]}, f(x)={self.__function_of_split(X_train[i][j], X_train.T[i].mean())}, i={i}, j={j}')
                    timed_node.set_next(self.__function_of_split(X_train[i][j], X_train.T[i].mean()), 
                                        y_train[i]
                                        if j > X_train[i].shape[0]-2 else 
                                        Node({True: None, False: None}, lambda a: self.__function_of_split(a, X_train.T[i].mean())))
                    timed_node = timed_node.next(X_train[i][j])
                
