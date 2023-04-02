import numpy as np
from numpy import array
from .Interfaces.ITeacher import ITeacher
from .CartTreeClassifier import CartTreeClassifier
from .Node import Node
# from .NodeConteiner import NodeContainer


class CartTreeTeacher(ITeacher):
    def __init__(self, 
                 cart_tree_classifier: CartTreeClassifier, 
                 function_of_split: callable, 
                 function_of_searching_of_split: callable,
                 tree_size: int=-1) -> None:
        super().__init__()
        self.__cart_tree_classifier = cart_tree_classifier
        self.__function_of_split = function_of_split
        self.__function_of_searching_of_split = function_of_searching_of_split
        self.__tree_size = tree_size

    def __creation_of_tree(self, now_node: Node, x: np.array, y: np.array, count: int):
        f = lambda a :self.__function_of_split(a, self.__function_of_searching_of_split(x, y))
        true_node = Node({True: None, False: None}, f)
        false_node = Node({True: None, False: None}, f)
        split = f(x)
        self.__creation_of_tree(true_node, x[split], y[split], count-1)
        self.__creation_of_tree(true_node, x[np.logical_not(split)], y[np.logical_not(split)], count-1)
        # true_node.set_next(True, self.__creation_of_tree(true_node, x[f(x)], y[f(x)], count-1))
        # false_node.set_next(True, self.__creation_of_tree(true_node, x[not f(x)], y[not f(x)], count-1))
        now_node.set_next(True, true_node)
        now_node.set_next(False, false_node)


    def teach(self, X_train: array, y_train: array, colums_spec: iter = ...):
        f = lambda a :self.__function_of_split(a, self.__function_of_searching_of_split(X_train, y_train))
        self.__cart_tree_classifier.tree = Node({True: None, False: None}, f)
        self.__creation_of_tree(self.__cart_tree_classifier.tree, X_train, y_train, self.__tree_size)

                
