import numpy as np
from .Interfaces.INode import INode
from Node import Node
from .Interfaces.IContainer import IContainer


class NodeContainer(INode, IContainer):
    def __init__(self, nexts: dict, function_of_next: callable, data: np.array) -> None:
        super().__init__()
        self.__nexts = nexts
        self.__function_of_next = function_of_next
        self.__data =data
    
    def get_data(self, param: any, next: any):
        return self.__data

    def set_next(self, param: any, next: any):
        self.__nexts[param] = next

    def next(self, x: any) -> any:
        return self.__nexts[self.__function_of_next(x)]
    
    def to_node(self):
        self.__class__ = Node
        self.__init__(self.__nexts, self.__function_of_next)
