import numpy as np
from .Interfaces.INode import INode


class Node(INode):
    def __init__(self, nexts: dict, function_of_next: callable) -> None:
        super().__init__()
        self.__nexts = nexts
        self.__function_of_next = function_of_next
    
    def get_all(self):
        return self.__nexts

    def set_next(self, param: any, next: any):
        self.__nexts[param] = next

    def next(self, x: any) -> any:
        return self.__nexts[self.__function_of_next(x)]



