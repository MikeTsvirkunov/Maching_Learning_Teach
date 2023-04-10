from abc import ABC, abstractmethod
from numpy import array


class INode(ABC):

    @abstractmethod
    def set_next(self, param: any, next: any):
        pass

    @abstractmethod
    def next(self, x: any) -> any:
        pass