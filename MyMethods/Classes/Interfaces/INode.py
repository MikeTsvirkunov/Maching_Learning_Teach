from abc import ABC, abstractmethod
from numpy import array


class INode(ABC):
    @abstractmethod
    def next(self, x: any) -> any:
        pass