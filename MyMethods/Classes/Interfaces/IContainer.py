from abc import ABC, abstractmethod
from numpy import array


class IContainer(ABC):

    @abstractmethod
    def get_data(self, param: any, next: any):
        pass