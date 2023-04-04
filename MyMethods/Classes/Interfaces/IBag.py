from abc import ABC, abstractmethod
from numpy import array


class IBag(ABC):

    @abstractmethod
    def get_bag(self) -> any:
        pass
