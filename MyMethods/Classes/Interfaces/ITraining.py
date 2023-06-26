from abc import ABC, abstractmethod
from numpy import array


class ITraining(ABC):

    @abstractmethod
    def train(self, X: array) -> any:
        pass
