from abc import ABC, abstractmethod
from numpy import array


class IPredictor(ABC):

    @abstractmethod
    def predict(self, x: array) -> array:
        pass
