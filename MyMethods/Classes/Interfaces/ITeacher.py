from abc import ABC, abstractmethod
from numpy import array


class ITeacher(ABC):

    @abstractmethod
    def teach(self, X_train: array, y_train: array, colums_spec: iter=[]):
        pass
