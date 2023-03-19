import numpy as np
from .Interfaces.ITeacher import ITeacher
from numpy import array


class KNumNeighborsTeacher(ITeacher):

    def __init__(self, classifier) -> None:
        super(ITeacher, self).__init__()
        self.classifier = classifier
    
    def teach(self, X_train: array, y_train: array):
        self.classifier.X_train = X_train
        self.classifier.y_train = y_train
