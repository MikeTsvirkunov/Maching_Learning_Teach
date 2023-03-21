import numpy as np
from .Interfaces.ITeacher import ITeacher


class NaiveBayesianTeacher(ITeacher):
    def __init__(self, classifier):
        self.classifier = classifier
    
    def teach(self, X_train: np.array, y_train: np.array, colums_spec: iter=[]):
        self.classifier.X_train = X_train
        self.classifier.y_train = y_train
        self.classifier.spreading_functions = colums_spec
