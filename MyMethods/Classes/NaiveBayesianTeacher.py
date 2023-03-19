import numpy as np

class NaiveBayesianTeacher:
    def __init__(self, classifier):
        self.classifier = classifier
    
    def teach(self, X_train: np.array, y_train: np.array, spreading_functions: list):
        self.classifier.spreading_functions = spreading_functions
        self.classifier.X_train = X_train
        self.classifier.y_train = y_train
