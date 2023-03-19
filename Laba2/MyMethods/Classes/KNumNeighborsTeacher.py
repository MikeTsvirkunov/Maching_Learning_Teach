import numpy as np

class KNumNeighborsTeacher:
    def __init__(self, classifier):
        self.classifier = classifier
    
    def teach(self, X_train: np.array, y_train: np.array):
        self.classifier.X_train = X_train
        self.classifier.y_train = y_train
