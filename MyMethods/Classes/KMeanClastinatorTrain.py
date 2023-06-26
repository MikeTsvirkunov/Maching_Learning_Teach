import numpy as np
from numpy import array
from .Interfaces.ITraining import ITraining
from .Interfaces.IPredictor import IPredictor
from Classes.KNumNeighborsClassifier import KNumNeighborsClassifier
from Classes.KNumNeighborsTeacher import KNumNeighborsTeacher
# Кап теорема


class KMeanClastinatorTrain(ITraining):
    def __init__(self, 
                 centers_of_mass: np.ndarray, 
                 k_neighbors: int, 
                 metric: callable, 
                 mass_center_searcher: callable,
                 function_of_priority: callable,
                 n_steps: int
                 ) -> None:
        self.metric = metric
        self.k_neighbors = k_neighbors
        self.function_of_priority = function_of_priority
        self.mass_center_searcher = mass_center_searcher
        self.centers_of_mass = centers_of_mass,
        self.centers_of_mass = np.array(self.centers_of_mass[0], copy=True)
        self.n_steps = n_steps
        self.classifier = KNumNeighborsClassifier(k=self.k_neighbors,
                                                  distance=self.metric,
                                                  function_of_priority=self.function_of_priority)
        self.teacher = KNumNeighborsTeacher(classifier=self.classifier)
    
    def train(self, X: array) -> any:
        self.teacher.teach(X_train=self.centers_of_mass, 
                           y_train=np.arange(self.centers_of_mass.shape[0]))
        for _ in range(self.n_steps):
            clastered_x = self.classifier.predict(X)
            u = np.unique(clastered_x)
            for c in range(self.centers_of_mass.shape[0]):
                if c in u:
                    self.centers_of_mass[c] = self.mass_center_searcher(X[c == clastered_x])
            self.classifier.X_train = np.array(self.centers_of_mass, copy=True)
            # self.teacher.teach(X_train=self.centers_of_mass,
            #                    y_train=np.arange(self.centers_of_mass.shape[0]))
        # print(self.centers_of_mass, self.classifier.X_train)
        return self.classifier
