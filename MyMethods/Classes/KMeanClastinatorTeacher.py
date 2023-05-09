import numpy as np
from numpy import array
from .Interfaces.ITraining import ITraining
from .Interfaces.IPredictor import IPredictor


class KMeanClastinatorTeacher(ITraining):
    def __init__(self, 
                 centers_of_mass: iter, 
                 k_neighbors: int, 
                 metric: callable, 
                 mass_center_searcher: callable, 
                #  classifier: IPredictor
                 ) -> None:
        self.metric = metric
        self.k_neighbors = k_neighbors
        self.mass_center_searcher = mass_center_searcher
        self.centers_of_mass = centers_of_mass
        # self.classifier = classifier
    
    def train(self, X: array) -> any:
        for _ in range(3):
            x = []
            y = np.array([])
            train_x = np.array(X, copy=True)
            # print(train_x.shape)
            for i in range(self.centers_of_mass.shape[0]):
                obj_indexis = np.argsort(self.metric(
                    train_x, self.centers_of_mass[i]))
                # print(train_x[obj_indexis][0:self.k_neighbors])
                x += train_x[obj_indexis][0:self.k_neighbors].tolist()
                y = np.append(y, [i]*self.k_neighbors)
                train_x = train_x[obj_indexis][self.k_neighbors:]

            classifier = KNumNeighborsClassifier(k=self.k_neighbors, distance=self.metric, function_of_priority=max_count_class)
            classifier.X_train = np.array(x)
            classifier.y_train = y
            # print(x)
            # print(y)
            clastered_x = classifier.predict(X)
            for c in range(self.num_of_clasters):
                self.centers_of_mass[c] = self.mass_center_searcher(
                    X[clastered_x == c])
            # print(self.centers_of_mass)
        return X, clastered_x
