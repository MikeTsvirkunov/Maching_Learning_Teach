import numpy as np
class KNumNeighborsClassifier:
    def __init__(self, k, distance) -> None:
        self.k = k
        self.distance = distance
        self.known_objects = list()
    
    def predict(self, x):
        # nearest_neighbors = np.array(map(lambda a: np.inf, range(self.k)))
        nearest_neighbors = list()
        for i in x:
            for j in self.known_objects:
                nearest_neighbors.append(j)
                sorted()
                # nearest_neighbors = np.delete(np.insert(nearest_neighbors, np.argmax(np.insert(nearest_neighbors>self.distance(i, j), self.k, True)), j), self.k)  
        return nearest_neighbors

        