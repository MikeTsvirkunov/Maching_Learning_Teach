class KNumNeighborsClassifier:
    def __init__(self, k, distance) -> None:
        self.k = k
        self.distance = distance
        self.known_objects = list()
    
    def predict(self, x):
        for i in x:
            nearest_neighbors = sorted(self.known_objects, key=lambda a: self.distance(i, a[1]))[0:self.k]
        return nearest_neighbors
