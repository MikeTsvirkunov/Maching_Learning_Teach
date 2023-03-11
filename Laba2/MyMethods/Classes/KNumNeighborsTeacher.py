class KNumNeighborsTeacher:
    def __init__(self, classifier):
        self.classifier = classifier
    
    def teach(self, X_train, y_train):
        for i in zip(X_train, y_train):
            self.classifier.known_objects.append(i)
