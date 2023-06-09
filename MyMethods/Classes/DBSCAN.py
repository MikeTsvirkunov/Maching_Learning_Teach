import numpy as np
from .Interfaces.IPredictor import IPredictor


class DBSCAN(IPredictor):
    def __init__(self, e, clasters_dots, metric) -> None:
        self.e = e
        self.clasters_dots = clasters_dots
        self.metric = metric

    def predict(self, x: np.array):
        k = list()
        for j in x:
            for c in self.clasters_dots:
                if self.e >= self.metric(j, self.clasters_dots[c]):
                    k.append(c)
                    break
        return np.array(k)
