import numpy as np

class TFiDF:
    def __init__(self, database):
        self.database = database

    def vectorize(self, w, wp):
        return sum(w == wp) / len(wp) * (1 + (np.log((len(self.database) + 1) / (sum([1 for exp in self.database if w in exp]) + 1))))

