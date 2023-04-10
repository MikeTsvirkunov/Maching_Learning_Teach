import numpy as np


def classic_split(mass: np.array, lvl: any, j: int):
    return mass.T[j] >= lvl


def bootstrap_split_of_train_data(x, y, n):
    indexes = range(y.shape[0])
    split_x = list()
    split_y = list()
    for _ in range(n):
        i = np.random.choice(indexes, size=300, replace=True)
        split_x.append(x[i])
        split_y.append(y[i])
    return split_x, split_y

