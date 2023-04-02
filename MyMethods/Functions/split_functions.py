from numpy import array


def classic_split(mass: array, lvl: any, j: int):
    return mass.T[j] >= lvl