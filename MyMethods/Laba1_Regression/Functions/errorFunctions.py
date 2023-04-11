import numpy as np


def MAE(predicted, real):
    return np.sum(np.abs(predicted - real))/real.shape[0]


def MSE(predicted, real):
    return np.sum((real-predicted)**2)/real.shape[0]


def RMSE(predicted, real):
    return np.sqrt(MSE(predicted, real))


def MAPE(predicted, real):
    return np.sum(np.abs((predicted - real) / real))/real.shape[0]


def R2(predicted, real):
    return 1 - (MSE(predicted, real) / np.sum((real - np.mean(real))**2))
