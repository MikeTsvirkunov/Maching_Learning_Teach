import numpy as np
from numpy import array
from .Interfaces.ITeacher import ITeacher
from .BaggingContainer import BaggingContainer
from collections import Counter
from .Interfaces.IBag import IBag


class BaggingTeacher(ITeacher, IBag):
    def __init__(self, 
                 begging_container: BaggingContainer,
                 teachers: iter,
                 teachers_params: iter,
                 split_function: callable,) -> None:
        super().__init__()
        self.__begging_container = begging_container
        self.__teachers = teachers
        self.__teachers_params = teachers_params
        self.__split_function = split_function

    def get_bag(self) -> any:
        return self.__teachers

    def teach(self, X_train: array, y_train: array, colums_spec: iter = ...):
        predictors_bag = self.__begging_container.get_bag()
        x, y = self.__split_function(X_train, y_train, len(predictors_bag))

        for train_data, result_data, teacher, teacher_params, predictor in zip(x, y, self.__teachers, self.__teachers_params, predictors_bag):
            params = [predictor] + teacher_params
            x = teacher(*params)
            x.teach(train_data, result_data, colums_spec=colums_spec)
