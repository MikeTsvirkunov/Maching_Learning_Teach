import unittest
from Classes.Regression import Regression
from Classes.RegressionTrainer import RegressionTrainer
from Functions.lossFunctions import euclideError, euclideErrorGradient
from Classes.Polinomizer import Polinomizer
from Functions.gradientDescentAlgorithm import gradientDescentAlgorithm
import numpy as np
import pandas as pd


class TestRegression(unittest.TestCase):
    def setUp(self):
        self.train_data = [[1, 2],
                           [2, 4], 
                           [4, 8],
                           [3, 6],
                           [6, 12],
                           [1, 3],
                           [2, 3]]
        self.check_data = [3, 6, 12, 9, 18, 4, 5]
        self.check_data2 = [5, 8, 14, 11, 20, 6, 7]
        self.check_data3 = [7, 20, 80, 45, 180, 10, 13]
        self.min_available_error = 1
        self.regressor = Regression()
        self.min_available_error = 0.01
        self.size_of_step = 0.001
        self.function_optimizer_error = lambda td, cd, w: gradientDescentAlgorithm(
            td=td, cd=cd, w=w, lost_function=euclideError, gradient=euclideErrorGradient, nsteps=100000, merror=self.min_available_error, step=self.size_of_step)
        self.trainer = RegressionTrainer(self.regressor, self.function_optimizer_error)
    
    def test_trainer(self):
        self.trainer.train(self.train_data, self.check_data)
        self.assertTrue(
            np.all(np.abs(self.regressor.predict(self.train_data) - np.array(self.check_data)) < 1)
        )
    
    def test_trainer1(self):
        self.trainer.train(Polinomizer(self.train_data, 1).polinomize(), self.check_data2)
        self.assertTrue(
            np.all(np.abs(self.regressor.predict(Polinomizer(self.train_data, 1).polinomize()) -
                   np.array(self.check_data2)) < 2)
        )
    
    def test_trainer2(self):
        self.min_available_error = -100
        self.size_of_step = 0.0001
        self.trainer.train(Polinomizer(
            self.train_data, 2).polinomize(), self.check_data3)
        self.assertTrue(
            np.all(np.abs(self.regressor.predict(Polinomizer(self.train_data, 2).polinomize()) -
                          np.array(self.check_data3)) < 10)
        )
        
        
if __name__ == '__main__':
    unittest.main()

