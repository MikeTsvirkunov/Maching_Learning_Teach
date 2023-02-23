import unittest
from unittest import mock
from Classes.Regression import Regression
from Classes.RegressionTrainer import RegressionTrainer
from Functions.errorFunctions import euclideError
from Functions.gradientDescentAlgorithm import gradientDescentAlgorithm
import numpy as np
import pandas as pd


class TestRegression(unittest.TestCase):
    def setUp(self):
        self.train_data = [[1, 2],
                           [2, 4], 
                           [4, 8],
                           [3, 6],
                           [6, 12]]
        self.check_data = [3, 6, 12, 9, 18]
        self.min_available_error = 0.1
        self.regressor = Regression()
        self.function_optimizer_error = lambda td, cd, w: gradientDescentAlgorithm(td=td, cd=cd, w=w, ferror=euclideError)
        self.trainer = RegressionTrainer(self.regressor, self.function_optimizer_error)
    
    def test_trainer(self):
        self.trainer.train(self.train_data, self.check_data)
        self.assertTrue(
            np.all(np.abs(self.regressor.predict(self.train_data) - np.array(self.check_data)) < self.min_available_error)
        )
        
        
if __name__ == '__main__':
    unittest.main()

