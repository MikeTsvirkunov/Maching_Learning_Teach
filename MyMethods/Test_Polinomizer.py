import unittest
from Classes.Polinomizer import Polinomizer


class TestPolinomizer(unittest.TestCase):
    def setUp(self):
        self.data = [[1, 2], 
                     [4, 5],
                     [6, 7]]
        self.power1 = 1
        self.power0 = 0
        self.power2 = 2
    
    def test_polinomize_with_power_biger_than_two(self):
        polinomizer = Polinomizer(self.data, self.power2)
        self.assertListEqual(polinomizer.polinomize(), [[1, 1, 1, 1, 2, 4], [1, 4, 16, 1, 5, 25], [1, 6, 36, 1, 7, 49]])
    
    
    def test_polinomize_with_power_of_one(self):
        polinomizer = Polinomizer(self.data, self.power1)
        self.assertListEqual(polinomizer.polinomize(), [[1, 1, 1, 2], [1, 4, 1, 5], [1, 6, 1, 7]])
    
    
    def test_polinomize_with_power_of_one(self):
        polinomizer = Polinomizer(self.data, self.power0)
        self.assertListEqual(polinomizer.polinomize(), [[1, 1], [1, 1], [1, 1]])

       
if __name__ == '__main__':
    unittest.main()

