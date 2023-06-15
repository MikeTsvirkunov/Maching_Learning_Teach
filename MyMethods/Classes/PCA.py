import numpy as np


class PCA:

    def __init__(self, covariance_counter, eigen_searcher):
        self.covariance_counter = covariance_counter
        self.eigen_searcher = eigen_searcher
        self.eigenvector_in_use = None
    
    def get_first_n(self, X, n):
        eigen_vars, eigen_vectors = self.eigen_searcher(self.covariance_counter(X))
        sorted_eigenvectors = eigen_vectors[:, np.argsort(eigen_vars)[::-1]]
        self.eigenvector_in_use = sorted_eigenvectors[:, 0:n]
        return np.dot(self.eigenvector_in_use.transpose(), X.transpose()).transpose()
