import numpy as np


class PCA:

    def __init__(self, covariance_counter, eigen_searcher):
        self.covariance_counter = covariance_counter
        self.eigen_searcher = eigen_searcher
    
    def get_first_n(self, X, n):
        eigen_values, eigen_vectors = self.eigen_searcher(self.covariance_counter(X))
        sorted_eigenvectors = eigen_vectors[:, np.argsort(eigen_values)[::-1]]
        eigenvector_subset = sorted_eigenvectors[:, 0:n]
        return np.dot(eigenvector_subset.transpose(), X.transpose()).transpose()
