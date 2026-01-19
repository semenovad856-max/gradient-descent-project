import numpy as np
from gradient_descent import GradientDescent

class BatchGD(GradientDescent):
    """Пакетный градиентный спуск"""
    
    def compute_gradient(self, X, y, theta):
        m = X.shape[0]
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * (X.T @ errors)
        return gradient.flatten()
