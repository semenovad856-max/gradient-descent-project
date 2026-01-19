import numpy as np
from gradient_descent import GradientDescent

class MiniBatchGD(GradientDescent):
    """Мини-пакетный градиентный спуск"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, batch_size=32):
        super().__init__(learning_rate, max_iter, tol)
        self.batch_size = batch_size
    
    def compute_gradient(self, X, y, theta):
        m = X.shape[0]
        indices = np.random.choice(m, self.batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        
        predictions = X_batch @ theta
        errors = predictions - y_batch
        gradient = (1/self.batch_size) * (X_batch.T @ errors)
        return gradient.flatten()
