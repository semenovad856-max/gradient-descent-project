import numpy as np
from gradient_descent import GradientDescent

class SGD(GradientDescent):
    """Стохастический градиентный спуск"""
    
    def compute_gradient(self, X, y, theta):
        m = X.shape[0]
        i = np.random.randint(0, m)
        x_i = X[i].reshape(1, -1)
        y_i = y[i]
        
        prediction = x_i @ theta
        error = prediction - y_i
        gradient = x_i.T * error
        return gradient.flatten()
