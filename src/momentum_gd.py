import numpy as np
from gradient_descent import GradientDescent

class MomentumGD(GradientDescent):
    """Градиентный спуск с моментом"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, beta=0.9):
        super().__init__(learning_rate, max_iter, tol)
        self.beta = beta
        self.velocity = None
    
    def compute_gradient(self, X, y, theta):
        m = X.shape[0]
        i = np.random.randint(0, m)
        x_i = X[i].reshape(1, -1)
        y_i = y[i]
        
        prediction = x_i @ theta
        error = prediction - y_i
        gradient = x_i.T * error
        
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        
        self.velocity = self.beta * self.velocity + (1 - self.beta) * gradient
        return self.velocity.flatten()
