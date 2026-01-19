import numpy as np

class GradientDescent:
    """Базовый класс для реализации методов градиентного спуска"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        """
        Инициализация параметров оптимизатора
        
        Parameters
        ----------
        learning_rate : float, default=0.01
            Скорость обучения (η)
        max_iter : int, default=1000
            Максимальное число итераций
        tol : float, default=1e-4
            Точность сходимости
        """
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.loss_history = []
        self.theta = None
        
    def compute_gradient(self, X, y, theta):
        """
        Вычисление градиента функции потерь
        
        Parameters
        ----------
        X : numpy.ndarray
            Матрица признаков с bias-столбцом
        y : numpy.ndarray
            Вектор целевых значений
        theta : numpy.ndarray
            Вектор весов модели
            
        Returns
        -------
        numpy.ndarray
            Градиент функции потерь
        """
        raise NotImplementedError("Этот метод должен быть реализован в дочернем классе")
    
    def fit(self, X, y):
        """
        Обучение модели методом градиентного спуска
        
        Parameters
        ----------
        X : numpy.ndarray
            Матрица признаков (без bias-столбца)
        y : numpy.ndarray
            Вектор целевых значений
            
        Returns
        -------
        self
            Обученная модель
        """
        # Добавляем столбец единиц для bias
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Преобразуем y к одномерному массиву, если необходимо
        if y.ndim > 1:
            y = y.ravel()
        
        # Инициализация весов
        m, n = X.shape
        self.theta = np.zeros(n)
        
        # Основной цикл градиентного спуска
        for i in range(self.max_iter):
            # Вычисление градиента
            grad = self.compute_gradient(X, y, self.theta)
            
            # Обновление весов
            self.theta -= self.lr * grad
            
            # Вычисление функции потерь (MSE)
            predictions = X @ self.theta
            loss = np.mean((predictions - y) ** 2)
            self.loss_history.append(loss)
            
            # Проверка условия сходимости
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                print(f"Ранняя остановка на итерации {i}")
                break
                
        return self
    
    def predict(self, X):
        """
        Предсказание для новых данных
        
        Parameters
        ----------
        X : numpy.ndarray
            Матрица признаков (без bias-столбца)
            
        Returns
        -------
        numpy.ndarray
            Предсказания модели
        """
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.theta
    
    def get_loss_history(self):
        """Возвращает историю значений функции потерь"""
        return self.loss_history
    
    def get_weights(self):
        """Возвращает обученные веса модели"""
        return self.theta
