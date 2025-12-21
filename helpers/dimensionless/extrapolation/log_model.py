from sklearn.base import RegressorMixin
from scipy.optimize import curve_fit
import numpy as np

class _LogModel(RegressorMixin):
    """Обёртка для логарифмической модели под интерфейс scikit-learn"""
    def __init__(self):
        self.a_ = None
        self.b_ = None
        self.c_ = None
    
    def log_func(self, t, a, b, c):
        return a + b * np.log(t + c)
    
    def fit(self, X, y):
        t = X[:, -1]  # время — последний признак (предполагаем, что t в конце)
        
        # Начальные приближения
        a0 = np.mean(y)
        b0 = -1.0
        c0 = 1.0
        
        try:
            popt, _ = curve_fit(
                self.log_func, t, y,
                p0=[a0, b0, c0],
                bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, np.inf]),
                maxfev=5000
            )
            self.a_, self.b_, self.c_ = popt
        except RuntimeError:
            # Fallback: просто запомним среднее
            self.a_ = np.mean(y)
            self.b_ = 0.0
            self.c_ = 1.0
        return self
    
    def predict(self, X):
        t = X[:, -1]
        return self.a_ + self.b_ * np.log(t + self.c_)
