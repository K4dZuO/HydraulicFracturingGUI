"""
Модуль ML-денойзинга для безразмерных кривых.
Опциональный модуль, включается флагом use_ml=True.
"""

import numpy as np
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Проверяем доступность ML библиотек
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class MLDenoiser:
    """
    ML-денойзер на основе автоэнкодера или простой нейронной сети.
    Опциональный компонент, работает только если ML библиотеки доступны.
    """
    
    def __init__(
        self,
        method: str = 'mlp',
        hidden_layers: Tuple[int, ...] = (32, 16),
        max_iter: int = 200
    ):
        """
        Инициализация ML-денойзера.
        
        Args:
            method: Метод ('mlp', 'autoencoder')
            hidden_layers: Архитектура скрытых слоёв
            max_iter: Максимальное количество итераций обучения
        """
        self.method = method
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.model = None
        self.scaler = None
        self.is_fitted = False
    
    def fit(self, Y: np.ndarray, curve: np.ndarray) -> 'MLDenoiser':
        """
        Обучение модели на данных.
        
        Args:
            Y: Безразмерная ось
            curve: Значения pD(Y)
        
        Returns:
            self
        """
        if not ML_AVAILABLE:
            raise RuntimeError("ML библиотеки недоступны. Установите scikit-learn.")
        
        Y = np.asarray(Y)
        curve = np.asarray(curve)
        
        # Очищаем данные
        valid_mask = np.isfinite(Y) & np.isfinite(curve)
        if not np.any(valid_mask) or np.sum(valid_mask) < 10:
            self.is_fitted = False
            return self
        
        Y_clean = Y[valid_mask]
        curve_clean = curve[valid_mask]
        
        # Нормализуем данные
        self.scaler = StandardScaler()
        Y_scaled = self.scaler.fit_transform(Y_clean.reshape(-1, 1))
        
        if self.method == 'mlp':
            # Простая MLP для реконструкции
            self.model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                max_iter=self.max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            )
            self.model.fit(Y_scaled, curve_clean)
        elif self.method == 'autoencoder':
            # Упрощённый автоэнкодер через MLP
            # Кодировщик
            encoder = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                max_iter=self.max_iter,
                random_state=42
            )
            # Обучаем на реконструкции
            encoder.fit(Y_scaled, curve_clean)
            self.model = encoder
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        self.is_fitted = True
        return self
    
    def predict(self, Y: np.ndarray) -> np.ndarray:
        """
        Предсказание отфильтрованных значений.
        
        Args:
            Y: Безразмерная ось
        
        Returns:
            Отфильтрованные значения pD
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Вызовите fit() сначала.")
        
        Y = np.asarray(Y)
        valid_mask = np.isfinite(Y)
        
        if not np.any(valid_mask):
            return np.full_like(Y, np.nan)
        
        Y_clean = Y[valid_mask]
        
        # Нормализуем
        Y_scaled = self.scaler.transform(Y_clean.reshape(-1, 1))
        
        # Предсказываем
        predicted = self.model.predict(Y_scaled)
        
        # Восстанавливаем полный массив
        result = np.full_like(Y, np.nan)
        result[valid_mask] = predicted
        
        return result
    
    def denoise(self, Y: np.ndarray, curve: np.ndarray) -> np.ndarray:
        """
        Универсальный метод денойзинга.
        
        Args:
            Y: Безразмерная ось
            curve: Значения pD(Y)
        
        Returns:
            Отфильтрованные значения
        """
        if not self.is_fitted:
            self.fit(Y, curve)
        
        return self.predict(Y)
    
    @staticmethod
    def is_available() -> bool:
        """Проверяет доступность ML функциональности."""
        return ML_AVAILABLE

