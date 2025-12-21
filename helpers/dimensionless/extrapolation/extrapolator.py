"""
Модуль экстраполяции размерных параметров (P, dP, Q).
Использует физические ограничения и модели трендов (экспонента, логарифм, ARIMA).
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.base import BaseEstimator

from .physics_constraints import apply_physical_constraints


def extrapolate_parameters(
    model_P: BaseEstimator,
    model_Q: BaseEstimator,
    model_dP: Optional[BaseEstimator] = None,
    P_last: float = None,
    dP_last: float = None,
    Q_last: float = None,
    t_future: np.ndarray = None,
    static_features: np.ndarray = None,
    # Новые параметры для поддержки разных моделей
    t_train: Optional[np.ndarray] = None,
    use_dP_from_data: bool = True,
    apply_constraints: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Экстраполирует P, Q, dP на n шагов вперёд по времени с применением физических ограничений.
    
    Args:
        model_P: Обученная модель для давления P
        model_Q: Обученная модель для дебита Q
        model_dP: Обученная модель для депрессии dP (опционально)
        P_last: Последнее значение давления из исторических данных
        dP_last: Последнее значение депрессии из исторических данных
        Q_last: Последнее значение дебита из исторических данных
        t_future: Временные точки для экстраполяции
        static_features: Статические параметры [Skin, h, ...] (для совместимости, может быть None)
        t_train: Временные точки для обучения (для моделей, которым нужен t)
        use_dP_from_data: Использовать модель dP из данных, если доступна
        apply_constraints: Применять физические ограничения
    
    Returns:
        P_ext, Q_ext, dP_ext: Экстраполированные значения с применением ограничений
    """
    
    # Поддержка старого интерфейса (для обратной совместимости)
    if t_future is None:
        raise ValueError("t_future обязателен")
    
    # Определяем, как предсказывать в зависимости от типа модели
    # Проверяем, есть ли метод predict с параметром t или только для временных рядов
    
    # Предсказание P
    if t_train is not None and hasattr(model_P, 'predict') and len(t_train) > 0:
        # Модели трендов (экспонента, логарифм) принимают t
        try:
            P_ext = model_P.predict(t_future)
        except Exception:
            # Fallback: если модель не принимает t напрямую
            if static_features is not None:
                X_future = np.hstack([
                    np.tile(static_features, (len(t_future), 1)),
                    t_future.reshape(-1, 1)
                ])
                P_ext = model_P.predict(X_future)
            else:
                # Для ARIMA - предсказываем n_periods шагов
                n_periods = len(t_future)
                P_ext = model_P.predict(n_periods)
    else:
        # Старый интерфейс со статическими признаками
        if static_features is not None:
            X_future = np.hstack([
                np.tile(static_features, (len(t_future), 1)),
                t_future.reshape(-1, 1)
            ])
            P_ext = model_P.predict(X_future)
        else:
            raise ValueError("Нужны либо t_train, либо static_features")
    
    # Предсказание Q
    if t_train is not None and hasattr(model_Q, 'predict') and len(t_train) > 0:
        try:
            Q_ext = model_Q.predict(t_future)
        except Exception:
            if static_features is not None:
                X_future = np.hstack([
                    np.tile(static_features, (len(t_future), 1)),
                    t_future.reshape(-1, 1)
                ])
                Q_ext = model_Q.predict(X_future)
            else:
                n_periods = len(t_future)
                Q_ext = model_Q.predict(n_periods)
    else:
        if static_features is not None:
            X_future = np.hstack([
                np.tile(static_features, (len(t_future), 1)),
                t_future.reshape(-1, 1)
            ])
            Q_ext = model_Q.predict(X_future)
        else:
            raise ValueError("Нужны либо t_train, либо static_features")
    
    # Предсказание dP
    if model_dP is not None and use_dP_from_data:
        # Используем отдельную модель для dP
        if t_train is not None and hasattr(model_dP, 'predict') and len(t_train) > 0:
            try:
                dP_ext = model_dP.predict(t_future)
            except Exception:
                if static_features is not None:
                    X_future = np.hstack([
                        np.tile(static_features, (len(t_future), 1)),
                        t_future.reshape(-1, 1)
                    ])
                    dP_ext = model_dP.predict(X_future)
                else:
                    n_periods = len(t_future)
                    dP_ext = model_dP.predict(n_periods)
        else:
            if static_features is not None:
                X_future = np.hstack([
                    np.tile(static_features, (len(t_future), 1)),
                    t_future.reshape(-1, 1)
                ])
                dP_ext = model_dP.predict(X_future)
            else:
                # Fallback: вычисляем из P
                if P_last is not None:
                    dP_ext = P_last - P_ext
                else:
                    raise ValueError("Нужен либо model_dP, либо P_last для вычисления dP")
    else:
        # Вычисляем dP из P (если P_last предоставлен)
        if P_last is not None:
            dP_ext = P_last - P_ext
        elif dP_last is not None:
            # Используем последнее значение dP как базу и добавляем изменение
            # Это упрощенный подход - лучше использовать модель
            dP_ext = np.full_like(t_future, dP_last)
        else:
            raise ValueError("Нужен model_dP или P_last/dP_last для вычисления dP")
    
    # Применяем физические ограничения
    if apply_constraints and P_last is not None and dP_last is not None and Q_last is not None:
        P_ext, dP_ext, Q_ext = apply_physical_constraints(
            P_ext, dP_ext, Q_ext,
            P_last=P_last,
            dP_last=dP_last,
            Q_last=Q_last,
            enforce_dP_monotonicity=True,
            enforce_P_monotonicity=True
        )
    
    # Базовая защита от отрицательных значений
    P_ext = np.maximum(P_ext, 0.0)
    dP_ext = np.maximum(dP_ext, 0.0)
    Q_ext = np.maximum(Q_ext, 0.0)
    
    return P_ext, dP_ext, Q_ext
