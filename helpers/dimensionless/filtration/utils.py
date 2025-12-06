"""
Вспомогательные функции для модуля фильтрации безразмерных кривых.
Включает функции для оценки качества сигнала, обнаружения осцилляций и т.д.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


def compute_snr(signal: np.ndarray, noise_estimate: Optional[np.ndarray] = None) -> float:
    """
    Вычисляет отношение сигнал/шум (SNR) в децибелах.
    
    Args:
        signal: Входной сигнал
        noise_estimate: Оценка шума (если None, используется стандартное отклонение)
    
    Returns:
        SNR в децибелах
    """
    signal = np.asarray(signal)
    valid_mask = np.isfinite(signal)
    
    if not np.any(valid_mask):
        return 0.0
    
    signal_clean = signal[valid_mask]
    signal_power = np.mean(signal_clean ** 2)
    
    if noise_estimate is not None:
        noise_clean = noise_estimate[valid_mask]
        noise_power = np.mean(noise_clean ** 2)
    else:
        # Используем стандартное отклонение как оценку шума
        noise_power = np.var(signal_clean)
    
    if noise_power == 0:
        return np.inf if signal_power > 0 else 0.0
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    
    return float(snr_db)


def compute_oscillation_score(signal: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    """
    Вычисляет оценку осцилляций в сигнале.
    Основана на подсчёте изменений знака второй производной.
    
    Args:
        signal: Входной сигнал
        x: Координаты (если None, используется равномерная сетка)
    
    Returns:
        Оценка осцилляций (0 = нет осцилляций, >0 = есть осцилляций)
    """
    signal = np.asarray(signal)
    valid_mask = np.isfinite(signal)
    
    if not np.any(valid_mask) or np.sum(valid_mask) < 3:
        return 0.0
    
    signal_clean = signal[valid_mask]
    
    if x is not None:
        x_clean = np.asarray(x)[valid_mask]
        if len(x_clean) < 3:
            return 0.0
        second_deriv = np.gradient(np.gradient(signal_clean, x_clean), x_clean)
    else:
        second_deriv = np.gradient(np.gradient(signal_clean))
    
    second_deriv = second_deriv[np.isfinite(second_deriv)]
    
    if len(second_deriv) < 2:
        return 0.0
    
    # Подсчитываем изменения знака второй производной
    sign_changes = np.sum(np.diff(np.sign(second_deriv)) != 0)
    
    # Нормализуем по длине сигнала
    oscillation_score = sign_changes / len(second_deriv)
    
    return float(oscillation_score)


def detect_log_scale(signal: np.ndarray, threshold: float = 10.0) -> bool:
    """
    Определяет, нужно ли применять фильтрацию в логарифмическом масштабе.
    
    Args:
        signal: Входной сигнал
        threshold: Порог для определения (если диапазон > threshold порядков, то нужен log)
    
    Returns:
        True, если нужна фильтрация в log-масштабе
    """
    signal = np.asarray(signal)
    valid_mask = np.isfinite(signal) & (signal > 0)
    
    if not np.any(valid_mask):
        return False
    
    signal_clean = signal[valid_mask]
    signal_min = np.min(signal_clean)
    signal_max = np.max(signal_clean)
    
    if signal_min <= 0:
        return False
    
    ratio = signal_max / signal_min
    
    return ratio > (10.0 ** threshold)


def estimate_noise_level(signal: np.ndarray, method: str = 'median') -> float:
    """
    Оценивает уровень шума в сигнале.
    
    Args:
        signal: Входной сигнал
        method: Метод оценки ('median', 'std', 'mad')
    
    Returns:
        Оценка уровня шума
    """
    signal = np.asarray(signal)
    valid_mask = np.isfinite(signal)
    
    if not np.any(valid_mask):
        return 0.0
    
    signal_clean = signal[valid_mask]
    
    if method == 'median':
        # Используем медианное абсолютное отклонение
        median = np.median(signal_clean)
        mad = np.median(np.abs(signal_clean - median))
        return float(mad)
    elif method == 'std':
        return float(np.std(signal_clean))
    elif method == 'mad':
        # Median Absolute Deviation
        median = np.median(signal_clean)
        mad = np.median(np.abs(signal_clean - median))
        return float(1.4826 * mad)  # Масштабирование для нормального распределения
    else:
        raise ValueError(f"Неизвестный метод: {method}")


def select_filter_method(
    signal: np.ndarray,
    x: Optional[np.ndarray] = None,
    snr_threshold: float = 20.0,
    oscillation_threshold: float = 0.3
) -> str:
    """
    Автоматически выбирает метод фильтрации на основе характеристик сигнала.
    
    Args:
        signal: Входной сигнал
        x: Координаты
        snr_threshold: Порог SNR для выбора метода
        oscillation_threshold: Порог осцилляций
    
    Returns:
        Название метода фильтрации ('savgol', 'gaussian', 'kalman', 'log_domain', 'hybrid')
    """
    signal = np.asarray(signal)
    valid_mask = np.isfinite(signal)
    
    if not np.any(valid_mask):
        return 'hybrid'
    
    signal_clean = signal[valid_mask]
    
    # Вычисляем характеристики
    snr = compute_snr(signal_clean)
    oscillation_score = compute_oscillation_score(signal_clean, x[valid_mask] if x is not None else None)
    use_log = detect_log_scale(signal_clean)
    
    # Логика выбора
    if use_log:
        return 'log_domain'
    
    if oscillation_score > oscillation_threshold:
        return 'kalman'  # Калман лучше для нестационарного шума
    
    if snr > snr_threshold:
        return 'savgol'  # SavGol для слабого шума
    elif snr > 10.0:
        return 'gaussian'  # Gaussian для умеренного шума
    else:
        return 'hybrid'  # Hybrid по умолчанию


def fill_missing_values(
    signal: np.ndarray,
    x: Optional[np.ndarray] = None,
    method: str = 'linear'
) -> np.ndarray:
    """
    Заполняет пропущенные значения (NaN) в сигнале.
    
    Args:
        signal: Входной сигнал
        x: Координаты
        method: Метод интерполяции ('linear', 'forward', 'backward', 'nearest')
    
    Returns:
        Сигнал с заполненными пропусками
    """
    signal = np.asarray(signal).copy()
    valid_mask = np.isfinite(signal)
    
    if np.all(valid_mask):
        return signal
    
    if not np.any(valid_mask):
        # Все значения пропущены - заполняем константой
        return np.zeros_like(signal)
    
    if x is not None:
        x_arr = np.asarray(x)
        x_valid = x_arr[valid_mask]
        signal_valid = signal[valid_mask]
        x_missing = x_arr[~valid_mask]
        
        if len(x_valid) < 2:
            # Недостаточно точек для интерполяции
            signal[~valid_mask] = np.mean(signal_valid) if len(signal_valid) > 0 else 0.0
            return signal
        
        if method == 'linear':
            signal[~valid_mask] = np.interp(x_missing, x_valid, signal_valid)
        elif method == 'nearest':
            from scipy.interpolate import interp1d
            interp = interp1d(x_valid, signal_valid, kind='nearest', 
                            bounds_error=False, fill_value='extrapolate')
            signal[~valid_mask] = interp(x_missing)
        else:
            signal[~valid_mask] = np.mean(signal_valid)
    else:
        # Простая интерполяция по индексам
        indices = np.arange(len(signal))
        indices_valid = indices[valid_mask]
        signal_valid = signal[valid_mask]
        indices_missing = indices[~valid_mask]
        
        if method == 'linear':
            signal[~valid_mask] = np.interp(indices_missing, indices_valid, signal_valid)
        elif method == 'forward':
            signal = pd.Series(signal).fillna(method='ffill').values
        elif method == 'backward':
            signal = pd.Series(signal).fillna(method='bfill').values
        else:
            signal[~valid_mask] = np.mean(signal_valid)
    
    return signal


def remove_outliers(
    signal: np.ndarray,
    method: str = 'iqr',
    threshold: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Обнаруживает и удаляет выбросы из сигнала.
    
    Args:
        signal: Входной сигнал
        method: Метод обнаружения ('iqr', 'zscore', 'modified_zscore')
        threshold: Порог для обнаружения
    
    Returns:
        Tuple (очищенный сигнал, маска выбросов)
    """
    signal = np.asarray(signal)
    valid_mask = np.isfinite(signal)
    
    if not np.any(valid_mask):
        return signal, np.zeros_like(signal, dtype=bool)
    
    signal_clean = signal[valid_mask]
    outlier_mask = np.zeros_like(signal, dtype=bool)
    
    if method == 'iqr':
        Q1 = np.percentile(signal_clean, 25)
        Q3 = np.percentile(signal_clean, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask[valid_mask] = (signal_clean < lower_bound) | (signal_clean > upper_bound)
    elif method == 'zscore':
        mean = np.mean(signal_clean)
        std = np.std(signal_clean)
        if std > 0:
            z_scores = np.abs((signal_clean - mean) / std)
            outlier_mask[valid_mask] = z_scores > threshold
    elif method == 'modified_zscore':
        median = np.median(signal_clean)
        mad = np.median(np.abs(signal_clean - median))
        if mad > 0:
            modified_z_scores = 0.6745 * np.abs((signal_clean - median) / mad)
            outlier_mask[valid_mask] = modified_z_scores > threshold
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    # Заменяем выбросы на NaN
    signal_cleaned = signal.copy()
    signal_cleaned[outlier_mask] = np.nan
    
    return signal_cleaned, outlier_mask

