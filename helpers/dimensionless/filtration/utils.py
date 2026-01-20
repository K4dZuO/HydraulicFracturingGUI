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
    Использует std(diff(signal)) или MAD для оценки шума.
    
    Args:
        signal: Входной сигнал
        noise_estimate: Оценка шума (если None, вычисляется через std(diff(signal)))
    
    Returns:
        SNR в децибелах
    """
    signal = np.asarray(signal)
    valid_mask = np.isfinite(signal)
    
    if not np.any(valid_mask):
        return 0.0
    
    signal_clean = signal[valid_mask]
    
    if len(signal_clean) < 2:
        return 0.0
    
    signal_power = np.mean(signal_clean ** 2)
    
    if noise_estimate is not None:
        noise_clean = noise_estimate[valid_mask] if len(noise_estimate) == len(signal) else noise_estimate
        noise_power = np.mean(noise_clean ** 2) if isinstance(noise_clean, np.ndarray) else noise_clean ** 2
    else:
        # noise_estimate = std(diff(signal)) or MAD
        diffs = np.diff(signal_clean)
        if len(diffs) > 0:
            noise_estimate_value = np.std(diffs)
        else:
            # Fallback: используем MAD
            median = np.median(signal_clean)
            mad = np.median(np.abs(signal_clean - median))
            noise_estimate_value = 1.4826 * mad if mad > 0 else np.std(signal_clean)
        
        noise_power = noise_estimate_value ** 2
    
    if noise_power == 0:
        return 100.0 if signal_power > 0 else 0.0
    
    snr_linear = signal_power / (noise_power + 1e-12)  # Добавляем eps для стабильности
    
    if snr_linear > 1e10:
        snr_db = 100.0
    else:
        snr_db = 10 * np.log10(snr_linear)
    
    return float(snr_db)


def compute_oscillation_score(signal: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    """
    Вычисляет оценку осцилляций в сигнале.
    Улучшенная версия: учитывает не только количество изменений знака,
    но и их амплитуду относительно масштаба сигнала.
    
    Args:
        signal: Входной сигнал
        x: Координаты (если None, используется равномерная сетка)
    
    Returns:
        Оценка осцилляций (0 = нет осцилляций, >0 = есть осцилляций)
    """
    signal = np.asarray(signal)
    valid_mask = np.isfinite(signal)
    
    if not np.any(valid_mask) or np.sum(valid_mask) < 5:
        return 0.0
    
    signal_clean = signal[valid_mask]
    
    if x is not None:
        x_clean = np.asarray(x)[valid_mask]
        if len(x_clean) < 5:
            return 0.0
        # Вычисляем первую и вторую производные
        first_deriv = np.gradient(signal_clean, x_clean)
        second_deriv = np.gradient(first_deriv, x_clean)
    else:
        first_deriv = np.gradient(signal_clean)
        second_deriv = np.gradient(first_deriv)
    
    second_deriv = second_deriv[np.isfinite(second_deriv)]
    
    if len(second_deriv) < 3:
        return 0.0
    
    # Нормализуем вторую производную относительно масштаба сигнала
    signal_range = np.max(signal_clean) - np.min(signal_clean)
    if signal_range < 1e-10:
        return 0.0
    
    # Относительная вторая производная
    normalized_second_deriv = np.abs(second_deriv) / signal_range
    
    # Подсчитываем изменения знака второй производной
    sign_changes = np.sum(np.diff(np.sign(second_deriv)) != 0)
    
    # Учитываем амплитуду осцилляций: средняя амплитуда нормализованной второй производной
    mean_amplitude = np.mean(normalized_second_deriv)
    
    # Комбинированная оценка: учитываем и частоту, и амплитуду осцилляций
    # Для данных ГРП нормальные изменения кривизны не должны считаться осцилляциями
    frequency_score = sign_changes / len(second_deriv)
    amplitude_score = mean_amplitude * 10.0  # Масштабируем для сопоставимости
    
    # Осцилляции значимы только если и частота, и амплитуда высоки
    oscillation_score = frequency_score * (1.0 + amplitude_score)
    
    return float(oscillation_score)


def detect_log_scale(signal: np.ndarray, threshold: float = 3.0) -> bool:
    """
    Определяет, нужно ли применять фильтрацию в логарифмическом масштабе.
    threshold = 3.0 (3 порядка) по умолчанию.
    
    Args:
        signal: Входной сигнал
        threshold: Порог для определения в порядках (по умолчанию 3.0 = 3 порядка)
    
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
    
    # return (np.log10(signal_max / signal_min) > threshold)
    # где threshold = 3.0 по умолчанию (3 порядка)
    ratio = signal_max / signal_min
    log_ratio = np.log10(ratio)
    
    return log_ratio > threshold


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
    oscillation_threshold: float = 0.6  # Порог осцилляций для Kalman
) -> str:
    """
    Автоматически выбирает метод фильтрации на основе характеристик сигнала.
    Kalman выбирается только если oscillation_score > 0.6 and snr < 10.
    
    Args:
        signal: Входной сигнал
        x: Координаты
        snr_threshold: Порог SNR для выбора метода
        oscillation_threshold: Порог осцилляций для Kalman (0.6 по умолчанию)
    
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
    use_log = detect_log_scale(signal_clean, threshold=3.0)  # 3 порядка
    
    # Логика выбора метода
    if use_log:
        return 'log_domain'
    
    # Kalman только если oscillation_score > 0.6 and snr < 10
    if oscillation_score > oscillation_threshold and snr < 10.0:
        return 'kalman'
    
    # Для PTA рекомендуется LOWESS вместо SavGol
    # Проверяем доступность LOWESS
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        lowess_available = True
    except ImportError:
        lowess_available = False
    
    if snr > snr_threshold:
        # Предпочитаем LOWESS для PTA, fallback на SavGol
        return 'lowess' if lowess_available else 'savgol'
    elif snr > 10.0:
        return 'gaussian'
    else:
        return 'hybrid'


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
