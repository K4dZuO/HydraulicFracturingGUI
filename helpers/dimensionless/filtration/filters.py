"""
Модуль фильтрации сигналов для безразмерных кривых.
Реализует различные методы фильтрации согласно статье.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import warnings

warnings.filterwarnings('ignore')

# Проверка доступности LOWESS
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    LOWESS_AVAILABLE = True
except ImportError:
    LOWESS_AVAILABLE = False


_last_diagnostics = {}


class SignalFilters:
    """
    Класс для применения различных методов фильтрации сигналов.
    Реализует методы из статьи: Savitzky-Golay, Gaussian, Kalman, Log-domain, Hybrid.
    """
    
    @staticmethod
    def lowess(
        curve: np.ndarray,
        x: Optional[np.ndarray] = None,
        frac: float = 0.05,
        it: int = 3
    ) -> np.ndarray:
        """
        LOWESS (Locally Weighted Scatterplot Smoothing) фильтр.
        Рекомендуется для PTA вместо Savitzky-Golay.
        Эталонная реализация из statsmodels.
        
        Args:
            curve: Входной сигнал
            x: Координаты (если None, используется равномерная сетка)
            frac: Доля точек для локальной регрессии (0.05 = 5%)
            it: Количество итераций для робастной регрессии
        
        Returns:
            Отфильтрованный сигнал
        """
        if not LOWESS_AVAILABLE:
            # Fallback на SavGol если LOWESS недоступен
            warnings.warn("LOWESS недоступен, используется SavGol fallback")
            return SignalFilters.savgol(curve, window_length=None, polyorder=2)
        
        curve = np.asarray(curve)
        valid_mask = np.isfinite(curve)
        
        if not np.any(valid_mask):
            return curve
        
        curve_clean = curve[valid_mask]
        n = len(curve_clean)
        
        if n < 3:
            return curve
        
        # Подготовка координат
        if x is not None:
            x_clean = np.asarray(x)[valid_mask]
        else:
            x_clean = np.arange(n, dtype=float)
        
        try:
            # Применяем LOWESS
            filtered = lowess(
                curve_clean, 
                x_clean, 
                frac=frac, 
                it=it, 
                return_sorted=False
            )
            
            # Восстанавливаем полный массив
            if np.all(valid_mask):
                return filtered
            else:
                result = curve.copy()
                result[valid_mask] = filtered
                return result
        except Exception as e:
            # Fallback на SavGol при ошибке
            warnings.warn(f"Ошибка LOWESS: {e}, используется SavGol fallback")
            return SignalFilters.savgol(curve, window_length=None, polyorder=2, x=x)
    
    @staticmethod
    def savgol(
        curve: np.ndarray,
        window_length: Optional[int] = None,
        polyorder: int = 2,
        mode: str = 'nearest'
    ) -> np.ndarray:
        """
        Фильтр Савицкого-Голея для сглаживания сигнала.
        
        Args:
            curve: Входной сигнал
            window_length: Длина окна (должна быть нечётной, >= polyorder+1)
            polyorder: Порядок полинома
            mode: Режим обработки краёв
        
        Returns:
            Отфильтрованный сигнал
        """
        curve = np.asarray(curve)
        valid_mask = np.isfinite(curve)
        
        if not np.any(valid_mask) or np.sum(valid_mask) < 3:
            return curve
        
        # Если все значения валидны, работаем напрямую
        if np.all(valid_mask):
            n = len(curve)
            if window_length is None:
                # Еще более консервативная стратегия для предотвращения артефактов
                # Используем 3% вместо 5% для более мягкого сглаживания
                w = max(5, int(round(n * 0.03)))
                if w % 2 == 0:
                    w += 1
                window_length = min(w, 9)  # Максимум 9 вместо 11
            # Валидация: window_length >= polyorder + 1 и нечётное
            window_length = max(window_length, polyorder + 1)
            if window_length % 2 == 0:
                window_length += 1
            window_length = min(window_length, n if n % 2 == 1 else n - 1)
            if window_length > n or n < polyorder + 3:
                return curve
            try:
                # Используем 'mirror' для более стабильной работы на краях
                filter_mode = 'mirror' if mode == 'nearest' else mode
                return savgol_filter(curve, window_length, polyorder, mode=filter_mode)
            except (ValueError, np.linalg.LinAlgError):
                return curve
        
        curve_clean = curve[valid_mask]
        n = len(curve_clean)
        
        # Автоматический выбор window_length - более консервативный
        if window_length is None:
            # Используем 3% вместо 5% для более мягкого сглаживания
            w = max(5, int(round(n * 0.03)))
            if w % 2 == 0:
                w += 1
            window_length = min(w, 9)  # Максимум 9 вместо 11
        
        # Валидация: window_length >= polyorder + 1 и нечётное
        window_length = max(window_length, polyorder + 1)
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, n if n % 2 == 1 else n - 1)
        
        if window_length > n or n < polyorder + 3:
            return curve
        
        try:
            # Используем 'mirror' для более стабильной работы на краях
            filter_mode = 'mirror' if mode == 'nearest' else mode
            filtered = savgol_filter(curve_clean, window_length, polyorder, mode=filter_mode)
            result = curve.copy()
            result[valid_mask] = filtered
            return result
        except (ValueError, np.linalg.LinAlgError):
            # Fallback: возвращаем исходный сигнал
            return curve
    
    @staticmethod
    def gaussian(
        curve: np.ndarray,
        sigma: Optional[float] = None,
        mode: str = 'nearest'
    ) -> np.ndarray:
        """
        Гауссовское сглаживание сигнала.
        Улучшенная версия с адаптивным выбором параметров.
        
        Args:
            curve: Входной сигнал
            sigma: Стандартное отклонение гауссова ядра (если None, выбирается автоматически)
            mode: Режим обработки краёв
        
        Returns:
            Отфильтрованный сигнал
        """
        curve = np.asarray(curve)
        valid_mask = np.isfinite(curve)
        
        if not np.any(valid_mask):
            return curve
        
        # Автоматический выбор sigma
        if sigma is None:
            n = len(curve) if np.all(valid_mask) else np.sum(valid_mask)
            # sigma = min(max(0.25, n / 200.0), 1.0)
            if n < 8:
                sigma = 0.3
            else:
                sigma = min(max(0.25, n / 200.0), 1.0)
        else:
            # Валидация sigma: [0.1, 3.0]
            if sigma < 0.1:
                warnings.warn(f"sigma={sigma} слишком мал, установлен 0.1")
                sigma = 0.1
            elif sigma > 3.0:
                warnings.warn(f"sigma={sigma} слишком велик, установлен 3.0")
                sigma = 3.0
        
        # Если все значения валидны, работаем напрямую
        if np.all(valid_mask):
            if len(curve) < 3:
                return curve
            try:
                # Используем mode='reflect' или 'mirror' для стабильной работы на краях
                filter_mode = 'reflect' if mode == 'nearest' else mode
                return gaussian_filter1d(curve, sigma=sigma, mode=filter_mode)
            except Exception:
                return curve
        
        curve_clean = curve[valid_mask]
        
        if len(curve_clean) < 3:
            return curve
        
        try:
            # Используем mode='reflect' или 'mirror' для стабильной работы на краях
            filter_mode = 'reflect' if mode == 'nearest' else mode
            filtered = gaussian_filter1d(curve_clean, sigma=sigma, mode=filter_mode)
            result = curve.copy()
            result[valid_mask] = filtered
            return result
        except Exception:
            return curve
    
    @staticmethod
    def kalman(
        curve: np.ndarray,
        process_noise: Optional[float] = None,
        measurement_noise: Optional[float] = None
    ) -> np.ndarray:
        """
        Простой одномерный фильтр Калмана для нестационарного шума.
        Простая 1D реализация БЕЗ тренда/скорости.
        
        Модель состояния: x_k = x_{k-1} + w_k (w_k ~ N(0, Q))
        Наблюдение: z_k = x_k + v_k (v_k ~ N(0, R))
        
        Args:
            curve: Входной сигнал
            process_noise: Дисперсия процессного шума (Q). Если None, вычисляется автоматически
            measurement_noise: Дисперсия шума измерений (R). Если None, вычисляется автоматически
        
        Returns:
            Отфильтрованный сигнал
        """
        curve = np.asarray(curve)
        valid_mask = np.isfinite(curve)
        
        if not np.any(valid_mask):
            return curve
        
        curve_clean = curve[valid_mask]
        n = len(curve_clean)
        
        if n < 2:
            return curve
        
        # Автоматическая оценка параметров шума
        if process_noise is None or measurement_noise is None:
            # R ~ var(diffs), Q ~ R*0.01
            diffs = np.diff(curve_clean)
            if len(diffs) > 0:
                noise_estimate = np.std(diffs)
            else:
                noise_estimate = np.std(curve_clean) if len(curve_clean) > 1 else 1.0
            
            if noise_estimate <= 0:
                noise_estimate = np.std(curve_clean) if len(curve_clean) > 1 else 1.0
            
            if measurement_noise is None:
                # R ~ var(diffs)
                measurement_noise = max(0.01, noise_estimate ** 2)
            
            if process_noise is None:
                # Q ~ R*0.01
                process_noise = max(0.01, measurement_noise * 0.01)
        
        # Инициализация
        x = curve_clean[0]  # x0 = curve_clean[0]
        P = np.var(curve_clean) if len(curve_clean) > 1 else 1.0  # P0 = var(signal) or 1.0
        
        filtered_values = [x]
        
        # Простая 1D итерация Калмана (scalar)
        for i in range(1, n):
            z = curve_clean[i]  # Измерение
            
            # Предсказание (scalar)
            x_pred = x  # x_k = x_{k-1} (модель без тренда)
            P_pred = P + process_noise  # P_pred = P + Q
            
            # Обновление (scalar)
            K = P_pred / (P_pred + measurement_noise)  # K = P_pred / (P_pred + R)
            x = x_pred + K * (z - x_pred)  # x = x_pred + K*(z - x_pred)
            P = (1 - K) * P_pred  # P = (1 - K) * P_pred
            
            filtered_values.append(x)
        
        result = curve.copy()
        result[valid_mask] = np.array(filtered_values)
        return result
    
    @staticmethod
    def log_domain(
        curve: np.ndarray,
        base_filter: str = 'savgol',
        **filter_kwargs
    ) -> np.ndarray:
        """
        Фильтрация в логарифмическом масштабе.
        Применять только если detect_log_scale → True (3 порядка).
        В лог-масштабе НЕ применять полиномиальную фильтрацию с высоким order (polyorder ≤ 2).
        
        Args:
            curve: Входной сигнал (должен быть положительным)
            base_filter: Базовый метод фильтрации ('savgol', 'gaussian', 'kalman')
            **filter_kwargs: Параметры базового фильтра
        
        Returns:
            Отфильтрованный сигнал
        """
        curve = np.asarray(curve)
        valid_mask = np.isfinite(curve) & (curve > 0)
        
        if not np.any(valid_mask):
            return curve
        
        curve_clean = curve[valid_mask]
        
        # Проверка положительности и обработка ≤0
        if np.any(curve_clean <= 0):
            # Если есть ≤0, делаем shift: signal_pos = signal_clean - min(signal_clean) + eps
            eps = 1e-10
            min_val = np.min(curve_clean)
            if min_val <= 0:
                curve_clean = curve_clean - min_val + eps
            else:
                curve_clean = curve_clean + eps
        
        # Переходим в логарифмический масштаб
        log_curve = np.log10(curve_clean)
        
        # В лог-масштабе НЕ применять polyorder > 2
        if base_filter == 'savgol':
            # Ограничиваем polyorder до 2
            if 'polyorder' in filter_kwargs and filter_kwargs['polyorder'] > 2:
                filter_kwargs['polyorder'] = 2
            elif 'polyorder' not in filter_kwargs:
                filter_kwargs['polyorder'] = 2
            filtered_log = SignalFilters.savgol(log_curve, **filter_kwargs)
        elif base_filter == 'gaussian':
            filtered_log = SignalFilters.gaussian(log_curve, **filter_kwargs)
        elif base_filter == 'kalman':
            filtered_log = SignalFilters.kalman(log_curve, **filter_kwargs)
        else:
            filtered_log = log_curve
        
        # Возвращаемся из логарифмического масштаба
        filtered = 10 ** filtered_log
        
        # Коррекция bias: если был shift, возвращаем назад
        if np.any(curve[valid_mask] <= 0):
            min_val_original = np.min(curve[valid_mask])
            if min_val_original <= 0:
                filtered = filtered - eps + min_val_original
        
        result = curve.copy()
        result[valid_mask] = filtered
        return result
    
    @staticmethod
    def hybrid(
        curve: np.ndarray,
        x: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Гибридный фильтр: fill_missing → savgol → gaussian.
        НЕ использует log_domain по умолчанию.
        
        Args:
            curve: Входной сигнал
            x: Координаты (опционально)
        
        Returns:
            Отфильтрованный сигнал
        """
        curve = np.asarray(curve)
        valid_mask = np.isfinite(curve)
        
        if not np.any(valid_mask):
            return curve
        
        # 1. fill_missing_values
        from .utils import fill_missing_values
        curve_filled = fill_missing_values(curve, x=x, method='linear')
        
        # 2. savgol(signal, window_length=None, polyorder=2)
        filtered = SignalFilters.savgol(curve_filled, window_length=None, polyorder=2, mode='nearest')
        
        # 3. gaussian(signal, sigma=0.5)
        filtered = SignalFilters.gaussian(filtered, sigma=0.5, mode='reflect')
        
        # Возвращаем полноразмерный массив (вставляем назад NaN, если были)
        # Если были NaN, они уже заполнены fill_missing_values, но нужно восстановить исходные NaN
        if not np.all(valid_mask):
            result = curve.copy()
            result[valid_mask] = filtered[valid_mask]
            # Восстанавливаем исходные NaN (если они были)
            result[~valid_mask] = np.nan
            return result
        
        return filtered
    
    @staticmethod
    def denoise(
        curve: np.ndarray,
        method: Optional[str] = None,
        x: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Универсальный метод денойзинга с автоматическим выбором фильтра.
        
        Args:
            curve: Входной сигнал
            method: Метод фильтрации (если None, выбирается автоматически)
            x: Координаты
            **kwargs: Дополнительные параметры фильтра
        
        Returns:
            Отфильтрованный сигнал
        """
        if method is None:
            from .utils import select_filter_method
            method = select_filter_method(curve, x)
        
        if method == 'lowess':
            frac = kwargs.get('frac', 0.05)
            it = kwargs.get('it', 3)
            return SignalFilters.lowess(curve, x=x, frac=frac, it=it)
        elif method == 'savgol':
            return SignalFilters.savgol(curve, **kwargs)
        elif method == 'gaussian':
            return SignalFilters.gaussian(curve, **kwargs)
        elif method == 'kalman':
            return SignalFilters.kalman(curve, **kwargs)
        elif method == 'log_domain':
            return SignalFilters.log_domain(curve, **kwargs)
        elif method == 'hybrid':
            return SignalFilters.hybrid(curve, x)
        else:
            raise ValueError(f"Неизвестный метод фильтрации: {method}")

