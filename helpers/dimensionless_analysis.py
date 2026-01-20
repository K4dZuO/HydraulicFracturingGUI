"""
Модуль для работы с безразмерными параметрами МГРП
Реализует конвертацию в безразмерные координаты и интерполяцию в пространстве безразмерных кривых
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List, Union, Any
from dataclasses import dataclass
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DimensionlessParameters:
    """Безразмерные параметры для МГРП"""
    # Фильтрационный параметр
    X: np.ndarray  # (0.00864 * k * h * Δp_i) / (μ * B * Q)
    # Ёмкостной параметр  
    Y: np.ndarray  # (Q * B * t) / (24 * φ * c_t * h * L² * Δp_i)
    
    # Исходные физические данные
    pressure: np.ndarray  # Исходное давление, атм
    flow_rate: np.ndarray  # Исходный дебит, м³/сут
    dP: np.ndarray  # Приращение давления, вычисленное из данных, атм
    
    # Исходные физические параметры
    k: float  # Проницаемость, мД
    h: float  # Толщина пласта, м
    mu: float  # Вязкость, мПа·с
    B: float  # Объемный коэффициент
    phi: float  # Пористость
    c_t: float  # Общая сжимаемость, 1/атм
    L: float  # Длина трещины, м
    delta_p_i: float  # Нормировочный перепад давления для pD (может быть скаляр)
    Q: float  # Дебит, м³/сут
    t: np.ndarray  # Время, ч


class DimensionlessConverter:
    """Конвертер в безразмерные параметры для МГРП"""
    
    def __init__(self):
        self.default_params = {
            'k': 1.0,  # мД
            'mu': 1.0,  # мПа·с
            'B': 1.0,  # безразмерный
            'phi': 0.1,  # безразмерный
            'c_t': 1e-4,  # 1/атм
        }
    
    def convert_to_dimensionless(self, 
                                time: pd.Series,
                                pressure: pd.Series,
                                flow_rate: pd.Series,
                                depression: pd.Series,
                                well_params: Dict[str, float],
                                x_mode: str = 'alt',
                                ) -> DimensionlessParameters:
        """
        Конвертация в безразмерные параметры    
        
        Args:
            time: Временной ряд, ч
            pressure: Давление, атм
            flow_rate: Дебит, м³/сут
            well_params: Параметры скважины (k, h, mu, B, phi, c_t, L, skin, N, a_L, dP)
            x_mode: Режим вычисления X:
                - 'darcy': X = (dp/dt) * (k * h) / (Q * mu * B)
                - 'constant': X = (0.00864 * k * h * Δp) / (μ * B * Q) (использует вектор Δp)
                - 'alt': X = (0.00864 * k * h * Δp_i) / (μ * B * Q) (использует вектор Δp_i и вектор Q)
        """
        # Извлекаем параметры
        k = well_params.get('k', self.default_params['k'])
        h = well_params.get('h', 10.0)
        mu = well_params.get('mu', self.default_params['mu'])
        B = well_params.get('B', self.default_params['B'])
        phi = well_params.get('phi', self.default_params['phi'])
        c_t = well_params.get('c_t', self.default_params['c_t'])
        L = well_params.get('L', 100.0)
        
        # Нормировочный перепад давления (скаляр) для pD: используем размах как надёжную норму
        try:
            delta_p_i = float(pressure.max()) - float(pressure.min())
        except Exception:
            delta_p_i = 1.0
        if not np.isfinite(delta_p_i) or delta_p_i == 0:
            delta_p_i = 1.0
        
        # Конвертируем в numpy массивы
        t = time.values
        p = pressure.values
        q = flow_rate.values
        
        # Обработка depression: если None, вычисляем из давления
        if depression is None:
            # Вычисляем приращение давления из самого давления
            dP = np.diff(p, prepend=p[0]) if len(p) > 0 else np.array([])
            # Логируем предупреждение
            try:
                from helpers.math_error_logger import log_math_error
                log_math_error(
                    subsystem="dimensionless_conversion",
                    method="convert_to_dimensionless",
                    error_type="missing_depression",
                    error_value=0.0,
                    error_message="depression is None, computed from pressure",
                    data_volume=len(p),
                    data_quality=1.0 - (np.sum(np.isnan(p)) / len(p)) if len(p) > 0 else 0.0
                )
            except Exception:
                pass  # Не прерываем выполнение при ошибке логирования
        else:
            dP = depression.values
        
        # Средний дебит (используется только для режима 'constant')
        Q = flow_rate.mean() if not flow_rate.empty else 1.0

        delta_p_vec = dP
            
        
        # Безопасная замена нулей на маленькое число во избежание деления на ноль
        # Используем безопасную версию для формул, чтобы избежать деления на ноль
        delta_p_vec_safe = np.where(np.abs(delta_p_vec) < 1e-12, 1e-12, delta_p_vec)
        
        # Фильтрационный параметр X
        if x_mode == 'darcy':
            # X = (dp/dt) * (k * h) / (Q * mu * B)
            dt = np.gradient(t)
            dt = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
            dp = np.gradient(p)
            X = (dp / dt) * (k * h) / ((Q if Q != 0 else 1e-12) * mu * B)
        elif x_mode == 'alt':
            # Стандартная формула: X = (0.00864 * k * h * Δp_i) / (μ * B * Q)
            # где Δp_i - вектор приращений давления из данных (dP), Q - вектор дебита из данных
            # Никакое масштабирование не применяется
            q_safe = np.where(np.abs(q) < 1e-12, 1e-12, q)
            X = (0.00864 * k * h * delta_p_vec) / (mu * B * q_safe)
        else:
            # Константный X по определению
            # Используем вектор Δp (dP из данных CSV, если передан, иначе вычисленный)
            # С защитой от деления на ноль
            Q_safe = Q if Q != 0 else 1.0
            X = (0.00864 * k * h * delta_p_vec) / (mu * B * Q_safe)
        
        # Ёмкостной параметр Y
        if x_mode == 'alt':
            # Стандартная формула: Y = (Q * B * t) / (24 * φ * c_t * h * L² * Δp_i)
            # где Δp_i - вектор приращений давления из данных (dP), Q - вектор дебита из данных
            # Никакое масштабирование не применяется
            Y = (q * B * t) / (24 * phi * c_t * h * L**2 * delta_p_vec_safe)
        else:
            # Стандартная формула: Y = (Q * B * t) / (24 * φ * c_t * h * L² * Δp)
            # Используем delta_p_vec_safe для защиты от деления на ноль
            Y = (Q * B * t) / (24 * phi * c_t * h * L**2 * delta_p_vec_safe)
        
        return DimensionlessParameters(
            X=X,
            Y=Y,
            pressure=p,  # Сохраняем исходные данные
            flow_rate=q,
            dP=dP,  # Приращение давления из CSV данных или вычисленное из давления
            k=k, h=h, mu=mu, B=B, phi=phi, c_t=c_t, L=L, 
            delta_p_i=delta_p_i, Q=Q, t=t
        )


# Функции для удобного использования
def convert_to_dimensionless_curves(time: pd.Series,
                                   pressure: pd.Series,
                                   flow_rate: pd.Series,
                                   depression: pd.Series,
                                   well_params: Dict[str, float],
                                   x_mode: str = 'alt',
                                   ) -> DimensionlessParameters:
    """Конвертация в безразмерные кривые"""
    converter = DimensionlessConverter()
    return converter.convert_to_dimensionless(time, pressure, flow_rate, depression, well_params, x_mode=x_mode)


def fit_xy_curve_coefficients(
    X_data: np.ndarray,
    Y_data: np.ndarray,
    X_calc: np.ndarray,
    Y_calc: np.ndarray,
    a_range: np.ndarray = None,
    b_range: np.ndarray = None,
    n_points: int = 200,
    fit_only_y: bool = False
) -> Dict[str, Any]:
    """
    Подбор коэффициентов поправки для расчётной кривой X-Y, чтобы она совпадала с эталонной.
    Использует OLS-регрессию с квадратичным членом: Y_fit = a * Y + b + c * (Y**2)
    
    Args:
        X_data: Эталонные значения X из данных
        Y_data: Эталонные значения Y из данных
        X_calc: Расчётные значения X
        Y_calc: Расчётные значения Y
        a_range: Диапазон перебора коэффициента a (по X). Игнорируется, используется OLS для X
        b_range: Диапазон перебора коэффициента b (по Y). Игнорируется, используется OLS для Y
        n_points: Количество точек для перебора (игнорируется)
        fit_only_y: Если True, подгоняется только Y (коэффициент a = 1.0)
    
    Returns:
        Словарь с результатами:
        - 'a': лучший коэффициент для X
        - 'b': лучший коэффициент для Y
        - 'c': коэффициент квадратичного члена для Y
        - 'rmse': RMSE ошибка после подгонки
        - 'rmse_before': RMSE ошибка до подгонки
        - 'accuracy': точность в процентах
        - 'r2': грубая оценка R²
        - 'X_fitted': подогнанные значения X
        - 'Y_fitted': подогнанные значения Y
        - 'c_clipped': True если c был обрезан до bounds
        - 'fallback_used': True если использован fallback (c=0)
    """
    # Убираем NaN и Inf
    mask = np.isfinite(X_data) & np.isfinite(Y_data) & np.isfinite(X_calc) & np.isfinite(Y_calc)
    X_data_clean = X_data[mask]
    Y_data_clean = Y_data[mask]
    X_calc_clean = X_calc[mask]
    Y_calc_clean = Y_calc[mask]
    
    if len(X_data_clean) < 2:
        return {
            'a': 1.0,
            'b': 0.0,
            'c': 0.0,
            'rmse': np.inf,
            'rmse_before': np.inf,
            'accuracy': 0.0,
            'r2': 0.0,
            'X_fitted': X_calc,
            'Y_fitted': Y_calc,
            'c_clipped': False,
            'fallback_used': False
        }
    
    # Интерполяция для выравнивания по X (чтобы длины совпадали)
    # Интерполируем Y_calc и X_calc на сетку X_data_clean
    X_calc_unique = None
    Y_calc_unique = None
    try:
        # Сортируем по X для интерполяции
        sort_idx = np.argsort(X_calc_clean)
        X_calc_sorted = X_calc_clean[sort_idx]
        Y_calc_sorted = Y_calc_clean[sort_idx]
        
        # Убираем дубликаты для интерполяции
        unique_mask = np.concatenate(([True], np.diff(X_calc_sorted) > 1e-10))
        X_calc_unique = X_calc_sorted[unique_mask]
        Y_calc_unique = Y_calc_sorted[unique_mask]
        
        # Интерполируем Y_calc на сетку X_data_clean
        Y_calc_interp = np.interp(X_data_clean, X_calc_unique, Y_calc_unique)
        
        # Заменяем NaN на линейную интерполяцию, если есть
        nan_mask = np.isnan(Y_calc_interp)
        if np.any(nan_mask):
            valid_mask = ~nan_mask
            if np.any(valid_mask):
                Y_calc_interp[nan_mask] = np.interp(
                    X_data_clean[nan_mask],
                    X_data_clean[valid_mask],
                    Y_calc_interp[valid_mask]
                )
    except Exception:
        # Если интерполяция не удалась, используем исходные данные
        Y_calc_interp = Y_calc_clean
        if len(X_data_clean) != len(X_calc_clean):
            # Если длины не совпадают, используем только общие точки
            min_len = min(len(X_data_clean), len(X_calc_clean))
            X_data_clean = X_data_clean[:min_len]
            Y_data_clean = Y_data_clean[:min_len]
            Y_calc_interp = Y_calc_clean[:min_len]
    
    # Вычисляем RMSE до подгонки (линейная подгонка: Y_fit = Y_calc)
    rmse_before = np.sqrt(np.mean((Y_data_clean - Y_calc_interp) ** 2))
    
    # Подгонка X: если fit_only_y, то a = 1.0, иначе используем OLS
    if fit_only_y:
        a = 1.0
    else:
        # OLS для X: X_fit = a * X_calc
        # Интерполируем X_calc на сетку X_data_clean для подгонки
        if X_calc_unique is not None:
            X_calc_for_fit = np.interp(X_data_clean, X_calc_unique, X_calc_unique)
        else:
            X_calc_for_fit = X_calc_clean[:len(X_data_clean)] if len(X_calc_clean) > len(X_data_clean) else X_calc_clean
        
        if len(X_calc_for_fit) > 0 and len(X_calc_for_fit) == len(X_data_clean) and np.any(X_calc_for_fit != 0):
            # Используем метод наименьших квадратов: a = (X_data^T * X_calc) / (X_calc^T * X_calc)
            a = np.dot(X_data_clean, X_calc_for_fit) / np.dot(X_calc_for_fit, X_calc_for_fit)
            if not np.isfinite(a):
                a = 1.0
        else:
            a = 1.0
    
    # Подгонка Y с квадратичным членом: Y_fit = a_y * Y + b + c * (Y**2)
    # Нормализация Y для числовой устойчивости
    Y_median = np.median(Y_calc_interp)
    if Y_median == 0 or not np.isfinite(Y_median):
        Y_median = 1.0
    
    Y_norm = Y_calc_interp / Y_median
    
    # Строим матрицу A для OLS: A = [Y_norm, 1, Y_norm^2]
    A = np.column_stack([Y_norm, np.ones_like(Y_norm), Y_norm ** 2])
    
    # Решаем least squares: A * θ = Y_data, где θ = (a_n, b, c_n)
    try:
        theta, residuals, rank, s = np.linalg.lstsq(A, Y_data_clean, rcond=None)
        a_n, b, c_n = theta[0], theta[1], theta[2]
    except Exception:
        # Fallback на линейную регрессию
        A_linear = np.column_stack([Y_norm, np.ones_like(Y_norm)])
        theta_linear, _, _, _ = np.linalg.lstsq(A_linear, Y_data_clean, rcond=None)
        a_n, b = theta_linear[0], theta_linear[1]
        c_n = 0.0
    
    # Преобразуем коэффициенты обратно в исходный масштаб
    a_y = a_n / Y_median
    c = c_n / (Y_median ** 2)
    
    # Ограничиваем c в пределах [-0.2, 0.2]
    c_clipped = False
    if c < -0.2:
        c = -0.2
        c_clipped = True
    elif c > 0.2:
        c = 0.2
        c_clipped = True
    
    # Вычисляем Y_fit с квадратичным членом
    Y_fit = a_y * Y_calc_interp + b + c * (Y_calc_interp ** 2)
    
    # Вычисляем RMSE после подгонки
    rmse_after = np.sqrt(np.mean((Y_data_clean - Y_fit) ** 2))
    
    # Fallback: если RMSE ухудшился, используем c=0
    fallback_used = False
    if rmse_after >= rmse_before:
        c = 0.0
        Y_fit = a_y * Y_calc_interp + b
        rmse_after = np.sqrt(np.mean((Y_data_clean - Y_fit) ** 2))
        fallback_used = True
    
    # Вычисляем финальные подогнанные значения для всех точек
    # Применяем коэффициенты к исходным массивам
    if fit_only_y:
        X_fitted = X_calc.copy()
    else:
        X_fitted = X_calc * a
    
    Y_fitted = a_y * Y_calc + b + c * (Y_calc ** 2)
    
    # Вычисляем метрики
    y_mean = np.nanmean(Y_data_clean)
    ss_tot = np.sum((Y_data_clean - y_mean) ** 2)
    if ss_tot > 0:
        r2 = 1 - (rmse_after ** 2 * len(Y_data_clean)) / ss_tot
    else:
        r2 = 0.0
    
    # Точность в процентах (нормализованная)
    y_range = np.nanmax(Y_data_clean) - np.nanmin(Y_data_clean)
    if y_range > 0:
        accuracy = max(0, (1 - rmse_after / y_range) * 100)
    else:
        accuracy = 0.0
    
    return {
        'a': a,  # Коэффициент для X
        'a_y': a_y,  # Коэффициент для Y (линейный член)
        'b': b,  # Свободный член для Y
        'c': c,  # Квадратичный коэффициент для Y
        'rmse': rmse_after,
        'rmse_before': rmse_before,
        'accuracy': accuracy,
        'r2': r2,
        'X_fitted': X_fitted,
        'Y_fitted': Y_fitted,
        'c_clipped': c_clipped,
        'fallback_used': fallback_used
    }
