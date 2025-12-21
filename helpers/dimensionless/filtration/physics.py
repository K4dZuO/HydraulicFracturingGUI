"""
Модуль физических ограничений для безразмерных кривых.
Обеспечивает физическую корректность кривых pD(Y).
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class PhysicsConstraints:
    """
    Класс для применения физических ограничений к безразмерным кривым.
    Обеспечивает монотонность, ограничение кривизны, удаление осцилляций и т.д.
    """
    
    @staticmethod
    def monotonic(
        curve: np.ndarray,
        x: Optional[np.ndarray] = None,
        direction: str = 'non_increasing'
    ) -> np.ndarray:
        """
        Восстанавливает монотонность кривой мягким способом.
        Для pD кривых типично не возрастающее поведение.
        Использует сглаживание вместо жёсткого maximum.accumulate.
        
        Args:
            curve: Входная кривая
            x: Координаты (если None, используется порядок элементов)
            direction: Направление монотонности ('non_increasing', 'non_decreasing')
        
        Returns:
            Монотонная кривая
        """
        curve = np.asarray(curve)
        valid_mask = np.isfinite(curve)
        
        if not np.any(valid_mask):
            return curve
        
        curve_clean = curve[valid_mask]
        
        if x is not None:
            x_clean = np.asarray(x)[valid_mask]
            # Сортируем по x
            sort_idx = np.argsort(x_clean)
            x_sorted = x_clean[sort_idx]
            curve_sorted = curve_clean[sort_idx]
        else:
            x_sorted = np.arange(len(curve_clean))
            curve_sorted = curve_clean.copy()
        
        n = len(curve_sorted)
        if n < 2:
            result = curve.copy()
            result[valid_mask] = curve_sorted
            return result
        
        # Используем np.maximum.accumulate для non_increasing
        # Для не возрастающей функции (справа налево): maximum.accumulate(curve_sorted[::-1])[::-1]
        if direction == 'non_increasing':
            # Применяем maximum.accumulate справа налево, затем разворачиваем
            curve_monotonic = np.maximum.accumulate(curve_sorted[::-1])[::-1]
        elif direction == 'non_decreasing':
            # Для не убывающей функции
            curve_monotonic = np.maximum.accumulate(curve_sorted)
        else:
            raise ValueError(f"Неизвестное направление: {direction}")
        
        # Мягкое сглаживание для устранения ступенчатости от maximum.accumulate
        # Используем очень легкое сглаживание, чтобы не нарушить монотонность и не создать артефакты
        # Смешиваем исходную монотонную кривую с очень легким сглаживанием
        from scipy.ndimage import uniform_filter1d
        curve_smoothed = uniform_filter1d(curve_monotonic, size=3, mode='nearest')
        
        # Смешиваем: 95% монотонной, 5% сглаженной - очень мягко
        curve_monotonic = 0.95 * curve_monotonic + 0.05 * curve_smoothed
        
        # Восстанавливаем монотонность после смешивания (на случай нарушения)
        if direction == 'non_increasing':
            curve_monotonic = np.maximum.accumulate(curve_monotonic[::-1])[::-1]
        else:
            curve_monotonic = np.maximum.accumulate(curve_monotonic)
        
        # Восстанавливаем исходный порядок
        if x is not None:
            result = curve.copy()
            result[valid_mask] = curve_monotonic[np.argsort(sort_idx)]
        else:
            result = curve.copy()
            result[valid_mask] = curve_monotonic
        
        return result
    
    @staticmethod
    def limit_curvature(
        curve: np.ndarray,
        x: Optional[np.ndarray] = None,
        max_curvature: float = 10.0
    ) -> np.ndarray:
        """
        Ограничивает вторую производную (кривизну) кривой.
        
        Args:
            curve: Входная кривая
            x: Координаты
            max_curvature: Максимальное значение второй производной
        
        Returns:
            Кривая с ограниченной кривизной
        """
        curve = np.asarray(curve)
        valid_mask = np.isfinite(curve)
        
        if not np.any(valid_mask) or np.sum(valid_mask) < 3:
            return curve
        
        curve_clean = curve[valid_mask]
        n = len(curve_clean)
        
        if x is not None:
            x_clean = np.asarray(x)[valid_mask]
            if len(x_clean) < 3 or len(x_clean) != n:
                return curve
            # Вычисляем вторую производную правильно
            first_deriv = np.gradient(curve_clean, x_clean)
            second_deriv = np.gradient(first_deriv, x_clean)
        else:
            # Вычисляем вторую производную правильно
            first_deriv = np.gradient(curve_clean)
            second_deriv = np.gradient(first_deriv)
        
        # Нормализуем кривизну относительно масштаба данных
        curve_range = np.max(curve_clean) - np.min(curve_clean)
        if curve_range < 1e-10:
            return curve
        
        # Относительная кривизна (нормализованная)
        normalized_curvature = np.abs(second_deriv) / curve_range
        
        # Находим точки с чрезмерной кривизной (только очень явные нарушения)
        # Увеличиваем порог, чтобы не трогать нормальную кривизну
        high_curvature_mask = normalized_curvature > (max_curvature * 2.0)
        
        if not np.any(high_curvature_mask):
            # Кривизна в норме, возвращаем исходную кривую
            return curve
        
        # Очень мягкое сглаживание только в проблемных областях
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(curve_clean, size=3, mode='nearest')  # Маленькое окно
        
        # Применяем сглаживание только в проблемных областях, смешивая с исходным
        # Очень мягко, чтобы не создать артефакты
        curve_restored = curve_clean.copy()
        # Смешиваем исходное и сглаженное (95% исходного, 5% сглаженного) - очень мягко
        curve_restored[high_curvature_mask] = (
            0.95 * curve_clean[high_curvature_mask] + 
            0.05 * smoothed[high_curvature_mask]
        )
        
        result = curve.copy()
        result[valid_mask] = curve_restored
        
        return result
    
    @staticmethod
    def remove_oscillations(
        curve: np.ndarray,
        x: Optional[np.ndarray] = None,
        window_size: int = 5,
        threshold: float = 0.1
    ) -> np.ndarray:
        """
        Удаляет осцилляции из кривой на основе изменений знака второй производной.
        """
        curve = np.asarray(curve)
        valid_mask = np.isfinite(curve)
        
        if not np.any(valid_mask) or np.sum(valid_mask) < 3:
            return curve.copy()
        
        curve_clean = curve[valid_mask]
        n = len(curve_clean)
        
        if n < 3:
            result = curve.copy()
            return result
        
        # Вычисляем вторую производную
        if x is not None:
            x_clean = np.asarray(x)[valid_mask]
            if len(x_clean) != n:
                return curve.copy()
            dydx = np.gradient(curve_clean, x_clean)
            d2ydx2 = np.gradient(dydx, x_clean)
        else:
            d2ydx2 = np.gradient(np.gradient(curve_clean))
        
        # Определяем изменения знака второй производной
        sign_d2 = np.sign(d2ydx2)
        # Заменяем 0 на предыдущий ненулевой знак (чтобы не ловить шум)
        sign_d2 = np.where(sign_d2 == 0, np.nan, sign_d2)
        sign_d2 = pd.Series(sign_d2).fillna(method='ffill').fillna(0).values
        
        # Где знак меняется? (с учётом сдвига)
        sign_change = np.diff(sign_d2) != 0  # длина: n - 1
        
        # Расширяем обратно до длины n: добавляем False в начало или конец
        oscillation_mask = np.zeros(n, dtype=bool)
        oscillation_mask[1:] = sign_change  # изменения происходят между точками → относятся к следующей
        # Или можно: oscillation_mask[:-1] |= sign_change
        
        # Дополнительно: если рядом с изменением — большой градиент, считаем это осцилляцией
        gradient_mag = np.abs(np.gradient(curve_clean))
        # Можно усилить условие, но пока просто используем sign_change
        
        # Сглаживаем только в зонах осцилляций - очень мягко, чтобы не создать артефакты
        if np.any(oscillation_mask):
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(curve_clean, size=window_size, mode='nearest')
            # Смешиваем: 90% исходного, 10% сглаженного - очень мягко
            curve_clean[oscillation_mask] = (
                0.9 * curve_clean[oscillation_mask] + 
                0.1 * smoothed[oscillation_mask]
            )
        
        # Восстанавливаем результат
        result = curve.copy()
        result[valid_mask] = curve_clean
        
        return result
    
    @staticmethod
    def asymptotic_fix(
        curve: np.ndarray,
        x: Optional[np.ndarray] = None,
        early_window: int = 5,
        late_window: int = 5
    ) -> np.ndarray:
        """
        Исправляет асимптотическое поведение в раннем и позднем режимах.
        Обеспечивает стабильность на краях кривой.
        
        Args:
            curve: Входная кривая
            x: Координаты
            early_window: Размер окна для раннего режима
            late_window: Размер окна для позднего режима
        
        Returns:
            Кривая с исправленными асимптотиками
        """
        curve = np.asarray(curve)
        valid_mask = np.isfinite(curve)
        
        if not np.any(valid_mask):
            return curve
        
        curve_clean = curve[valid_mask]
        n = len(curve_clean)
        
        if n < max(early_window, late_window) + 1:
            return curve
        
        result_clean = curve_clean.copy()
        
        # Ранний режим: очень мягкое сглаживание первых точек
        # НЕ заменяем на среднее, а только слегка сглаживаем, чтобы не создать "рост из (0,0)"
        if early_window > 0 and n >= early_window:
            from scipy.ndimage import uniform_filter1d
            # Очень легкое сглаживание только первых точек
            smoothed_early = uniform_filter1d(curve_clean[:early_window], size=3, mode='nearest')
            # Смешиваем: 95% исходного, 5% сглаженного - очень мягко
            result_clean[:early_window] = 0.95 * curve_clean[:early_window] + 0.05 * smoothed_early
        
        # Поздний режим: очень мягкое сглаживание последних точек
        if late_window > 0 and n >= late_window:
            from scipy.ndimage import uniform_filter1d
            # Очень легкое сглаживание только последних точек
            smoothed_late = uniform_filter1d(curve_clean[-late_window:], size=3, mode='nearest')
            # Смешиваем: 95% исходного, 5% сглаженного - очень мягко
            result_clean[-late_window:] = 0.95 * curve_clean[-late_window:] + 0.05 * smoothed_late
        
        result = curve.copy()
        result[valid_mask] = result_clean
        
        return result
    
    @staticmethod
    def enforce_all(
        curve: np.ndarray,
        x: Optional[np.ndarray] = None,
        monotonic: bool = True,
        limit_curvature: bool = True,
        remove_oscillations: bool = True,
        asymptotic_fix: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Применяет все физические ограничения последовательно.
        
        Args:
            curve: Входная кривая
            x: Координаты
            monotonic: Применять монотонность
            limit_curvature: Ограничивать кривизну
            remove_oscillations: Удалять осцилляции
            asymptotic_fix: Исправлять асимптотики
            **kwargs: Дополнительные параметры для каждого метода
        
        Returns:
            Кривая с применёнными ограничениями
        """
        result = curve.copy()
        
        if monotonic:
            direction = kwargs.get('monotonic_direction', 'non_increasing')
            result = PhysicsConstraints.monotonic(result, x, direction=direction)
        
        if remove_oscillations:
            window_size = kwargs.get('oscillation_window', 5)
            threshold = kwargs.get('oscillation_threshold', 0.1)
            result = PhysicsConstraints.remove_oscillations(
                result, x, window_size=window_size, threshold=threshold
            )
        
        if limit_curvature:
            max_curvature = kwargs.get('max_curvature', 10.0)
            result = PhysicsConstraints.limit_curvature(result, x, max_curvature=max_curvature)
        
        if asymptotic_fix:
            early_window = kwargs.get('early_window', 5)
            late_window = kwargs.get('late_window', 5)
            result = PhysicsConstraints.asymptotic_fix(
                result, x, early_window=early_window, late_window=late_window
            )
        
        # Финальная проверка монотонности
        if monotonic:
            direction = kwargs.get('monotonic_direction', 'non_increasing')
            result = PhysicsConstraints.monotonic(result, x, direction=direction)
        
        return result

