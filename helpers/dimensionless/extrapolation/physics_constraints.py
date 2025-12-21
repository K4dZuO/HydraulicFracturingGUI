"""
Модуль физических ограничений для экстраполяции.
Обеспечивает монотонность, положительность и физическую корректность предсказаний.
"""

import numpy as np
from typing import Tuple


def apply_physical_constraints(
    P_ext: np.ndarray,
    dP_ext: np.ndarray,
    Q_ext: np.ndarray,
    P_last: float,
    dP_last: float,
    Q_last: float,
    enforce_dP_monotonicity: bool = True,
    enforce_P_monotonicity: bool = True,
    max_Q_ratio: float = 10.0,
    min_Q_ratio: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Применяет физические ограничения к экстраполированным значениям.
    
    Args:
        P_ext: Экстраполированные значения давления
        dP_ext: Экстраполированные значения депрессии
        Q_ext: Экстраполированные значения дебита
        P_last: Последнее значение давления из исторических данных
        dP_last: Последнее значение депрессии из исторических данных
        Q_last: Последнее значение дебита из исторических данных
        enforce_dP_monotonicity: Принудительно обеспечить монотонное возрастание dP
        enforce_P_monotonicity: Принудительно обеспечить монотонное убывание P
        max_Q_ratio: Максимальный коэффициент изменения Q относительно последнего значения
        min_Q_ratio: Минимальный коэффициент изменения Q относительно последнего значения
    
    Returns:
        (P_ext_corrected, dP_ext_corrected, Q_ext_corrected): Исправленные значения
    """
    
    # 1. Ограничения на давление P
    if enforce_P_monotonicity:
        # Давление должно только падать (или оставаться постоянным)
        # Ограничиваем максимальное значение последним известным
        P_ext = np.minimum(P_ext, P_last)
        
        # Обеспечиваем монотонное убывание
        for i in range(1, len(P_ext)):
            P_ext[i] = min(P_ext[i], P_ext[i-1])
        
        # Защита от отрицательных значений
        P_ext = np.maximum(P_ext, 0.0)
    
    # 2. Ограничения на депрессию dP
    # dP должно быть положительным
    dP_ext = np.maximum(dP_ext, 0.0)
    
    if enforce_dP_monotonicity:
        # Обеспечиваем, что dP начинается с последнего известного значения
        if len(dP_ext) > 0:
            # Если первое значение меньше последнего известного, устанавливаем его
            if dP_ext[0] < dP_last:
                dP_ext[0] = dP_last
            
            # Обеспечиваем монотонное возрастание
            dP_ext = np.maximum.accumulate(dP_ext)
    
    # 3. Ограничения на дебит Q
    # Q должно быть положительным
    Q_ext = np.maximum(Q_ext, 0.0)
    
    # Ограничиваем разумными пределами относительно последнего значения
    Q_max = Q_last * max_Q_ratio
    Q_min = Q_last * min_Q_ratio
    Q_ext = np.clip(Q_ext, Q_min, Q_max)
    
    # 4. Согласованность между P и dP
    # Если есть начальное давление, можем проверить согласованность
    # dP должно соответствовать изменению давления
    # Но так как мы не знаем P_start здесь, просто обеспечиваем положительность
    
    return P_ext, dP_ext, Q_ext
