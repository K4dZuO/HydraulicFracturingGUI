"""
Модуль нормированной метрики валидации Y.
Включает filtered_RMSE_Y, RMSE_log_Y, physically_acceptable_share.
"""

import numpy as np
from typing import Dict, Optional


def compute_validation_metrics_y(
    Y_ref: np.ndarray,
    Y_pred: np.ndarray,
    dP_ref: Optional[np.ndarray] = None,
    dP_pred: Optional[np.ndarray] = None,
    dP_threshold: float = 1e-6
) -> Dict[str, float]:
    """
    Вычисляет нормированные метрики валидации для Y.
    
    Args:
        Y_ref: Эталонные значения Y
        Y_pred: Предсказанные значения Y
        dP_ref: Эталонные значения dP (для фильтрации)
        dP_pred: Предсказанные значения dP (для фильтрации)
        dP_threshold: Порог для валидности dP
    
    Returns:
        Словарь с метриками:
        - filtered_RMSE_Y: RMSE по Y с маской dP > threshold
        - RMSE_log_Y: RMSE в логарифмическом пространстве
        - physically_acceptable_share: Процент валидных точек
    """
    # Создаем маску валидности
    if dP_ref is not None and dP_pred is not None:
        # Используем обе маски (эталон и предсказание)
        valid_mask_ref = np.abs(dP_ref) >= dP_threshold
        valid_mask_pred = np.abs(dP_pred) >= dP_threshold
        valid_mask = valid_mask_ref & valid_mask_pred
    elif dP_ref is not None:
        valid_mask = np.abs(dP_ref) >= dP_threshold
    elif dP_pred is not None:
        valid_mask = np.abs(dP_pred) >= dP_threshold
    else:
        # Если dP не передан, используем маску по валидности Y
        valid_mask = np.isfinite(Y_ref) & np.isfinite(Y_pred) & (Y_ref > 0) & (Y_pred > 0)
    
    # Фильтруем невалидные значения
    Y_ref_valid = Y_ref[valid_mask]
    Y_pred_valid = Y_pred[valid_mask]
    
    # Процент валидных точек
    physically_acceptable_share = np.sum(valid_mask) / len(valid_mask) if len(valid_mask) > 0 else 0.0
    
    if len(Y_ref_valid) == 0:
        return {
            'filtered_RMSE_Y': np.inf,
            'RMSE_log_Y': np.inf,
            'physically_acceptable_share': 0.0
        }
    
    # filtered_RMSE_Y (маска dP > threshold)
    filtered_RMSE_Y = np.sqrt(np.nanmean((Y_ref_valid - Y_pred_valid)**2))
    
    # RMSE_log_Y (в логарифмическом пространстве)
    # Защита от нулей и отрицательных значений
    Y_ref_log = np.log10(np.maximum(Y_ref_valid, 1e-30))
    Y_pred_log = np.log10(np.maximum(Y_pred_valid, 1e-30))
    RMSE_log_Y = np.sqrt(np.nanmean((Y_ref_log - Y_pred_log)**2))
    
    return {
        'filtered_RMSE_Y': float(filtered_RMSE_Y),
        'RMSE_log_Y': float(RMSE_log_Y),
        'physically_acceptable_share': float(physically_acceptable_share)
    }

