"""
Модуль фильтрации безразмерных кривых для МГРП.
Реализует физически корректную фильтрацию и унифицированную интерполяцию.
"""

from .filters import SignalFilters
from .physics import PhysicsConstraints
from .utils import (
    compute_snr,
    compute_oscillation_score,
    detect_log_scale,
    select_filter_method,
    fill_missing_values,
)

# Опциональный импорт ML-денойзера
try:
    __all__ = [
        'SignalFilters',
        'PhysicsConstraints',
        'compute_snr',
        'compute_oscillation_score',
        'detect_log_scale',
        'select_filter_method',
        'fill_missing_values',
    ]
except ImportError:
    __all__ = [
        'SignalFilters',
        'PhysicsConstraints',
        'compute_snr',
        'compute_oscillation_score',
        'detect_log_scale',
        'select_filter_method',
        'fill_missing_values',
    ]

