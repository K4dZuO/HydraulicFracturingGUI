"""
Модуль экстраполяции безразмерных кривых для МГРП.
Реализует экстраполяцию размерных параметров и вычисление безразмерных X, Y.
"""

from .extrapolator import extrapolate_parameters
from .validation import compute_validation_metrics_y
from .eri import ExtrapolationReliabilityEvaluator

__all__ = [
    'extrapolate_parameters',
    'compute_validation_metrics_y',
    'ExtrapolationReliabilityEvaluator'
]

