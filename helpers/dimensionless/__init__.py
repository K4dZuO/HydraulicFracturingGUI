"""
Пакет для работы с безразмерными кривыми МГРП.
Включает модуль фильтрации и интерполяции.
"""

from .filtration import (
    SignalFilters,
    PhysicsConstraints
)

__all__ = [
    'SignalFilters',
    'PhysicsConstraints'
]

