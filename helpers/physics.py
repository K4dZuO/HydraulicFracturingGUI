import numpy as np
from typing import Tuple


def compute_transmissivity(permeability_md: np.ndarray, thickness_m: np.ndarray) -> np.ndarray:
    """
    Векторизованно вычисляет трансмиссивность T = k * h.
    Ожидает согласованные массивы одинаковой длины.
    """
    return np.multiply(permeability_md, thickness_m)


def compute_pore_volume(porosity: np.ndarray, thickness_m: np.ndarray) -> np.ndarray:
    """
    Векторизованно вычисляет удельный поровый объём на колонну: Vp = phi * h.
    """
    return np.multiply(porosity, thickness_m)


def compute_darcy_flux(permeability_m2: np.ndarray, viscosity_pa_s: np.ndarray, pressure_grad_pa_per_m: np.ndarray) -> np.ndarray:
    """
    Закон Дарси в одномерной форме (удельный расход): q = - (k / mu) * dp/dx.
    Знак минус отражает направление потока по убыванию давления.
    """
    return -np.divide(permeability_m2, viscosity_pa_s) * pressure_grad_pa_per_m


def compute_diffusivity(permeability_m2: np.ndarray, compressibility_pa_inv: np.ndarray, viscosity_pa_s: np.ndarray, porosity: np.ndarray) -> np.ndarray:
    """
    Давление-диффузивность: alpha = k / (mu * c_t * phi).
    """
    denominator = np.multiply(viscosity_pa_s, np.multiply(compressibility_pa_inv, porosity))
    return np.divide(permeability_m2, denominator)
