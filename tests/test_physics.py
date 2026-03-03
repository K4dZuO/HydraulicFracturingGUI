import pytest
import numpy as np
from typing import Tuple
from helpers.physics import *


class TestComputeTransmissivity:
    """Тесты для функции compute_transmissivity"""
    
    def test_basic_calculation(self):
        """Проверка базового расчета трансмиссивности"""
        permeability = np.array([100, 200, 300])  # мД
        thickness = np.array([10, 20, 30])  # м
        expected = np.array([1000, 4000, 9000])
        
        result = compute_transmissivity(permeability, thickness)
        np.testing.assert_array_equal(result, expected)
    
    def test_single_values(self):
        """Проверка работы с одиночными значениями (массивы из одного элемента)"""
        permeability = np.array([50])
        thickness = np.array([5])
        expected = np.array([250])
        
        result = compute_transmissivity(permeability, thickness)
        np.testing.assert_array_equal(result, expected)
    
    def test_negative_values(self):
        """Проверка обработки отрицательных значений"""
        permeability = np.array([-100, 200, -300])
        thickness = np.array([10, -20, 30])
        expected = np.array([-1000, -4000, -9000])
        
        result = compute_transmissivity(permeability, thickness)
        np.testing.assert_array_equal(result, expected)
    
    def test_zero_values(self):
        """Проверка обработки нулевых значений"""
        permeability = np.array([0, 100, 0])
        thickness = np.array([10, 0, 30])
        expected = np.array([0, 0, 0])
        
        result = compute_transmissivity(permeability, thickness)
        np.testing.assert_array_equal(result, expected)
    
    def test_shape_mismatch(self):
        """Проверка обработки несовпадающих размеров массивов"""
        permeability = np.array([100, 200])
        thickness = np.array([10, 20, 30])
        
        # Должно вызвать ошибку при умножении
        with pytest.raises(ValueError):
            compute_transmissivity(permeability, thickness)
        # Проверяем, что результат имеет правильную форму (broadcasting)
    
    def test_float_values(self):
        """Проверка работы с дробными числами"""
        permeability = np.array([100.5, 200.75])
        thickness = np.array([10.25, 20.5])
        expected = np.array([100.5 * 10.25, 200.75 * 20.5])
        
        result = compute_transmissivity(permeability, thickness)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)


class TestComputePoreVolume:
    """Тесты для функции compute_pore_volume"""
    
    def test_basic_calculation(self):
        """Проверка базового расчета порового объема"""
        porosity = np.array([0.2, 0.3, 0.4])
        thickness = np.array([10, 20, 30])
        expected = np.array([2.0, 6.0, 12.0])
        
        result = compute_pore_volume(porosity, thickness)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)
    
    def test_porosity_range(self):
        """Проверка работы с пористостью в допустимом диапазоне (0-1)"""
        porosity = np.array([0, 0.5, 1.0])
        thickness = np.array([10, 10, 10])
        expected = np.array([0, 5.0, 10.0])
        
        result = compute_pore_volume(porosity, thickness)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)
    
    def test_extreme_porosity(self):
        """Проверка работы с экстремальными значениями пористости"""
        porosity = np.array([0.001, 0.999])
        thickness = np.array([100, 100])
        expected = np.array([0.1, 99.9])
        
        result = compute_pore_volume(porosity, thickness)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)


class TestComputeDarcyFlux:
    """Тесты для функции compute_darcy_flux"""
    
    def test_basic_calculation(self):
        """Проверка базового расчета потока по закону Дарси"""
        permeability = np.array([1e-12, 2e-12])  # м²
        viscosity = np.array([1e-3, 1e-3])  # Па·с
        pressure_grad = np.array([1000, 2000])  # Па/м
        # q = -(k/mu) * dp/dx
        expected = np.array([
            -(1e-12 / 1e-3) * 1000,
            -(2e-12 / 1e-3) * 2000
        ])
        
        result = compute_darcy_flux(permeability, viscosity, pressure_grad)
        np.testing.assert_array_almost_equal(result, expected, decimal=20)
    
    def test_negative_pressure_gradient(self):
        """Проверка работы с отрицательным градиентом давления"""
        permeability = np.array([1e-12])
        viscosity = np.array([1e-3])
        pressure_grad = np.array([-1000])
        # Двойной минус даст положительный поток
        expected = np.array([(1e-12 / 1e-3) * 1000])
        
        result = compute_darcy_flux(permeability, viscosity, pressure_grad)
        np.testing.assert_array_almost_equal(result, expected, decimal=20)
    
    def test_zero_viscosity_handling(self):
        """Проверка обработки нулевой вязкости (деление на ноль)"""
        permeability = np.array([1e-12])
        viscosity = np.array([0.0])  # Нулевая вязкость
        pressure_grad = np.array([1000])
        
        # Ожидаем inf или ошибку деления на ноль
        result = compute_darcy_flux(permeability, viscosity, pressure_grad)
        assert np.isinf(result[0]) or np.isnan(result[0])
    
    def test_different_units_scaling(self):
        """Проверка масштабирования при разных единицах измерения"""
        # Проверяем, что функция корректно работает с малыми/большими значениями
        permeability = np.array([1e-15, 1e-10])  # Диапазон проницаемостей
        viscosity = np.array([0.001, 1.0])  # Разная вязкость
        pressure_grad = np.array([1e3, 1e6])  # Разные градиенты
        
        result = compute_darcy_flux(permeability, viscosity, pressure_grad)
        # Проверяем, что результаты имеют правильный знак и порядок
        assert np.all(result[0] < 0)  # Отрицательный поток
        assert np.all(result[1] < 0)  # Отрицательный поток


class TestComputeDiffusivity:
    """Тесты для функции compute_diffusivity"""
    
    def test_basic_calculation(self):
        """Проверка базового расчета диффузивности"""
        permeability = np.array([1e-12])  # м²
        compressibility = np.array([1e-9])  # 1/Па
        viscosity = np.array([1e-3])  # Па·с
        porosity = np.array([0.2])
        
        # alpha = k / (mu * c_t * phi)
        expected = 1e-12 / (1e-3 * 1e-9 * 0.2)
        
        result = compute_diffusivity(permeability, compressibility, viscosity, porosity)
        np.testing.assert_almost_equal(result[0], expected, decimal=10)
    
    def test_multiple_values(self):
        """Проверка расчета для нескольких значений"""
        permeability = np.array([1e-12, 2e-12])
        compressibility = np.array([1e-9, 2e-9])
        viscosity = np.array([1e-3, 2e-3])
        porosity = np.array([0.2, 0.3])
        
        expected = np.array([
            1e-12 / (1e-3 * 1e-9 * 0.2),
            2e-12 / (2e-3 * 2e-9 * 0.3)
        ])
        
        result = compute_diffusivity(permeability, compressibility, viscosity, porosity)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)
    
    def test_zero_porosity_handling(self):
        """Проверка обработки нулевой пористости (деление на ноль)"""
        permeability = np.array([1e-12])
        compressibility = np.array([1e-9])
        viscosity = np.array([1e-3])
        porosity = np.array([0.0])
        
        result = compute_diffusivity(permeability, compressibility, viscosity, porosity)
        assert np.isinf(result[0]) or np.isnan(result[0])
    
    def test_zero_compressibility_handling(self):
        """Проверка обработки нулевой сжимаемости (деление на ноль)"""
        permeability = np.array([1e-12])
        compressibility = np.array([0.0])
        viscosity = np.array([1e-3])
        porosity = np.array([0.2])
        
        result = compute_diffusivity(permeability, compressibility, viscosity, porosity)
        assert np.isinf(result[0]) or np.isnan(result[0])
    
    def test_physical_plausibility(self):
        """Проверка физической правдоподобности результатов"""
        # Типичные значения для нефтяного пласта
        permeability = np.array([1e-13])  # 100 мД ≈ 1e-13 м²
        compressibility = np.array([1e-9])  # 1e-9 1/Па
        viscosity = np.array([1e-3])  # 1 сПз = 1e-3 Па·с
        porosity = np.array([0.2])
        
        result = compute_diffusivity(permeability, compressibility, viscosity, porosity)
        
        # Диффузивность должна быть положительной
        assert result[0] > 0
        # Типичные значения ~ 0.1-10 м²/с
        assert 0.01 < result[0] < 100