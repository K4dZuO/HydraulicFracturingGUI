import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from helpers.timeseries import *


class TestComputeDerivative:
    """Тесты для функции compute_derivative"""
    
    def test_basic_numeric_derivative(self):
        """Тест производной для числовых данных"""
        x = pd.Series([1, 2, 3, 4, 5], name='x')
        y = pd.Series([1, 4, 9, 16, 25], name='y')  # y = x^2
        
        result = compute_derivative(y, x)
        
        # Производная x^2 = 2x
        expected = pd.Series([2, 4, 6, 8, 10], name='dy/dx')
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_constant_function(self):
        """Тест производной константной функции"""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([5, 5, 5, 5, 5])
        
        result = compute_derivative(y, x)
        
        # Производная константы = 0
        expected = pd.Series([0, 0, 0, 0, 0])
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)
    
    def test_datetime_index_derivative(self):
        """Тест производной с datetime индексом"""
        dates = pd.date_range('2023-01-01', periods=5, freq='H')
        x = pd.Series(dates, name='time')
        y = pd.Series([0, 1, 4, 9, 16], name='value')  # y растет квадратично со временем
        
        result = compute_derivative(y, x)
        
        # Проверяем, что результат имеет правильный тип и размер
        assert len(result) == 5
        assert result.name == 'dvalue/dtime'
        assert isinstance(result, pd.Series)
    
    def test_mismatched_lengths(self):
        """Тест на ошибку при несовпадающих длинах"""
        x = pd.Series([1, 2, 3])
        y = pd.Series([1, 4, 9, 16])
        
        with pytest.raises(ValueError, match="Длины x и y должны совпадать"):
            compute_derivative(y, x)
    
    def test_single_point(self):
        """Тест с одной точкой данных"""
        x = pd.Series([1])
        y = pd.Series([5])
        
        result = compute_derivative(y, x)
        
        # Для одной точки gradient вернет 0
        assert len(result) == 1
        assert result.iloc[0] == 0
    
    def test_linear_function(self):
        """Тест производной линейной функции"""
        x = pd.Series([0, 1, 2, 3, 4])
        y = pd.Series([2, 5, 8, 11, 14])  # y = 3x + 2
        
        result = compute_derivative(y, x)
        
        # Производная должна быть примерно 3 везде
        np.testing.assert_allclose(result, 3, rtol=1e-10)


class TestInterpolateSeries:
    """Тесты для функции interpolate_series"""
    
    def test_basic_linear_interpolation(self):
        """Тест линейной интерполяции"""
        y = pd.Series([1, np.nan, 3, np.nan, 5])
        
        result = interpolate_series(y, method='linear')
        
        expected = pd.Series([1, 2, 3, 4, 5])
        pd.testing.assert_series_equal(result, expected)
    
    def test_time_interpolation(self):
        """Тест временной интерполяции"""
        index = pd.date_range('2023-01-01', periods=5, freq='D')
        y = pd.Series([1, np.nan, 3, np.nan, 5], index=index)
        
        result = interpolate_series(y, method='time')
        
        # При равномерном интервале time ~ linear
        expected = pd.Series([1, 2, 3, 4, 5], index=index)
        pd.testing.assert_series_equal(result, expected)
    
    def test_non_datetime_index_falls_back_to_linear(self):
        """Тест, что для не-временного индекса используется линейная интерполяция"""
        y = pd.Series([1, np.nan, 3], index=[10, 20, 30])
        
        result = interpolate_series(y, method='time')
        
        expected = pd.Series([1, 2, 3], index=[10, 20, 30])
        pd.testing.assert_series_equal(result, expected)
    
    def test_no_nans_returns_original(self):
        """Тест, что Series без пропусков возвращается без изменений"""
        y = pd.Series([1, 2, 3, 4, 5])
        
        result = interpolate_series(y)
        
        pd.testing.assert_series_equal(result, y)
    
    def test_all_nans_returns_all_nans(self):
        """Тест, что все NaN остаются NaN после интерполяции"""
        y = pd.Series([np.nan, np.nan, np.nan])
        
        result = interpolate_series(y)
        
        # Все значения должны остаться NaN
        assert result.isna().all()
    
    def test_polynomial_interpolation(self):
        """Тест полиномиальной интерполяции"""
        y = pd.Series([1, np.nan, 9])  # x^2 в точках 1, 2, 3
        
        result = interpolate_series(y, method='polynomial', order=2)
        
        expected = pd.Series([1, 4, 9])  # 2^2 = 4
        pd.testing.assert_series_equal(result, expected)
    
    def test_unsupported_method_falls_back(self):
        """Тест обработки неподдерживаемого метода интерполяции"""
        y = pd.Series([1, np.nan, 3])
        
        # Перехватываем любое исключение, так как функция ловит Exception
        result = interpolate_series(y, method='unsupported_method')
        
        # Должен вернуться результат линейной интерполяции
        assert not result.isna().any()


class TestSmoothSeries:
    """Тесты для функции smooth_series"""
    
    def test_basic_smoothing(self):
        """Тест базового сглаживания"""
        y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        result = smooth_series(y, window=3)
        
        # Первые и последние значения не полностью сглаживаются из-за center=True
        assert result.iloc[0] == 1  # Первое значение не меняется
        assert abs(result.iloc[2] - 3) < 1e-10  # Среднее 2,3,4 = 3
    
    def test_window_size_1(self):
        """Тест с окном размером 1 (без сглаживания)"""
        y = pd.Series([1, 2, 3, 4, 5])
        
        result = smooth_series(y, window=1)
        
        pd.testing.assert_series_equal(result, y)
    
    def test_window_size_less_than_1(self):
        """Тест с окном меньше 1 (без сглаживания)"""
        y = pd.Series([1, 2, 3, 4, 5])
        
        result = smooth_series(y, window=0)
        
        pd.testing.assert_series_equal(result, y)
    
    def test_smoothing_with_nans(self):
        """Тест сглаживания с NaN значениями"""
        y = pd.Series([1, np.nan, 3, 4, 5])
        
        result = smooth_series(y, window=3)
        
        # NaN влияет на сглаживание
        assert np.isnan(result.iloc[1])
    
    def test_large_window(self):
        """Тест с окном больше длины ряда"""
        y = pd.Series([1, 2, 3])
        
        result = smooth_series(y, window=10)
        
        # Должно работать, min_periods=1
        assert len(result) == 3
        assert result.iloc[1] == 2  # Среднее всех значений
    
    def test_constant_series(self):
        """Тест сглаживания константного ряда"""
        y = pd.Series([5, 5, 5, 5, 5])
        
        result = smooth_series(y, window=3)
        
        pd.testing.assert_series_equal(result, y)
    
    def test_noisy_data_smoothing(self):
        """Тест сглаживания зашумленных данных"""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = pd.Series(np.sin(x) + np.random.normal(0, 0.1, len(x)))
        
        smoothed = smooth_series(y, window=5)
        
        # Сглаженные данные должны иметь меньшую дисперсию
        assert smoothed.std() < y.std()
        # И быть коррелированными
        assert abs(y.corr(smoothed)) > 0.8
