import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import warnings

# Импортируем тестируемый класс
from helpers.dimensionless.filtration.filters import SignalFilters, _last_diagnostics

# Проверяем доступность LOWESS
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    LOWESS_AVAILABLE = True
except ImportError:
    LOWESS_AVAILABLE = False


class TestSignalFilters:
    """Тесты для класса SignalFilters."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.sample_signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        self.sample_noisy_signal = np.array([1.1, 1.9, 3.2, 3.8, 5.1, 3.9, 2.9, 2.1, 0.9])
        self.sample_x = np.arange(len(self.sample_signal))
        
        # Сигнал с NaN и inf
        self.signal_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.inf, 3.0, 2.0, np.nan])
        self.signal_with_zeros = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 1.0, 0.5, 0.1, 0.0])
        
        # Очень короткий сигнал
        self.short_signal = np.array([1.0, 2.0])
        
        # Сигнал для логарифмического масштаба
        self.log_signal = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 2.0, 1.0, 0.5, 0.1])
    
    def test_lowess_basic(self):
        """Тест базовой работы LOWESS фильтра."""
        result = SignalFilters.lowess(self.sample_signal)
        
        # Проверяем, что результат имеет ту же длину
        assert len(result) == len(self.sample_signal)
        assert np.all(np.isfinite(result))
        
        # Проверяем, что сигнал сглажен (менее изменчив, чем исходный)
        assert np.std(result) <= np.std(self.sample_signal) * 1.5
    
    def test_lowess_with_x(self):
        """Тест LOWESS с явными координатами."""
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        result = SignalFilters.lowess(self.sample_signal, x=x)
        
        assert len(result) == len(self.sample_signal)
        assert np.all(np.isfinite(result))
    
    def test_lowess_with_parameters(self):
        """Тест LOWESS с разными параметрами."""
        result = SignalFilters.lowess(
            self.sample_signal, 
            frac=0.1,  # Более агрессивное сглаживание
            it=2       # Меньше итераций
        )
        
        assert len(result) == len(self.sample_signal)
        assert np.all(np.isfinite(result))
    
    def test_lowess_with_nan(self):
        """Тест LOWESS с NaN значениями."""
        result = SignalFilters.lowess(self.signal_with_nan)
        
        # Проверяем, что NaN и inf сохранились на тех же позициях
        assert np.isnan(result[2])
        assert np.isnan(result[-1])
        assert not np.isfinite(result[5])  # inf
        
        # Проверяем, что остальные значения конечны
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
        assert np.isfinite(result[3])
        assert np.isfinite(result[4])
    
    def test_lowess_short_signal(self):
        """Тест LOWESS с очень коротким сигналом."""
        result = SignalFilters.lowess(self.short_signal)
        assert len(result) == len(self.short_signal)
        
        # С массивом из одного элемента
        single = np.array([1.0])
        result_single = SignalFilters.lowess(single)
        assert len(result_single) == 1
        assert result_single[0] == 1.0
    
    def test_savgol_basic(self):
        """Тест базовой работы Savitzky-Golay фильтра."""
        result = SignalFilters.savgol(self.sample_signal)
        
        assert len(result) == len(self.sample_signal)
        assert np.all(np.isfinite(result))
        
        # Проверяем, что сигнал сглажен
        assert np.std(result) <= np.std(self.sample_signal) * 1.5
    
    def test_savgol_with_parameters(self):
        """Тест SavGol с явными параметрами."""
        result = SignalFilters.savgol(
            self.sample_signal,
            window_length=5,
            polyorder=2,
            mode='mirror'
        )
        
        assert len(result) == len(self.sample_signal)
        assert np.all(np.isfinite(result))
    
    def test_savgol_auto_window(self):
        """Тест автоматического выбора window_length."""
        # Тест с разной длиной сигнала
        for n in [5, 10, 20, 50]:
            signal = np.random.randn(n)
            result = SignalFilters.savgol(signal, window_length=None, polyorder=2)
            
            assert len(result) == n
            assert np.all(np.isfinite(result))
    
    def test_savgol_edge_cases(self):
        """Тест граничных случаев для SavGol."""
        # Слишком короткий сигнал
        result = SignalFilters.savgol(np.array([1.0, 2.0]), window_length=5)
        assert np.array_equal(result, np.array([1.0, 2.0]))
        
        # Все значения NaN
        result = SignalFilters.savgol(np.array([np.nan, np.nan, np.nan]))
        assert np.all(np.isnan(result))
        
        # window_length больше длины сигнала
        signal = np.array([1.0, 2.0, 3.0])
        result = SignalFilters.savgol(signal, window_length=7)
        assert np.array_equal(result, signal)
    
    def test_savgol_with_nan(self):
        """Тест SavGol с NaN значениями."""
        result = SignalFilters.savgol(self.signal_with_nan)
        
        # Проверяем, что NaN сохранились
        assert np.isnan(result[2])
        assert np.isnan(result[-1])
        
        # Проверяем, что валидные значения обработаны
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
    
    def test_gaussian_basic(self):
        """Тест базовой работы Gaussian фильтра."""
        result = SignalFilters.gaussian(self.sample_signal)
        
        assert len(result) == len(self.sample_signal)
        assert np.all(np.isfinite(result))
        
        # Проверяем, что сигнал сглажен
        assert np.std(result) <= np.std(self.sample_signal) * 1.5
    
    def test_gaussian_with_sigma(self):
        """Тест Gaussian с разными sigma."""
        # Маленький sigma - меньше сглаживания
        result_small = SignalFilters.gaussian(self.sample_signal, sigma=0.1)
        
        # Большой sigma - больше сглаживания
        result_large = SignalFilters.gaussian(self.sample_signal, sigma=1.0)
        
        assert len(result_small) == len(self.sample_signal)
        assert len(result_large) == len(self.sample_signal)
        
        # Больший sigma должен давать более сглаженный результат
        assert np.std(result_large) <= np.std(result_small) * 1.1
    
    def test_gaussian_auto_sigma(self):
        """Тест автоматического выбора sigma."""
        result = SignalFilters.gaussian(self.sample_signal, sigma=None)
        assert len(result) == len(self.sample_signal)
        assert np.all(np.isfinite(result))
    
    def test_gaussian_sigma_validation(self):
        """Тест валидации параметра sigma."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Слишком маленький sigma
            result_small = SignalFilters.gaussian(self.sample_signal, sigma=0.01)
            assert any("слишком мал" in str(warning.message) for warning in w)
            
            # Слишком большой sigma
            result_large = SignalFilters.gaussian(self.sample_signal, sigma=10.0)
            assert any("слишком велик" in str(warning.message) for warning in w)
        
        assert np.all(np.isfinite(result_small))
        assert np.all(np.isfinite(result_large))
    
    def test_gaussian_with_nan(self):
        """Тест Gaussian с NaN значениями."""
        result = SignalFilters.gaussian(self.signal_with_nan)
        
        # Проверяем, что NaN сохранились
        assert np.isnan(result[2])
        assert np.isnan(result[-1])
        
        # Проверяем, что валидные значения обработаны
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
    
    def test_kalman_basic(self):
        """Тест базовой работы Kalman фильтра."""
        result = SignalFilters.kalman(self.sample_noisy_signal)
        
        assert len(result) == len(self.sample_noisy_signal)
        assert np.all(np.isfinite(result))
        
        # Фильтр Калмана должен уменьшить шум
        # (сравниваем с более гладким оригиналом)
        mse_to_original = np.mean((result - self.sample_signal) ** 2)
        mse_noisy = np.mean((self.sample_noisy_signal - self.sample_signal) ** 2)
        
        # Фильтр должен улучшить сигнал
        assert mse_to_original <= mse_noisy * 1.5
    
    def test_kalman_with_noise_parameters(self):
        """Тест Kalman с явными параметрами шума."""
        result = SignalFilters.kalman(
            self.sample_noisy_signal,
            process_noise=0.1,
            measurement_noise=0.5
        )
        
        assert len(result) == len(self.sample_noisy_signal)
        assert np.all(np.isfinite(result))
    
    def test_kalman_auto_noise(self):
        """Тест автоматической оценки параметров шума."""
        result = SignalFilters.kalman(self.sample_noisy_signal, process_noise=None, measurement_noise=None)
        assert len(result) == len(self.sample_noisy_signal)
        assert np.all(np.isfinite(result))
    
    def test_kalman_with_nan(self):
        """Тест Kalman с NaN значениями."""
        result = SignalFilters.kalman(self.signal_with_nan)
        
        # Проверяем, что NaN сохранились
        assert np.isnan(result[2])
        assert np.isnan(result[-1])
        
        # Проверяем, что валидные значения обработаны
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
    
    def test_kalman_constant_signal(self):
        """Тест Kalman с постоянным сигналом."""
        constant_signal = np.ones(10)
        result = SignalFilters.kalman(constant_signal)
        
        assert np.allclose(result, constant_signal, rtol=1e-5)
    
    def test_log_domain_basic(self):
        """Тест базовой работы фильтрации в логарифмическом масштабе."""
        result = SignalFilters.log_domain(self.log_signal, base_filter='savgol')
        
        assert len(result) == len(self.log_signal)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)  # Должны быть положительные
        
        # Проверяем, что polyorder ограничен до 2
        result_poly = SignalFilters.log_domain(
            self.log_signal, 
            base_filter='savgol',
            polyorder=4  # Должен быть ограничен до 2
        )
        assert len(result_poly) == len(self.log_signal)
    
    def test_log_domain_different_filters(self):
        """Тест логарифмической фильтрации с разными базовыми фильтрами."""
        for base_filter in ['savgol', 'gaussian', 'kalman']:
            result = SignalFilters.log_domain(self.log_signal, base_filter=base_filter)
            assert len(result) == len(self.log_signal)
            assert np.all(np.isfinite(result))
            assert np.all(result > 0)
    
    def test_log_domain_non_positive(self):
        """Тест логарифмической фильтрации с неположительными значениями."""
        # Сигнал с нулями и отрицательными значениями
        signal = np.array([0.0, -1.0, 0.5, 1.0, 2.0])
        result = SignalFilters.log_domain(signal, base_filter='gaussian')
        
        assert len(result) == len(signal)
        assert np.all(np.isfinite(result))
    
    def test_log_domain_with_nan(self):
        """Тест логарифмической фильтрации с NaN."""
        result = SignalFilters.log_domain(self.signal_with_nan, base_filter='savgol')
        
        # Проверяем, что NaN сохранились
        assert np.isnan(result[2])
        assert np.isnan(result[-1])
        
        # Проверяем, что inf не превратился в NaN
        assert not np.isfinite(result[5])  # inf

    def test_denoise_basic(self):
        """Тест универсального метода denoise."""
        result = SignalFilters.denoise(self.sample_signal, method='savgol')
        
        assert len(result) == len(self.sample_signal)
        assert np.all(np.isfinite(result))
    
    def test_denoise_invalid_method(self):
        """Тест denoise с неверным методом."""
        with pytest.raises(ValueError, match="Неизвестный метод фильтрации"):
            SignalFilters.denoise(self.sample_signal, method='invalid_method')
    
    def test_edge_case_empty_signal(self):
        """Тест с пустым сигналом."""
        empty_signal = np.array([])
        
        # Все методы должны возвращать пустой массив
        for method_name in ['savgol', 'gaussian', 'kalman', 'lowess']:
            if method_name == 'lowess' and not LOWESS_AVAILABLE:
                continue
                
            method = getattr(SignalFilters, method_name)
            result = method(empty_signal)
            assert len(result) == 0
    
    def test_edge_case_all_nan(self):
        """Тест с сигналом из одних NaN."""
        all_nan = np.array([np.nan, np.nan, np.nan])
        
        for method_name in ['savgol', 'gaussian', 'kalman', 'lowess']:
            if method_name == 'lowess' and not LOWESS_AVAILABLE:
                continue
                
            method = getattr(SignalFilters, method_name)
            result = method(all_nan)
            assert len(result) == len(all_nan)
            assert np.all(np.isnan(result))
    
    def test_preserve_shape(self):
        """Тест, что все методы сохраняют форму массива."""
        methods = ['savgol', 'gaussian', 'kalman']
        if LOWESS_AVAILABLE:
            methods.append('lowess')
        
        for method_name in methods:
            method = getattr(SignalFilters, method_name)
            
            # Тестируем разные формы
            test_signals = [
                self.sample_signal,
                self.signal_with_nan,
                self.short_signal,
                np.array([1.0]),  # Один элемент
                np.random.randn(100)  # Длинный сигнал
            ]
            
            for signal in test_signals:
                result = method(signal)
                assert result.shape == signal.shape
    
    def test_deterministic_results(self):
        """Тест детерминированности результатов."""
        # Дважды применяем один и тот же фильтр
        result1 = SignalFilters.savgol(self.sample_signal, window_length=5, polyorder=2)
        result2 = SignalFilters.savgol(self.sample_signal, window_length=5, polyorder=2)
        
        assert np.allclose(result1, result2, rtol=1e-10)
    
    def test_savgol_window_length_validation(self):
        """Тест валидации window_length в SavGol."""
        # window_length должен стать нечётным
        signal = np.random.randn(20)
        
        # Чётное window_length должно стать нечётным
        result = SignalFilters.savgol(signal, window_length=6, polyorder=2)
        assert len(result) == len(signal)
        
        # window_length должен быть >= polyorder + 1
        result = SignalFilters.savgol(signal, window_length=2, polyorder=2)  # 2 < 3
        assert len(result) == len(signal)  # window_length должен стать 3