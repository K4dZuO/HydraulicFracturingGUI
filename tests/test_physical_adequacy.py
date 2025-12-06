import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.well_data import WellData, WellTimeSeries, FlowRegimeAnalysis, TypeCurveMatch
from helpers.grp_analysis import (
    compute_pressure_derivative, compute_flow_rate_derivative,
    analyze_flow_regime, generate_type_curves, match_type_curves,
    compute_productivity_index, detect_flow_regime_transitions
)
from helpers.ml_methods import (
    apply_ml_interpolation, apply_ml_filter, detect_outliers, clean_data
)


class TestPhysicalAdequacy:
    """Тесты на физическую адекватность результатов"""
    
    def test_pressure_values_adequacy(self):
        """Тест адекватности значений давления"""
        # Давление должно быть положительным и в разумных пределах
        time = pd.Series([0, 1, 2, 3, 4, 5])
        pressure = pd.Series([300, 295, 290, 285, 280, 275])
        flow_rate = pd.Series([100, 98, 96, 94, 92, 90])
        
        ts = WellTimeSeries(
            time=time, pressure=pressure, flow_rate=flow_rate,
            skin=0.05, thickness=10.0, fractures_count=10,
            fracture_width=1125.0, fracture_length=280.0, a_l_ratio=0.446429
        )
        
        # Проверяем базовые физические ограничения
        assert ts.pressure.min() > 0, "Давление должно быть положительным"
        assert ts.pressure.max() < 10000, "Давление не должно превышать 10000 атм"
        assert ts.pressure.is_monotonic_decreasing, "Давление должно убывать со временем"
    
    def test_flow_rate_values_adequacy(self):
        """Тест адекватности значений дебита"""
        time = pd.Series([0, 1, 2, 3, 4, 5])
        pressure = pd.Series([300, 295, 290, 285, 280, 275])
        flow_rate = pd.Series([100, 98, 96, 94, 92, 90])
        
        ts = WellTimeSeries(
            time=time, pressure=pressure, flow_rate=flow_rate,
            skin=0.05, thickness=10.0, fractures_count=10,
            fracture_width=1125.0, fracture_length=280.0, a_l_ratio=0.446429
        )
        
        # Проверяем базовые физические ограничения
        assert ts.flow_rate.min() > 0, "Дебит должен быть положительным"
        assert ts.flow_rate.max() < 10000, "Дебит не должен превышать 10000 м³/сут"
        assert ts.flow_rate.is_monotonic_decreasing, "Дебит должен убывать со временем"
    
    def test_well_parameters_adequacy(self):
        """Тест адекватности параметров скважины"""
        # Создаем тестовые данные с реалистичными параметрами
        time = pd.Series([0, 1, 2, 3, 4, 5])
        pressure = pd.Series([300, 295, 290, 285, 280, 275])
        flow_rate = pd.Series([100, 98, 96, 94, 92, 90])
        
        ts = WellTimeSeries(
            time=time, pressure=pressure, flow_rate=flow_rate,
            skin=0.05, thickness=10.0, fractures_count=10,
            fracture_width=1125.0, fracture_length=280.0, a_l_ratio=0.446429
        )
        
        # Проверяем параметры скважины
        assert -10 <= ts.skin <= 50, "Skin должен быть в диапазоне [-10, 50]"
        assert ts.thickness > 0, "Толщина пласта должна быть положительной"
        assert ts.thickness < 1000, "Толщина пласта не должна превышать 1000 м"
        assert 1 <= ts.fractures_count <= 100, "Количество трещин должно быть в диапазоне [1, 100]"
        assert ts.fracture_width > 0, "Ширина трещины должна быть положительной"
        assert ts.fracture_length > 0, "Длина трещины должна быть положительной"
        assert 0 < ts.a_l_ratio <= 1, "Отношение a/L должно быть в диапазоне (0, 1]"
    
    def test_derivative_adequacy(self):
        """Тест адекватности производных"""
        time = pd.Series([0, 1, 2, 3, 4, 5])
        pressure = pd.Series([300, 295, 290, 285, 280, 275])
        flow_rate = pd.Series([100, 98, 96, 94, 92, 90])
        
        ts = WellTimeSeries(
            time=time, pressure=pressure, flow_rate=flow_rate,
            skin=0.05, thickness=10.0, fractures_count=10,
            fracture_width=1125.0, fracture_length=280.0, a_l_ratio=0.446429
        )
        
        # Проверяем производную давления
        dp_dt = compute_pressure_derivative(ts)
        assert dp_dt.max() <= 0, "Производная давления должна быть отрицательной (давление убывает)"
        assert not dp_dt.isna().all(), "Производная не должна состоять только из NaN"
        
        # Проверяем производную дебита
        dq_dt = compute_flow_rate_derivative(ts)
        assert dq_dt.max() <= 0, "Производная дебита должна быть отрицательной (дебит убывает)"
        assert not dq_dt.isna().all(), "Производная не должна состоять только из NaN"
    
    def test_productivity_index_adequacy(self):
        """Тест адекватности индекса продуктивности"""
        time = pd.Series([0, 1, 2, 3, 4, 5])
        pressure = pd.Series([300, 295, 290, 285, 280, 275])
        flow_rate = pd.Series([100, 98, 96, 94, 92, 90])
        
        ts = WellTimeSeries(
            time=time, pressure=pressure, flow_rate=flow_rate,
            skin=0.05, thickness=10.0, fractures_count=10,
            fracture_width=1125.0, fracture_length=280.0, a_l_ratio=0.446429
        )
        
        productivity = compute_productivity_index(ts)
        
        # Проверяем физическую адекватность результатов
        assert productivity['productivity_index'] > 0, "Индекс продуктивности должен быть положительным"
        assert productivity['fracture_efficiency'] > 0, "Эффективность ГРП должна быть положительной"
        assert productivity['average_flow_rate'] > 0, "Средний дебит должен быть положительным"
        assert productivity['average_pressure'] > 0, "Среднее давление должно быть положительным"
        assert productivity['total_fracture_length'] > 0, "Общая длина трещин должна быть положительной"
        
        # Проверяем разумные пределы
        assert productivity['productivity_index'] < 100, "Индекс продуктивности не должен быть слишком большим"
        assert productivity['average_flow_rate'] < 10000, "Средний дебит не должен быть слишком большим"
        assert productivity['average_pressure'] < 10000, "Среднее давление не должно быть слишком большим"
    
    def test_flow_regime_analysis_adequacy(self):
        """Тест адекватности анализа режима течения"""
        time = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pressure = pd.Series([300, 290, 280, 270, 260, 250, 240, 230, 220, 210, 200])
        flow_rate = pd.Series([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50])
        
        ts = WellTimeSeries(
            time=time, pressure=pressure, flow_rate=flow_rate,
            skin=0.05, thickness=10.0, fractures_count=10,
            fracture_width=1125.0, fracture_length=280.0, a_l_ratio=0.446429
        )
        
        analysis = analyze_flow_regime(ts)
        
        # Проверяем корректность результатов
        assert isinstance(analysis.regime_type, str), "Тип режима должен быть строкой"
        assert analysis.regime_type in ["Билинейное течение", "Линейное течение", "Псевдорадиальное течение", "Неопределенный"], \
            "Тип режима должен быть одним из известных"
        assert 0 <= analysis.confidence <= 1, "Уверенность должна быть в диапазоне [0, 1]"
        
        # Проверяем параметры
        assert 'slope' in analysis.parameters, "Параметры должны содержать наклон"
        assert isinstance(analysis.parameters['slope'], (int, float)), "Наклон должен быть числом"
    
    def test_type_curves_adequacy(self):
        """Тест адекватности эталонных кривых"""
        time_range = np.logspace(-1, 2, 50)
        curves = generate_type_curves(0.05, 10, 0.446429, time_range)
        
        for curve_type, (t, q) in curves.items():
            # Проверяем, что все значения положительны
            assert np.all(q > 0), f"Дебит в кривой {curve_type} должен быть положительным"
            assert np.all(t > 0), f"Время в кривой {curve_type} должно быть положительным"
            
            # Проверяем монотонность (дебит должен убывать со временем)
            assert np.all(np.diff(q) <= 0), f"Дебит в кривой {curve_type} должен убывать со временем"
            
            # Проверяем разумные пределы
            assert q.max() < 10000, f"Максимальный дебит в кривой {curve_type} не должен превышать 10000"
            assert q.min() > 0, f"Минимальный дебит в кривой {curve_type} должен быть положительным"
    
    def test_ml_interpolation_adequacy(self):
        """Тест адекватности ML-интерполяции"""
        time = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        values = pd.Series([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50])
        
        # Добавляем пропуски
        values_with_gaps = values.copy()
        values_with_gaps.iloc[3] = np.nan
        values_with_gaps.iloc[7] = np.nan
        
        interpolated = apply_ml_interpolation(time, values_with_gaps, 'random_forest')
        
        # Проверяем адекватность интерполированных значений
        assert not interpolated.isna().any(), "Интерполированные значения не должны содержать NaN"
        assert interpolated.min() > 0, "Интерполированные значения должны быть положительными"
        assert interpolated.max() < 1000, "Интерполированные значения не должны быть слишком большими"
        
        # Проверяем, что интерполированные значения находятся в разумных пределах
        original_values = values[~values.isna()]
        interpolated_values = interpolated[~values.isna()]
        
        # Интерполированные значения не должны сильно отличаться от исходных
        diff = np.abs(interpolated_values - original_values)
        assert diff.max() < 50, "Интерполированные значения не должны сильно отличаться от исходных"
    
    def test_ml_filtering_adequacy(self):
        """Тест адекватности ML-фильтрации"""
        time = pd.Series(range(20))
        clean_values = pd.Series([100 - i for i in range(20)])
        noisy_values = clean_values + np.random.normal(0, 2, 20)
        
        filtered = apply_ml_filter(noisy_values, 'savitzky_golay', window_length=5, polyorder=2)
        
        # Проверяем адекватность отфильтрованных значений
        assert not filtered.isna().any(), "Отфильтрованные значения не должны содержать NaN"
        assert filtered.min() > 0, "Отфильтрованные значения должны быть положительными"
        assert filtered.max() < 1000, "Отфильтрованные значения не должны быть слишком большими"
        
        # Проверяем, что фильтрация уменьшила шум
        noise_reduction = np.std(noisy_values - filtered) < np.std(noisy_values - clean_values)
        assert noise_reduction, "Фильтрация должна уменьшать шум"
    
    def test_outlier_detection_adequacy(self):
        """Тест адекватности обнаружения выбросов"""
        # Создаем данные с известными выбросами
        values = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])  # 100 - выброс
        
        outliers = detect_outliers(values, 'iqr', threshold=1.5)
        
        # Проверяем корректность обнаружения
        assert isinstance(outliers, pd.Series), "Результат должен быть pandas Series"
        assert outliers.dtype == bool, "Результат должен быть булевым"
        assert len(outliers) == len(values), "Длина результата должна совпадать с длиной входных данных"
        
        # Проверяем, что выброс обнаружен
        assert outliers.iloc[5] == True, "Значение 100 должно быть обнаружено как выброс"
        
        # Проверяем, что нормальные значения не помечены как выбросы
        normal_values = outliers.iloc[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]]
        assert normal_values.sum() == 0, "Нормальные значения не должны быть помечены как выбросы"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
