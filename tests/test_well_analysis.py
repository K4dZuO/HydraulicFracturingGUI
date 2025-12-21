import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.well_data import WellData, WellTimeSeries, FlowRegimeAnalysis, TypeCurveMatch
from helpers.parse_well_data import parse_well_data, validate_well_data
from helpers.grp_analysis import (
    compute_pressure_derivative, compute_flow_rate_derivative,
    analyze_flow_regime, generate_type_curves, match_type_curves,
    compute_productivity_index, detect_flow_regime_transitions
)
from helpers.ml_methods import (
    MLInterpolator, MLFilter,
    apply_ml_interpolation, apply_ml_filter, detect_outliers, clean_data
)


class TestWellDataSchemas:
    """Тесты для схем данных"""
    
    def test_well_data_creation(self):
        """Тест создания объекта WellData"""
        data = WellData(
            skin=0.05, h=10.0, n=10, w=1125.0, l=280.0, a_l=0.446429,
            elem_idx=1, x=0.0, y=0.0, t=0.0, p=300.0, dp=0.0, q=100.0
        )
        assert data.skin == 0.05
        assert data.h == 10.0
        assert data.n == 10
    
    def test_well_data_validation(self):
        """Тест валидации данных WellData"""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            WellData(
                skin=-15, h=10.0, n=10, w=1125.0, l=280.0, a_l=0.446429,
                elem_idx=1, x=0.0, y=0.0, t=0.0, p=300.0, dp=0.0, q=100.0
            )
    
    def test_well_time_series_creation(self):
        """Тест создания WellTimeSeries"""
        time = pd.Series([0, 1, 2, 3, 4])
        pressure = pd.Series([300, 295, 290, 285, 280])
        flow_rate = pd.Series([100, 98, 96, 94, 92])
        
        ts = WellTimeSeries(
            time=time, pressure=pressure, flow_rate=flow_rate,
            skin=0.05, thickness=10.0, fractures_count=10,
            fracture_width=1125.0, fracture_length=280.0, a_l_ratio=0.446429
        )
        
        assert len(ts.time) == 5
        assert ts.skin == 0.05


class TestParseWellData:
    """Тесты для парсинга данных скважины"""
    
    def test_parse_well_data_success(self):
        """Тест успешного парсинга CSV"""
        # Создаем тестовый CSV файл
        test_data = {
            'Skin': [0.05] * 5,
            'h': [10.0] * 5,
            'N': [10] * 5,
            'W': [1125.0] * 5,
            'L': [280.0] * 5,
            'a/L': [0.446429] * 5,
            'ElemIdx': [1, 2, 3, 4, 5],
            'X': [0.0] * 5,
            'Y': [0.0] * 5,
            't': [0, 1, 2, 3, 4],
            'P': [300, 295, 290, 285, 280],
            'dP': [0, -5, -5, -5, -5],
            'Q': [100, 98, 96, 94, 92]
        }
        
        df = pd.DataFrame(test_data)
        df.to_csv('test_well_data.csv', index=False)
        
        try:
            well_data_list, error = parse_well_data('test_well_data.csv')
            assert error is None
            assert well_data_list is not None
            assert len(well_data_list) > 0
            well_data = well_data_list[0]
            assert len(well_data.time) == 5
            assert well_data.skin == 0.05
        finally:
            # Удаляем тестовый файл
            if os.path.exists('test_well_data.csv'):
                os.remove('test_well_data.csv')
    
    def test_parse_well_data_missing_columns(self):
        """Тест парсинга с отсутствующими колонками"""
        test_data = {'Skin': [0.05], 'h': [10.0]}  # Неполные данные
        df = pd.DataFrame(test_data)
        df.to_csv('test_incomplete.csv', index=False)
        
        try:
            well_data_list, error = parse_well_data('test_incomplete.csv')
            assert well_data_list is None
            assert error is not None
            assert "Отсутствуют обязательные колонки" in error
        finally:
            if os.path.exists('test_incomplete.csv'):
                os.remove('test_incomplete.csv')
    
    def test_validate_well_data(self):
        """Тест валидации данных скважины"""
        test_data = {
            'Skin': [0.05] * 3,
            'h': [10.0] * 3,
            'N': [10] * 3,
            'W': [1125.0] * 3,
            'L': [280.0] * 3,
            'a/L': [0.446429] * 3,
            'ElemIdx': [1, 2, 3],
            'X': [0.0] * 3,
            'Y': [0.0] * 3,
            't': [0, 1, 2],
            'P': [300, 295, 290],
            'dP': [0, -5, -5],
            'Q': [100, 98, 96]
        }
        
        df = pd.DataFrame(test_data)
        is_valid, error = validate_well_data(df)
        assert is_valid
        assert error is None


class TestGRPAnalysis:
    """Тесты для анализа ГРП"""
    
    def test_compute_pressure_derivative(self):
        """Тест вычисления производной давления"""
        time = pd.Series([0, 1, 2, 3, 4])
        pressure = pd.Series([300, 295, 290, 285, 280])
        flow_rate = pd.Series([100, 98, 96, 94, 92])
        
        ts = WellTimeSeries(
            time=time, pressure=pressure, flow_rate=flow_rate,
            skin=0.05, thickness=10.0, fractures_count=10,
            fracture_width=1125.0, fracture_length=280.0, a_l_ratio=0.446429
        )
        
        dp_dt = compute_pressure_derivative(ts)
        assert len(dp_dt) == len(time)
        assert not dp_dt.isna().all()
    
    def test_analyze_flow_regime(self):
        """Тест анализа режима течения"""
        time = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pressure = pd.Series([300, 290, 280, 270, 260, 250, 240, 230, 220, 210, 200])
        flow_rate = pd.Series([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50])
        
        ts = WellTimeSeries(
            time=time, pressure=pressure, flow_rate=flow_rate,
            skin=0.05, thickness=10.0, fractures_count=10,
            fracture_width=1125.0, fracture_length=280.0, a_l_ratio=0.446429
        )
        
        analysis = analyze_flow_regime(ts)
        assert isinstance(analysis, FlowRegimeAnalysis)
        assert analysis.regime_type in ["Билинейное течение", "Линейное течение", "Псевдорадиальное течение"]
        assert 0 <= analysis.confidence <= 1
    
    def test_generate_type_curves(self):
        """Тест генерации эталонных кривых"""
        time_range = np.logspace(-1, 2, 50)
        curves = generate_type_curves(0.05, 10, 0.446429, time_range)
        
        assert 'bilinear' in curves
        assert 'linear' in curves
        assert 'pseudoradial' in curves
        
        for curve_type, (t, q) in curves.items():
            assert len(t) == len(q)
            assert len(t) == 50
    
    def test_compute_well_productivity_index(self):
        """Тест вычисления индекса продуктивности"""
        time = pd.Series([0, 1, 2, 3, 4])
        pressure = pd.Series([300, 295, 290, 285, 280])
        flow_rate = pd.Series([100, 98, 96, 94, 92])
        
        ts = WellTimeSeries(
            time=time, pressure=pressure, flow_rate=flow_rate,
            skin=0.05, thickness=10.0, fractures_count=10,
            fracture_width=1125.0, fracture_length=280.0, a_l_ratio=0.446429
        )
        
        productivity = compute_productivity_index(ts)
        assert 'productivity_index' in productivity
        assert 'fracture_efficiency' in productivity
        assert productivity['productivity_index'] > 0


class TestMLMethods:
    """Тесты для ML-методов"""
    
    def test_ml_interpolator_random_forest(self):
        """Тест ML-интерполятора с Random Forest"""
        time = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        values = pd.Series([100, 95, 90, 85, 80, 75, 70, 65, 60, 55])
        
        interpolator = MLInterpolator('random_forest')
        interpolator.fit(time, values)
        
        new_time = pd.Series([0.5, 1.5, 2.5])
        predictions = interpolator.predict(new_time)
        
        assert len(predictions) == 3
        assert not predictions.isna().any()
    
    def test_ml_filter_savitzky_golay(self):
        """Тест фильтра Савицкого-Голея"""
        # Создаем данные с шумом
        time = pd.Series(range(20))
        clean_values = pd.Series([100 - i for i in range(20)])
        noisy_values = clean_values + np.random.normal(0, 2, 20)
        
        filter_obj = MLFilter('savitzky_golay')
        filtered = filter_obj.filter(noisy_values, window_length=5, polyorder=2)
        
        assert len(filtered) == len(noisy_values)
        assert not filtered.isna().any()
    
    def test_detect_outliers_iqr(self):
        """Тест обнаружения выбросов методом IQR"""
        values = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])  # 100 - выброс
        outliers = detect_outliers(values, 'iqr', threshold=1.5)
        
        assert outliers.sum() > 0
        assert outliers.iloc[5] == True  # 100 должен быть выбросом
    
    def test_clean_data(self):
        """Тест комплексной очистки данных"""
        time = pd.Series(range(20))
        values = pd.Series([100 - i + np.random.normal(0, 1, 20) for i in range(20)])
        
        clean_time, clean_values = clean_data(time, values)
        
        assert len(clean_time) <= len(time)
        assert len(clean_values) == len(clean_time)
        assert not clean_values.isna().any()
        
        # Проверяем физическую адекватность
        min_val = clean_values.min()
        max_val = clean_values.max()
        # Преобразуем в скаляр, если это numpy scalar или pandas scalar
        if hasattr(min_val, 'item'):
            min_val = min_val.item()
        if hasattr(max_val, 'item'):
            max_val = max_val.item()
        assert float(min_val) > 0, "Значения должны быть положительными"
        assert float(max_val) < 1000, "Значения не должны быть слишком большими"


class TestIntegration:
    """Интеграционные тесты"""
    
    def test_full_analysis_pipeline(self):
        """Тест полного пайплайна анализа"""
        # Создаем тестовые данные
        time = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pressure = pd.Series([300, 290, 280, 270, 260, 250, 240, 230, 220, 210, 200])
        flow_rate = pd.Series([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50])
        
        ts = WellTimeSeries(
            time=time, pressure=pressure, flow_rate=flow_rate,
            skin=0.05, thickness=10.0, fractures_count=10,
            fracture_width=1125.0, fracture_length=280.0, a_l_ratio=0.446429
        )
        
        # 1. Анализ режима течения
        flow_regime = analyze_flow_regime(ts)
        assert flow_regime.regime_type is not None
        assert isinstance(flow_regime.regime_type, str)
        
        # 2. Вычисление индекса продуктивности
        productivity = compute_productivity_index(ts)
        assert productivity['productivity_index'] > 0
        assert isinstance(productivity['productivity_index'], (int, float))
        
        # 3. Обнаружение переходов
        transitions = detect_flow_regime_transitions(ts)
        assert isinstance(transitions, list)
        
        # 4. Сопоставление с эталонными кривыми
        match_result = match_type_curves(ts)
        assert isinstance(match_result, TypeCurveMatch)
        assert isinstance(match_result.curve_type, str)
        
        # Проверяем физическую адекватность результатов
        assert 0 <= flow_regime.confidence <= 1, "Уверенность должна быть в диапазоне [0, 1]"
        avg_flow = productivity['average_flow_rate']
        avg_pressure = productivity['average_pressure']
        # Преобразуем в скаляр, если это numpy scalar или pandas scalar
        if hasattr(avg_flow, 'item'):
            avg_flow = avg_flow.item()
        if hasattr(avg_pressure, 'item'):
            avg_pressure = avg_pressure.item()
        assert float(avg_flow) > 0, "Средний дебит должен быть положительным"
        assert float(avg_pressure) > 0, "Среднее давление должно быть положительным"
    
    def test_parse_well_data_multiple_groups(self):
        """Тест парсинга файла с несколькими группами данных"""
        # Создаем тестовые данные с двумя группами
        data1 = {
            'Skin': [0.05] * 10,
            'h': [10.0] * 10,
            'N': [5] * 10,
            'W': [1000.0] * 10,
            'L': [200.0] * 10,
            'a/L': [0.5] * 10,
            'ElemIdx': list(range(1, 11)),
            'X': [0.0] * 10,
            'Y': [0.0] * 10,
            't': list(range(10)),
            'P': [300 - i * 2 for i in range(10)],
            'dP': [0] + [-2] * 9,
            'Q': [100 - i * 1 for i in range(10)]
        }
        
        data2 = {
            'Skin': [0.1] * 8,
            'h': [15.0] * 8,
            'N': [8] * 8,
            'W': [1200.0] * 8,
            'L': [250.0] * 8,
            'a/L': [0.6] * 8,
            'ElemIdx': list(range(11, 19)),
            'X': [0.0] * 8,
            'Y': [0.0] * 8,
            't': list(range(8)),
            'P': [280 - i * 3 for i in range(8)],
            'dP': [0] + [-3] * 7,
            'Q': [90 - i * 2 for i in range(8)]
        }
        
        # Объединяем данные
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        df_combined = pd.concat([df1, df2], ignore_index=True)
        
        # Сохраняем во временный файл
        temp_file = "temp_test_multiple.csv"
        df_combined.to_csv(temp_file, index=False)
        
        try:
            # Парсим данные
            well_data_list, error_msg = parse_well_data(temp_file)
            
            assert error_msg is None, f"Ошибка парсинга: {error_msg}"
            assert well_data_list is not None, "Список данных не должен быть None"
            assert len(well_data_list) == 2, f"Ожидалось 2 группы, получено {len(well_data_list)}"
            
            # Проверяем первую группу
            well1 = well_data_list[0]
            assert well1.skin == 0.05
            assert well1.fractures_count == 5
            assert well1.a_l_ratio == 0.5
            assert len(well1.time) == 10
            
            # Проверяем вторую группу
            well2 = well_data_list[1]
            assert well2.skin == 0.1
            assert well2.fractures_count == 8
            assert well2.a_l_ratio == 0.6
            assert len(well2.time) == 8
            
        finally:
            # Удаляем временный файл
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_ml_pipeline(self):
        """Тест ML-пайплайна"""
        # Создаем данные с пропусками и шумом
        time = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        clean_values = pd.Series([100 - i for i in range(11)])
        noisy_values = clean_values + np.random.normal(0, 1, 11)
        
        # Добавляем пропуски
        noisy_values.iloc[3] = np.nan
        noisy_values.iloc[7] = np.nan
        
        # 1. ML-интерполяция
        interpolated = apply_ml_interpolation(time, noisy_values, 'random_forest')
        assert not interpolated.isna().any()
        
        # 2. ML-фильтрация
        filtered = apply_ml_filter(interpolated, 'savitzky_golay')
        assert not filtered.isna().any()
        
        # 3. Обнаружение выбросов
        outliers = detect_outliers(filtered, 'iqr')
        assert isinstance(outliers, pd.Series)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
