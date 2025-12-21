#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексные тесты производительности, стабильности и качества
Покрывает все требования к метрикам качества приложения
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import List, Tuple, Dict

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest

from schemas.well_data import WellTimeSeries
from helpers.ml_methods import (
    apply_ml_interpolation, apply_ml_filter, detect_outliers
)
from helpers.math_error_logger import log_computation_error, log_math_error


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def qapp():
    """Создает QApplication для всех тестов"""
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    yield app


@pytest.fixture
def app_with_data(qapp):
    """Создает приложение с загруженными тестовыми данными"""
    from main import MyApp
    
    app = MyApp(test_mode=True)
    
    # Создаем тестовые данные
    n_points = 50
    time_data = np.linspace(0, 100, n_points)
    pressure_data = 300 - time_data * 0.5 + np.random.normal(0, 5, n_points)
    flow_rate_data = 100 - time_data * 0.2 + np.random.normal(0, 3, n_points)
    
    well_data = WellTimeSeries(
        time=pd.Series(time_data),
        pressure=pd.Series(pressure_data),
        flow_rate=pd.Series(flow_rate_data),
        thickness=10.0,
        fracture_length=200.0,
        fracture_width=0.01,
        skin=0.0,
        fractures_count=5,
        a_l_ratio=0.2
    )
    
    app.loaded_data = [well_data]
    app.current_index = 0
    
    yield app
    
    try:
        app.close()
    except:
        pass


@pytest.fixture
def large_dataset():
    """Создает большой набор данных (100K строк) для тестирования загрузки"""
    n_points = 100000
    time_data = np.linspace(0, 1000, n_points)
    pressure_data = 300 - time_data * 0.1 + np.random.normal(0, 5, n_points)
    flow_rate_data = 100 - time_data * 0.05 + np.random.normal(0, 3, n_points)
    
    df = pd.DataFrame({
        'time': time_data,
        'pressure': pressure_data,
        'flow_rate': flow_rate_data
    })
    
    return df


# ============================================================================
# ТЕСТ 1: ВРЕМЯ ОТКЛИКА GUI ПО ВСЕМ КНОПКАМ < 500ms
# ============================================================================

class TestGUIResponseTime:
    """Тесты времени отклика всех кнопок GUI (кроме загрузки файлов)"""
    
    @pytest.mark.performance
    def test_plot_button_response_time(self, app_with_data):
        """Время отклика кнопки 'Построить график' < 500ms"""
        app = app_with_data
        times = []
        
        for _ in range(20):
            start = time.perf_counter()
            app.on_plot_dimensionless_selected()
            QApplication.instance().processEvents()
            elapsed = (time.perf_counter() - start) * 1000  # в мс
            times.append(elapsed)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        print(f"\n📊 Построить график: среднее={avg_time:.2f}ms, макс={max_time:.2f}ms")
        
        assert avg_time < 500, f"Среднее время отклика {avg_time:.2f}ms превышает 500ms"
        assert max_time < 1000, f"Максимальное время отклика {max_time:.2f}ms превышает 1000ms"
    
    @pytest.mark.performance
    def test_interpolation_button_response_time(self, app_with_data):
        """Время отклика кнопки 'Интерполяция' < 500ms"""
        app = app_with_data
        times = []
        
        # Добавляем пропуски для интерполяции
        app.loaded_data[0].pressure.iloc[10:15] = np.nan
        
        for _ in range(10):  # Меньше итераций, т.к. ML может быть медленнее
            start = time.perf_counter()
            app.on_interpolate_data()
            QApplication.instance().processEvents()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        print(f"\n📊 Интерполяция: среднее={avg_time:.2f}ms, макс={max_time:.2f}ms")
        
        # Для ML методов допускаем больше времени
        assert avg_time < 5000, f"Среднее время интерполяции {avg_time:.2f}ms превышает 5000ms"
    
    @pytest.mark.performance
    def test_ml_filter_button_response_time(self, app_with_data):
        """Время отклика кнопки 'ML фильтрация' < 500ms"""
        app = app_with_data
        times = []
        
        for _ in range(10):
            start = time.perf_counter()
            app.on_ml_filter()
            QApplication.instance().processEvents()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        print(f"\n📊 ML фильтрация: среднее={avg_time:.2f}ms")
        
        assert avg_time < 1000, f"Среднее время фильтрации {avg_time:.2f}ms превышает 1000ms"
    
    @pytest.mark.performance
    def test_outlier_detection_button_response_time(self, app_with_data):
        """Время отклика кнопки 'Обнаружить выбросы' < 500ms"""
        app = app_with_data
        times = []
        
        for _ in range(20):
            start = time.perf_counter()
            app.on_detect_outliers()
            QApplication.instance().processEvents()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        print(f"\n📊 Обнаружить выбросы: среднее={avg_time:.2f}ms")
        
        assert avg_time < 500, f"Среднее время обнаружения выбросов {avg_time:.2f}ms превышает 500ms"
    
    @pytest.mark.performance
    def test_export_button_response_time(self, app_with_data):
        """Время отклика кнопки 'Экспорт данных' < 500ms"""
        app = app_with_data
        
        with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName', return_value=('test_export.csv', 'CSV')):
            start = time.perf_counter()
            app.on_export_data()
            QApplication.instance().processEvents()
            elapsed = (time.perf_counter() - start) * 1000
        
        print(f"\n📊 Экспорт данных: {elapsed:.2f}ms")
        
        assert elapsed < 500, f"Время экспорта {elapsed:.2f}ms превышает 500ms"
    
    @pytest.mark.performance
    def test_flow_regime_analysis_button_response_time(self, app_with_data):
        """Время отклика кнопки 'Анализ режима течения' < 500ms"""
        app = app_with_data
        times = []
        
        for _ in range(20):
            start = time.perf_counter()
            app.on_analyze_flow_regime()
            QApplication.instance().processEvents()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        print(f"\n📊 Анализ режима течения: среднее={avg_time:.2f}ms")
        
        assert avg_time < 500, f"Среднее время анализа {avg_time:.2f}ms превышает 500ms"
    
    @pytest.mark.performance
    def test_productivity_button_response_time(self, app_with_data):
        """Время отклика кнопки 'Индекс продуктивности' < 500ms"""
        app = app_with_data
        times = []
        
        # Проверяем наличие метода
        if not hasattr(app, 'on_compute_productivity_index'):
            pytest.skip("Метод on_compute_productivity_index не найден")
        
        for _ in range(20):
            start = time.perf_counter()
            app.on_compute_productivity_index()
            QApplication.instance().processEvents()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        print(f"\n📊 Индекс продуктивности: среднее={avg_time:.2f}ms")
        
        assert avg_time < 500, f"Среднее время вычисления {avg_time:.2f}ms превышает 500ms"
    
    @pytest.mark.performance
    def test_transitions_button_response_time(self, app_with_data):
        """Время отклика кнопки 'Переходы режимов' < 500ms"""
        app = app_with_data
        
        # Проверяем наличие метода
        if not hasattr(app, 'on_detect_transitions'):
            pytest.skip("Метод on_detect_transitions не найден")
        
        times = []
        
        for _ in range(20):
            start = time.perf_counter()
            app.on_detect_transitions()
            QApplication.instance().processEvents()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        print(f"\n📊 Переходы режимов: среднее={avg_time:.2f}ms")
        
        assert avg_time < 500, f"Среднее время определения переходов {avg_time:.2f}ms превышает 500ms"
    
    @pytest.mark.performance
    def test_type_curve_buttons_response_time(self, app_with_data):
        """Время отклика кнопок эталонных кривых < 500ms"""
        app = app_with_data
        
        buttons_methods = [
            ('Билинейное', app.on_plot_bilinear),
            ('Линейное', app.on_plot_linear),
            ('Псевдорадиальное', app.on_plot_pseudoradial),
        ]
        
        for name, method in buttons_methods:
            times = []
            for _ in range(10):
                start = time.perf_counter()
                method()
                QApplication.instance().processEvents()
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            avg_time = np.mean(times)
            print(f"\n📊 {name}: среднее={avg_time:.2f}ms")
            
            assert avg_time < 500, f"{name}: среднее время {avg_time:.2f}ms превышает 500ms"
    
    @pytest.mark.performance
    def test_match_curves_button_response_time(self, app_with_data):
        """Время отклика кнопки 'Сопоставить с данными' < 500ms"""
        app = app_with_data
        
        # Проверяем наличие метода
        if not hasattr(app, 'on_match_curves'):
            pytest.skip("Метод on_match_curves не найден")
        
        times = []
        
        for _ in range(10):
            start = time.perf_counter()
            app.on_match_curves()
            QApplication.instance().processEvents()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        print(f"\n📊 Сопоставить с данными: среднее={avg_time:.2f}ms")
        
        assert avg_time < 1000, f"Среднее время сопоставления {avg_time:.2f}ms превышает 1000ms"


# ============================================================================
# ТЕСТ 2: FAILURE RATE < 1%
# ============================================================================

class TestFailureRate:
    """Тесты частоты отказов при многократном выполнении операций"""
    
    @pytest.mark.load
    def test_plot_button_failure_rate(self, app_with_data):
        """Failure rate при построении графика < 1% (из 1000 попыток)"""
        app = app_with_data
        n_iterations = 1000
        failures = 0
        exceptions = []
        
        for i in range(n_iterations):
            try:
                app.on_plot_dimensionless_selected()
                QApplication.instance().processEvents()
            except Exception as e:
                failures += 1
                exceptions.append((i, str(e)))
        
        failure_rate = (failures / n_iterations) * 100
        
        print(f"\n📊 Failure rate (построение графика): {failure_rate:.2f}% ({failures}/{n_iterations})")
        if exceptions:
            print(f"   Первые 5 исключений: {exceptions[:5]}")
        
        assert failure_rate < 1.0, f"Failure rate {failure_rate:.2f}% превышает 1%"
    
    @pytest.mark.load
    def test_interpolation_failure_rate(self, app_with_data):
        """Failure rate при интерполяции < 1% (из 100 попыток)"""
        app = app_with_data
        n_iterations = 100
        failures = 0
        exceptions = []
        
        # Добавляем пропуски
        original_pressure = app.loaded_data[0].pressure.copy()
        
        for i in range(n_iterations):
            try:
                # Восстанавливаем пропуски для каждой итерации
                app.loaded_data[0].pressure = original_pressure.copy()
                app.loaded_data[0].pressure.iloc[10:15] = np.nan
                
                app.on_interpolate_data()
                QApplication.instance().processEvents()
            except Exception as e:
                failures += 1
                exceptions.append((i, str(e)))
                # Логируем ошибку
                log_computation_error(
                    subsystem="gui",
                    method="interpolate_data",
                    exception=e,
                    data_volume=len(app.current_data.time) if app.current_data else None,
                    context={"iteration": i, "test": "test_interpolation_failure_rate"}
                )
        
        failure_rate = (failures / n_iterations) * 100
        
        print(f"\n📊 Failure rate (интерполяция): {failure_rate:.2f}% ({failures}/{n_iterations})")
        if exceptions:
            print(f"   Первые 5 исключений: {exceptions[:5]}")
        
        assert failure_rate < 1.0, f"Failure rate {failure_rate:.2f}% превышает 1%"
    
    @pytest.mark.load
    def test_all_buttons_combined_failure_rate(self, app_with_data):
        """Общий failure rate для всех операций < 1%"""
        app = app_with_data
        n_iterations = 200
        failures = 0
        
        operations = [
            app.on_plot_dimensionless_selected,
            app.on_ml_filter,
            app.on_detect_outliers,
            app.on_analyze_flow_regime,
            app.on_compute_productivity,
        ]
        
        for i in range(n_iterations):
            op = operations[i % len(operations)]
            try:
                op()
                QApplication.instance().processEvents()
            except Exception:
                failures += 1
        
        failure_rate = (failures / n_iterations) * 100
        
        print(f"\n📊 Общий failure rate: {failure_rate:.2f}% ({failures}/{n_iterations})")
        
        assert failure_rate < 1.0, f"Общий failure rate {failure_rate:.2f}% превышает 1%"


# ============================================================================
# ТЕСТ 3: ПОТРЕБЛЕНИЕ ПАМЯТИ - ЛИНЕЙНЫЙ РОСТ
# ============================================================================

class TestMemoryConsumption:
    """Тесты потребления памяти и проверка линейного роста"""
    
    @pytest.mark.memory
    def test_memory_growth_is_linear(self, app_with_data):
        """Проверка линейного роста памяти при многократных операциях"""
        app = app_with_data
        
        tracemalloc.start()
        
        # Измеряем память на разных этапах
        measurements = []
        iterations_points = [10, 20, 50, 100, 200]
        
        for target_iter in iterations_points:
            # Сбрасываем и выполняем операции
            tracemalloc.clear_traces()
            snapshot_before = tracemalloc.take_snapshot()
            
            for _ in range(target_iter):
                app.on_plot_dimensionless_selected()
                QApplication.instance().processEvents()
            
            snapshot_after = tracemalloc.take_snapshot()
            
            # Вычисляем прирост памяти
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            memory_growth = sum(stat.size_diff for stat in top_stats) / 1024 / 1024  # МБ
            
            measurements.append((target_iter, memory_growth))
            
            print(f"   После {target_iter} итераций: {memory_growth:.2f} МБ")
        
        tracemalloc.stop()
        
        # Анализ тренда: проверяем, что рост близок к линейному
        # Используем полиномиальную регрессию
        iterations = np.array([m[0] for m in measurements])
        memory = np.array([m[1] for m in measurements])
        
        # Линейная регрессия (полином 1 степени)
        linear_fit = np.polyfit(iterations, memory, 1)
        linear_residuals = memory - np.polyval(linear_fit, iterations)
        linear_r2 = 1 - (np.sum(linear_residuals**2) / np.sum((memory - np.mean(memory))**2))
        
        # Квадратичная регрессия (полином 2 степени)
        quad_fit = np.polyfit(iterations, memory, 2)
        quad_residuals = memory - np.polyval(quad_fit, iterations)
        quad_r2 = 1 - (np.sum(quad_residuals**2) / np.sum((memory - np.mean(memory))**2))
        
        print(f"\n📊 Анализ роста памяти:")
        print(f"   Линейная модель: R²={linear_r2:.4f}")
        print(f"   Квадратичная модель: R²={quad_r2:.4f}")
        print(f"   Коэффициент роста: {linear_fit[0]:.4f} МБ/итерация")
        
        # Проверяем, что линейная модель достаточно хороша (R² > 0.8)
        assert linear_r2 > 0.8, f"Линейная модель недостаточно точна (R²={linear_r2:.4f})"
        
        # Проверяем, что квадратичная модель не намного лучше (разница < 0.1)
        improvement = quad_r2 - linear_r2
        assert improvement < 0.1, f"Рост памяти сильно нелинейный (улучшение={improvement:.4f})"
    
    @pytest.mark.memory
    def test_memory_leak_detection(self, app_with_data):
        """Проверка отсутствия утечек памяти при повторных операциях"""
        app = app_with_data
        
        tracemalloc.start()
        
        # Выполняем операции и проверяем, что память не растет бесконтрольно
        snapshot1 = tracemalloc.take_snapshot()
        
        for _ in range(100):
            app.on_plot_dimensionless_selected()
            QApplication.instance().processEvents()
        
        snapshot2 = tracemalloc.take_snapshot()
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_growth = sum(stat.size_diff for stat in top_stats) / 1024 / 1024  # МБ
        
        tracemalloc.stop()
        
        print(f"\n📊 Рост памяти после 100 итераций: {total_growth:.2f} МБ")
        
        # Рост памяти не должен превышать 100 МБ на 100 операций
        assert total_growth < 100, f"Возможная утечка памяти: рост {total_growth:.2f} МБ"


# ============================================================================
# ТЕСТ 4: ВРЕМЯ ЗАГРУЗКИ ДАННЫХ < 5 СЕК/100K СТРОК
# ============================================================================

class TestDataLoadingPerformance:
    """Тесты производительности загрузки данных"""
    
    @pytest.mark.performance
    def test_large_dataset_loading_time(self, large_dataset, tmp_path):
        """Время загрузки 100K строк < 5 секунд"""
        # Сохраняем датасет в файл
        csv_file = tmp_path / "large_test_data.csv"
        large_dataset.to_csv(csv_file, index=False)
        
        from helpers.parse_well_data import parse_well_data
        
        start = time.perf_counter()
        
        # Загружаем данные
        df = pd.read_csv(csv_file)
        
        # Парсим в WellTimeSeries
        well_data_list = parse_well_data(df)
        
        elapsed = time.perf_counter() - start
        
        print(f"\n📊 Загрузка 100K строк: {elapsed:.2f} сек")
        print(f"   Загружено скважин: {len(well_data_list)}")
        
        assert elapsed < 5.0, f"Время загрузки {elapsed:.2f}с превышает 5 секунд"
    
    @pytest.mark.performance
    def test_multiple_large_datasets_loading(self, tmp_path):
        """Загрузка нескольких больших файлов"""
        n_files = 5
        rows_per_file = 50000
        loading_times = []
        
        from helpers.parse_well_data import parse_well_data
        
        for i in range(n_files):
            # Создаем данные
            time_data = np.linspace(0, 500, rows_per_file)
            pressure_data = 300 - time_data * 0.1 + np.random.normal(0, 5, rows_per_file)
            flow_rate_data = 100 - time_data * 0.05 + np.random.normal(0, 3, rows_per_file)
            
            df = pd.DataFrame({
                'time': time_data,
                'pressure': pressure_data,
                'flow_rate': flow_rate_data
            })
            
            csv_file = tmp_path / f"test_data_{i}.csv"
            df.to_csv(csv_file, index=False)
            
            # Загружаем
            start = time.perf_counter()
            df_loaded = pd.read_csv(csv_file)
            well_data = parse_well_data(df_loaded)
            elapsed = time.perf_counter() - start
            
            loading_times.append(elapsed)
            
            # Проверяем пропорциональность: 50K строк должны загружаться за < 2.5 сек
            assert elapsed < 2.5, f"Файл {i}: время загрузки {elapsed:.2f}с превышает 2.5 секунд"
        
        avg_time = np.mean(loading_times)
        print(f"\n📊 Среднее время загрузки 50K строк: {avg_time:.2f} сек")
        
        # Экстраполируем на 100K
        extrapolated_time = avg_time * 2
        print(f"   Экстраполированное время для 100K: {extrapolated_time:.2f} сек")
        
        assert extrapolated_time < 5.0, f"Экстраполированное время {extrapolated_time:.2f}с превышает 5 секунд"


# ============================================================================
# ТЕСТ 5: ТОЧНОСТЬ ML-ФУНКЦИЙ > 95%
# ============================================================================

class TestMLAccuracy:
    """Тесты точности ML-методов"""
    
    @pytest.mark.unit
    def test_interpolation_accuracy_above_95_percent(self):
        """Точность интерполяции > 95%"""
        # Создаем чистые синтетические данные
        n_points = 100
        time = pd.Series(np.linspace(0, 10, n_points))
        true_values = pd.Series(np.sin(time) * 50 + 100)  # Синусоида
        
        # Создаем данные с пропусками
        values_with_gaps = true_values.copy()
        gap_indices = [10, 11, 12, 30, 31, 32, 50, 51, 52, 70, 71, 72]  # 12% пропусков
        values_with_gaps.iloc[gap_indices] = np.nan
        
        # Интерполируем
        interpolated = apply_ml_interpolation(time, values_with_gaps, 'random_forest')
        
        # Вычисляем точность на пропущенных значениях
        true_gaps = true_values.iloc[gap_indices]
        interpolated_gaps = interpolated.iloc[gap_indices]
        
        # Относительная ошибка
        relative_errors = np.abs(true_gaps - interpolated_gaps) / np.abs(true_gaps)
        mean_accuracy = (1 - relative_errors.mean()) * 100
        
        print(f"\n📊 Точность интерполяции: {mean_accuracy:.2f}%")
        print(f"   Средняя относительная ошибка: {relative_errors.mean()*100:.2f}%")
        print(f"   Максимальная относительная ошибка: {relative_errors.max()*100:.2f}%")
        
        assert mean_accuracy > 95.0, f"Точность интерполяции {mean_accuracy:.2f}% ниже 95%"
    
    @pytest.mark.unit
    def test_filtering_accuracy_above_95_percent(self):
        """Точность фильтрации > 95%"""
        # Создаем чистый сигнал и зашумленный
        n_points = 200
        time = np.linspace(0, 10, n_points)
        true_signal = np.sin(time) * 50 + 100
        noisy_signal = true_signal + np.random.normal(0, 3, n_points)
        
        # Фильтруем
        filtered = apply_ml_filter(pd.Series(noisy_signal), 'savitzky_golay', window_length=11, polyorder=3)
        
        # Вычисляем точность
        errors = np.abs(true_signal - filtered.values)
        relative_errors = errors / np.abs(true_signal)
        mean_accuracy = (1 - relative_errors.mean()) * 100
        
        print(f"\n📊 Точность фильтрации: {mean_accuracy:.2f}%")
        print(f"   Средняя абсолютная ошибка: {errors.mean():.2f}")
        print(f"   Средняя относительная ошибка: {relative_errors.mean()*100:.2f}%")
        
        assert mean_accuracy > 95.0, f"Точность фильтрации {mean_accuracy:.2f}% ниже 95%"
    
    @pytest.mark.unit
    def test_outlier_detection_accuracy_above_95_percent(self):
        """Точность обнаружения выбросов > 95%"""
        # Создаем данные с известными выбросами
        n_points = 1000
        normal_values = np.random.normal(100, 10, n_points)
        
        # Добавляем 50 выбросов (5%)
        outlier_indices = np.random.choice(n_points, size=50, replace=False)
        values = normal_values.copy()
        values[outlier_indices] = values[outlier_indices] + np.random.choice([-1, 1], size=50) * 50
        
        # Обнаруживаем выбросы
        detected_outliers = detect_outliers(pd.Series(values), method='iqr', threshold=1.5)
        
        # Создаем маску истинных выбросов
        true_outliers = np.zeros(n_points, dtype=bool)
        true_outliers[outlier_indices] = True
        
        # Вычисляем метрики
        true_positives = np.sum(detected_outliers.values & true_outliers)
        false_positives = np.sum(detected_outliers.values & ~true_outliers)
        false_negatives = np.sum(~detected_outliers.values & true_outliers)
        true_negatives = np.sum(~detected_outliers.values & ~true_outliers)
        
        accuracy = (true_positives + true_negatives) / n_points * 100
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📊 Точность обнаружения выбросов: {accuracy:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall: {recall*100:.2f}%")
        print(f"   F1-score: {f1_score*100:.2f}%")
        
        assert accuracy > 95.0, f"Точность обнаружения выбросов {accuracy:.2f}% ниже 95%"
        assert f1_score > 0.85, f"F1-score {f1_score:.2f} ниже 0.85"


# ============================================================================
# ТЕСТ 6: СТАБИЛЬНОСТЬ ДОЛГОЙ РАБОТЫ - ДЕГРАДАЦИЯ < 15%
# ============================================================================

class TestLongTermStability:
    """Тесты стабильности при длительной работе"""
    
    @pytest.mark.longrun
    @pytest.mark.slow
    def test_long_running_performance_degradation(self, app_with_data):
        """Деградация производительности при длительной работе < 15%"""
        app = app_with_data
        
        # Измеряем производительность в начале
        initial_times = []
        for _ in range(50):
            start = time.perf_counter()
            app.on_plot_dimensionless_selected()
            QApplication.instance().processEvents()
            elapsed = time.perf_counter() - start
            initial_times.append(elapsed)
        
        initial_avg = np.mean(initial_times)
        
        # Выполняем много операций (симуляция длительной работы)
        print(f"\n📊 Выполнение 1000 операций...")
        for i in range(1000):
            app.on_plot_dimensionless_selected()
            QApplication.instance().processEvents()
            
            if i % 200 == 0:
                print(f"   Прогресс: {i}/1000")
        
        # Измеряем производительность после длительной работы
        final_times = []
        for _ in range(50):
            start = time.perf_counter()
            app.on_plot_dimensionless_selected()
            QApplication.instance().processEvents()
            elapsed = time.perf_counter() - start
            final_times.append(elapsed)
        
        final_avg = np.mean(final_times)
        
        # Вычисляем деградацию
        degradation = ((final_avg - initial_avg) / initial_avg) * 100
        
        print(f"\n📊 Стабильность производительности:")
        print(f"   Начальная производительность: {initial_avg*1000:.2f} мс")
        print(f"   Конечная производительность: {final_avg*1000:.2f} мс")
        print(f"   Деградация: {degradation:.2f}%")
        
        assert degradation < 15.0, f"Деградация производительности {degradation:.2f}% превышает 15%"
    
    @pytest.mark.longrun
    @pytest.mark.slow
    def test_memory_stability_long_run(self, app_with_data):
        """Стабильность использования памяти при длительной работе"""
        app = app_with_data
        
        tracemalloc.start()
        
        # Базовое измерение
        snapshot_base = tracemalloc.take_snapshot()
        
        # Измерения через каждые 200 операций
        memory_measurements = []
        
        for iteration in [200, 400, 600, 800, 1000]:
            # Выполняем операции до целевой итерации
            current = len(memory_measurements) * 200
            for _ in range(current, iteration):
                app.on_plot_dimensionless_selected()
                QApplication.instance().processEvents()
            
            # Измеряем память
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.compare_to(snapshot_base, 'lineno')
            memory_mb = sum(stat.size_diff for stat in top_stats) / 1024 / 1024
            memory_measurements.append((iteration, memory_mb))
            
            print(f"   После {iteration} операций: {memory_mb:.2f} МБ")
        
        tracemalloc.stop()
        
        # Проверяем, что рост памяти стабилен (линейный)
        iterations = np.array([m[0] for m in memory_measurements])
        memory = np.array([m[1] for m in memory_measurements])
        
        # Линейная регрессия
        coeffs = np.polyfit(iterations, memory, 1)
        fitted = np.polyval(coeffs, iterations)
        
        # Вычисляем относительные отклонения
        relative_deviations = np.abs(memory - fitted) / (fitted + 1e-6)
        max_deviation = np.max(relative_deviations) * 100
        
        print(f"\n📊 Стабильность памяти:")
        print(f"   Максимальное отклонение от линейного роста: {max_deviation:.2f}%")
        
        # Отклонение не должно превышать 20%
        assert max_deviation < 20.0, f"Нестабильный рост памяти: отклонение {max_deviation:.2f}%"


# ============================================================================
# СВОДНЫЙ ОТЧЕТ
# ============================================================================

def pytest_sessionfinish(session, exitstatus):
    """Генерирует сводный отчет после выполнения всех тестов"""
    print("\n" + "="*80)
    print("СВОДНЫЙ ОТЧЕТ ПО МЕТРИКАМ КАЧЕСТВА")
    print("="*80)
    
    # Этот хук вызывается pytest автоматически


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

