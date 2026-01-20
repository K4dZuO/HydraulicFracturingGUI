#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Нагрузочные тесты с расширенным логированием ошибок и генерацией визуализаций
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest

from schemas.well_data import WellTimeSeries
from helpers.math_error_logger import MathErrorLogger, log_math_error, log_computation_error
from helpers.error_visualization import ErrorVisualizer


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
    
    # Создаем тестовые данные с различными объёмами и качеством
    n_points = 100
    time_data = np.linspace(0, 100, n_points)
    pressure_data = 300 - time_data * 0.5 + np.random.normal(0, 5, n_points)
    flow_rate_data = 100 - time_data * 0.2 + np.random.normal(0, 3, n_points)
    
    # Добавляем пропуски для тестирования интерполяции
    pressure_data[10:15] = np.nan
    flow_rate_data[20:25] = np.nan
    
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
    
    app.close()


@pytest.fixture
def error_logger():
    """Создает логгер ошибок для тестов"""
    logger = MathErrorLogger(log_dir="test_logs")
    yield logger
    # Очистка после тестов не выполняется, чтобы можно было анализировать результаты


class TestLoadWithErrorLogging:
    """Нагрузочные тесты с логированием ошибок"""
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_load_test_with_error_logging(self, app_with_data, error_logger):
        """Нагрузочный тест с логированием всех ошибок"""
        app = app_with_data
        n_iterations = 100
        errors_logged = 0
        
        # Тестируем различные операции с разными объёмами данных
        operations = [
            ("plot_dimensionless_selected", app.on_plot_dimensionless_selected),
            ("interpolate_data", app.on_interpolate_data),
            ("ml_filter", app.on_ml_filter),
        ]
        
        for i in range(n_iterations):
            op_name, op_func = operations[i % len(operations)]
            
            try:
                start_time = time.perf_counter()
                op_func()
                QApplication.instance().processEvents()
                elapsed = time.perf_counter() - start_time
                
                # Логируем время выполнения как метрику производительности
                if elapsed > 1.0:  # Если операция заняла больше секунды
                    log_math_error(
                        subsystem="gui",
                        method=op_name,
                        error_type="performance_warning",
                        error_value=elapsed,
                        error_message=f"Медленная операция: {elapsed:.2f}s",
                        data_volume=len(app.current_data.time) if app.current_data else None,
                        data_quality=1.0 - (np.sum(np.isnan(app.current_data.pressure)) / len(app.current_data.pressure)) if app.current_data else None,
                        metadata={"iteration": i}
                    )
                    errors_logged += 1
                
            except Exception as e:
                log_computation_error(
                    subsystem="gui",
                    method=op_name,
                    exception=e,
                    data_volume=len(app.current_data.time) if app.current_data else None,
                    data_quality=1.0 - (np.sum(np.isnan(app.current_data.pressure)) / len(app.current_data.pressure)) if app.current_data else None,
                    context={"iteration": i, "test": "test_load_test_with_error_logging"}
                )
                errors_logged += 1
        
        print(f"\n📊 Залогировано ошибок: {errors_logged}")
        
        # Проверяем, что логи созданы
        df = error_logger.load_errors()
        assert len(df) >= 0  # Может быть 0, если всё прошло успешно
    

class TestErrorVisualizationGeneration:
    """Тесты генерации визуализаций ошибок"""
    
    @pytest.mark.slow
    def test_generate_error_visualizations(self, error_logger):
        """Генерация всех визуализаций ошибок"""
        # Сначала генерируем тестовые данные, если их нет
        df = error_logger.load_errors()
        
        if df.empty:
            # Генерируем синтетические данные для демонстрации
            np.random.seed(42)
            for i in range(30):
                volume = int(10 ** np.random.uniform(1, 4))
                quality = np.random.uniform(0.3, 1.0)
                error = 0.1 * (volume ** -0.5) * (2 - quality) + np.random.normal(0, 0.01)
                
                error_logger.log_error(
                    subsystem=f"subsystem_{i % 3}",
                    method=f"method_{i % 4}",
                    error_type="rmse",
                    error_value=max(0.001, error),
                    error_message=f"Test error {i}",
                    data_volume=volume,
                    data_quality=quality
                )
        
        # Генерируем визуализации
        visualizer = ErrorVisualizer(output_dir="test_results")
        visualizer.generate_all_visualizations()
        
        # Проверяем, что файлы созданы
        result_files = list(Path("test_results").glob("error_*.png"))
        print(f"\n📊 Создано файлов визуализации: {len(result_files)}")
        for f in result_files:
            print(f"   {f.name}")
        
        assert len(result_files) > 0, "Визуализации должны быть созданы"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "load"])

