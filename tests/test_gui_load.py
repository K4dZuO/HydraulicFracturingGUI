#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Нагрузочные тесты для GUI приложения
- 1000 кликов по "Построить график" - программа не ломается
- 100 кликов по интерполяции - программа не виснет
- Проверка идемпотентности интерполяции
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest

from schemas.well_data import WellTimeSeries
from helpers.math_error_logger import log_computation_error, get_logger


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
    
    app = MyApp(test_mode=True)  # Отключаем QMessageBox в тестах
    
    # Создаем тестовые данные с пропусками для тестирования интерполяции
    n_points = 50
    time_data = np.linspace(0, 100, n_points)
    pressure_data = 300 - time_data * 0.5 + np.random.normal(0, 5, n_points)
    flow_rate_data = 100 - time_data * 0.2 + np.random.normal(0, 3, n_points)
    
    # Добавляем несколько пропусков (NaN) для тестирования интерполяции
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


class TestGUILoadPlotButton:
    """Нагрузочные тесты для кнопки 'Построить график'"""
    
    @pytest.mark.slow
    def test_1000_clicks_plot_button_stability(self, app_with_data):
        """1000 кликов по кнопке построения графика - программа не должна ломаться"""
        app = app_with_data
        
        initial_memory = None
        crash_count = 0
        exception_count = 0
        
        start_time = time.time()
        
        for i in range(1000):
            try:
                # Симулируем клик по кнопке
                if hasattr(app, 'plot_btn') and app.plot_btn:
                    # Вызываем обработчик напрямую (быстрее чем через QTest)
                    app.on_plot_dimensionless_selected()
                
                # Проверяем, что приложение не зависло
                qapp = QApplication.instance()
                qapp.processEvents()  # Обрабатываем события
                
                # Каждые 100 итераций проверяем, что всё ещё работает
                if i % 100 == 0:
                    # Проверяем, что виджет графика существует
                    assert hasattr(app, 'dimensionless_plot'), "Виджет графика исчез"
                    assert app.dimensionless_plot is not None, "Виджет графика стал None"
                    
            except Exception as e:
                exception_count += 1
                print(f"Итерация {i}: исключение {e}")
                # Логируем ошибку
                log_computation_error(
                    subsystem="gui",
                    method="plot_dimensionless_selected",
                    exception=e,
                    data_volume=len(app.current_data.time) if app.current_data else None,
                    context={"iteration": i, "test": "test_1000_clicks_plot_button_stability"}
                )
                if exception_count > 10:
                    pytest.fail(f"Слишком много исключений после {i} итераций")
        
        elapsed_time = time.time() - start_time
        
        # Проверки
        assert exception_count == 0, f"Было {exception_count} исключений"
        assert hasattr(app, 'dimensionless_plot'), "Виджет графика должен существовать"
        assert app.dimensionless_plot is not None, "Виджет графика не должен быть None"
        
        print(f"\n✓ 1000 кликов выполнено за {elapsed_time:.2f} секунд")
        print(f"  Среднее время на клик: {elapsed_time/1000*1000:.2f} мс")
    
    def test_rapid_plot_clicks_no_hang(self, app_with_data):
        """Быстрые клики по графику не должны вызывать зависание"""
        app = app_with_data
        
        # Быстрая последовательность из 50 кликов
        for i in range(50):
            app.on_plot_dimensionless_selected()
            
            # Обрабатываем события между кликами
            QApplication.instance().processEvents()
            
            # Проверяем, что приложение отзывчиво
            if i % 10 == 0:
                # Попытка обработать события должна завершиться быстро
                start = time.time()
                QApplication.instance().processEvents()
                elapsed = time.time() - start
                
                assert elapsed < 0.1, f"Обработка событий заняла {elapsed:.2f}s - возможное зависание"
    
    def test_plot_button_memory_usage(self, app_with_data):
        """Проверка использования памяти при многократных кликах"""
        import tracemalloc
        
        app = app_with_data
        
        tracemalloc.start()
        
        # Базовое измерение
        snapshot1 = tracemalloc.take_snapshot()
        
        # Выполняем 100 кликов
        for i in range(100):
            app.on_plot_dimensionless_selected()
            QApplication.instance().processEvents()
        
        snapshot2 = tracemalloc.take_snapshot()
        
        # Вычисляем разницу
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Суммируем использование памяти
        total_memory = sum(stat.size_diff for stat in top_stats)
        
        # Использование памяти не должно расти слишком сильно (менее 100 МБ)
        assert total_memory < 100 * 1024 * 1024, \
            f"Использование памяти выросло на {total_memory / 1024 / 1024:.2f} МБ"
        
        tracemalloc.stop()
        
        print(f"\n✓ Использование памяти: {total_memory / 1024 / 1024:.2f} МБ")


class TestGUILoadInterpolation:
    """Нагрузочные тесты для интерполяции"""
    
    @pytest.mark.slow
    def test_100_clicks_interpolation_no_hang(self, app_with_data):
        """100 кликов по интерполяции - странa не должна виснуть"""
        app = app_with_data
        
        start_time = time.time()
        exception_count = 0
        hang_detected = False
        
        for i in range(100):
            try:
                # Проверяем время начала операции
                op_start = time.time()
                
                # Вызываем интерполяцию
                app.on_interpolate_data()
                
                # Обрабатываем события
                QApplication.instance().processEvents()
                
                # Проверяем, что операция завершилась быстро (не более 5 секунд)
                op_elapsed = time.time() - op_start
                if op_elapsed > 5.0:
                    hang_detected = True
                    print(f"⚠️ Предупреждение: операция {i} заняла {op_elapsed:.2f}s")
                
                # Каждые 10 итераций проверяем отзывчивость
                if i % 10 == 0:
                    # Попытка обработать события должна завершиться быстро
                    event_start = time.time()
                    QApplication.instance().processEvents()
                    event_elapsed = time.time() - event_start
                    
                    if event_elapsed > 1.0:
                        hang_detected = True
                        print(f"⚠️ Предупреждение: обработка событий на итерации {i} заняла {event_elapsed:.2f}s")
                
            except Exception as e:
                exception_count += 1
                print(f"Итерация {i}: исключение {e}")
                # Логируем ошибку
                log_computation_error(
                    subsystem="gui",
                    method="interpolate_data",
                    exception=e,
                    data_volume=len(app.current_data.time) if app.current_data else None,
                    context={"iteration": i, "test": "test_100_clicks_interpolation_no_hang"}
                )
                if exception_count > 5:
                    pytest.fail(f"Слишком много исключений после {i} итераций")
        
        elapsed_time = time.time() - start_time
        
        # Проверки
        assert exception_count == 0, f"Было {exception_count} исключений"
        assert not hang_detected, "Обнаружено зависание"
        assert app.current_data is not None, "Данные должны существовать"
        
        print(f"\n✓ 100 кликов по интерполяции выполнено за {elapsed_time:.2f} секунд")
        print(f"  Среднее время на операцию: {elapsed_time/100*1000:.2f} мс")
    
    def test_interpolation_idempotency(self, app_with_data):
        """Проверка идемпотентности: результат интерполяции не меняется при повторном вызове"""
        app = app_with_data
        
        # Сохраняем исходные данные
        original_pressure = app.current_data.pressure.copy()
        original_flow_rate = app.current_data.flow_rate.copy()
        
        # Первая интерполяция
        app.on_interpolate_data()
        QApplication.instance().processEvents()
        
        first_pressure = app.current_data.pressure.copy()
        first_flow_rate = app.current_data.flow_rate.copy()
        
        # Вторая интерполяция на тех же данных
        app.on_interpolate_data()
        QApplication.instance().processEvents()
        
        second_pressure = app.current_data.pressure.copy()
        second_flow_rate = app.current_data.flow_rate.copy()
        
        # Результаты должны быть одинаковыми (идемпотентность)
        # Используем относительную разницу из-за возможных небольших численных различий
        pressure_diff = np.abs(first_pressure - second_pressure) / (np.abs(first_pressure) + 1e-10)
        flow_rate_diff = np.abs(first_flow_rate - second_flow_rate) / (np.abs(first_flow_rate) + 1e-10)
        
        max_pressure_diff = np.max(pressure_diff)
        max_flow_rate_diff = np.max(flow_rate_diff)
        
        # Максимальная относительная разница должна быть очень мала (< 0.1%)
        assert max_pressure_diff < 1e-3, \
            f"Идемпотентность нарушена для давления: макс. разница {max_pressure_diff*100:.4f}%"
        assert max_flow_rate_diff < 1e-3, \
            f"Идемпотентность нарушена для дебита: макс. разница {max_flow_rate_diff*100:.4f}%"
        
        print(f"\n✓ Идемпотентность проверена: макс. разница давления {max_pressure_diff*100:.6f}%, дебита {max_flow_rate_diff*100:.6f}%")
    
    def test_interpolation_on_already_interpolated_data(self, app_with_data):
        """Тест: интерполяция уже интерполированных данных не меняет результат"""
        app = app_with_data
        
        # Сохраняем исходные данные
        original_pressure = app.current_data.pressure.values.copy()
        original_flow_rate = app.current_data.flow_rate.values.copy()
        
        # Первая интерполяция
        app.on_interpolate_data()
        QApplication.instance().processEvents()
        
        first_pressure = app.current_data.pressure.values.copy()
        first_flow_rate = app.current_data.flow_rate.values.copy()
        
        # Проверяем, что данные изменились (если были пропуски)
        if np.any(np.isnan(original_pressure)):
            assert not np.array_equal(original_pressure, first_pressure), \
                "Интерполяция должна была заполнить пропуски"
        
        # Вторая интерполяция на уже интерполированных данных
        app.on_interpolate_data()
        QApplication.instance().processEvents()
        
        second_pressure = app.current_data.pressure.values.copy()
        second_flow_rate = app.current_data.flow_rate.values.copy()
        
        # Результаты должны быть одинаковыми
        np.testing.assert_array_almost_equal(
            first_pressure, second_pressure, decimal=6,
            err_msg="Результат интерполяции изменился при повторном вызове"
        )
        np.testing.assert_array_almost_equal(
            first_flow_rate, second_flow_rate, decimal=6,
            err_msg="Результат интерполяции изменился при повторном вызове"
        )


class TestGUIPerformance:
    """Тесты производительности GUI"""
    
    def test_plot_creation_performance(self, app_with_data):
        """Проверка производительности создания графика"""
        app = app_with_data
        
        times = []
        
        for i in range(20):
            start = time.time()
            app.on_plot_dimensionless_selected()
            QApplication.instance().processEvents()
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        # Среднее время должно быть разумным (< 500 мс)
        assert avg_time < 0.5, f"Среднее время создания графика {avg_time*1000:.2f} мс слишком велико"
        
        # Максимальное время тоже должно быть разумным (< 1 с)
        assert max_time < 1.0, f"Максимальное время создания графика {max_time*1000:.2f} мс слишком велико"
        
        print(f"\n✓ Производительность графика: среднее {avg_time*1000:.2f} мс, макс. {max_time*1000:.2f} мс")
    
    def test_interpolation_performance(self, app_with_data):
        """Проверка производительности интерполяции"""
        app = app_with_data
        
        times = []
        
        for i in range(10):
            start = time.time()
            app.on_interpolate_data()
            QApplication.instance().processEvents()
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        # Среднее время интерполяции должно быть разумным (< 5 с для ML моделей)
        assert avg_time < 5.0, f"Среднее время интерполяции {avg_time*1000:.2f} мс слишком велико"
        
        # Максимальное время (< 15 с с учетом ML обучения и проверки стабильности)
        assert max_time < 15.0, f"Максимальное время интерполяции {max_time*1000:.2f} мс слишком велико"
        
        print(f"\n✓ Производительность интерполяции: среднее {avg_time*1000:.2f} мс, макс. {max_time*1000:.2f} мс")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "slow"])

