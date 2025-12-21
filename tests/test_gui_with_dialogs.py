#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты GUI с автоматическим закрытием диалоговых окон
Эмулирует полное пользовательское взаимодействие
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

from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest

from schemas.well_data import WellTimeSeries


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
    """Создает приложение с загруженными тестовыми данными (БЕЗ test_mode!)"""
    from main import MyApp
    
    # Создаем приложение в ОБЫЧНОМ режиме (test_mode=False)
    # Диалоги будут появляться!
    app = MyApp(test_mode=False)
    
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


def close_message_boxes():
    """Автоматически закрывает все открытые QMessageBox"""
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QMessageBox):
            # Находим кнопку OK и нажимаем её
            for button in widget.buttons():
                if widget.buttonRole(button) == QMessageBox.AcceptRole:
                    button.click()
                    break


def setup_dialog_auto_closer(interval_ms=50):
    """
    Настраивает таймер для автоматического закрытия диалогов
    
    Args:
        interval_ms: Интервал проверки в миллисекундах
    
    Returns:
        QTimer object
    """
    timer = QTimer()
    timer.setInterval(interval_ms)
    timer.timeout.connect(close_message_boxes)
    timer.start()
    return timer


# ============================================================================
# ТЕСТЫ С РЕАЛЬНЫМИ ДИАЛОГАМИ
# ============================================================================

class TestGUIWithRealDialogs:
    """Тесты GUI с реальными диалоговыми окнами (автоматически закрываются)"""
    
    @pytest.mark.performance
    def test_plot_button_with_dialog_auto_close(self, app_with_data):
        """Тест кнопки 'Построить график' с автозакрытием диалогов"""
        app = app_with_data
        
        # Настраиваем автоматическое закрытие диалогов каждые 50ms
        timer = setup_dialog_auto_closer(interval_ms=50)
        
        times = []
        
        try:
            for _ in range(10):
                start = time.perf_counter()
                
                # Вызываем метод - диалог появится и автоматически закроется
                app.on_plot_dimensionless_selected()
                
                # Обрабатываем события (включая закрытие диалогов)
                QApplication.instance().processEvents()
                
                # Небольшая задержка для гарантии закрытия
                QTest.qWait(100)
                
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)
            
            avg_time = np.mean(times)
            max_time = np.max(times)
            
            print(f"\n📊 Построить график (с диалогами): среднее={avg_time:.2f}ms, макс={max_time:.2f}ms")
            
            # Критерий более мягкий - учитываем время на диалоги
            # Диалог + закрытие ≈ 100-200ms
            assert avg_time < 1000, f"Среднее время {avg_time:.2f}ms превышает 1000ms"
            
        finally:
            # Останавливаем таймер
            timer.stop()
    
    @pytest.mark.performance
    def test_interpolation_button_with_dialog(self, app_with_data):
        """Тест кнопки 'Интерполяция' с автозакрытием диалогов"""
        app = app_with_data
        
        # Добавляем пропуски для интерполяции
        app.loaded_data[0].pressure.iloc[10:15] = np.nan
        
        timer = setup_dialog_auto_closer(interval_ms=50)
        
        times = []
        
        try:
            for _ in range(5):
                # Восстанавливаем пропуски
                app.loaded_data[0].pressure.iloc[10:15] = np.nan
                
                start = time.perf_counter()
                app.on_interpolate_data()
                QApplication.instance().processEvents()
                QTest.qWait(100)
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)
            
            avg_time = np.mean(times)
            print(f"\n📊 Интерполяция (с диалогами): среднее={avg_time:.2f}ms")
            
            # ML интерполяция + диалог
            assert avg_time < 10000, f"Среднее время {avg_time:.2f}ms превышает 10000ms"
            
        finally:
            timer.stop()
    
    @pytest.mark.performance
    def test_ml_filter_button_with_dialog(self, app_with_data):
        """Тест кнопки 'ML фильтрация' с автозакрытием диалогов"""
        app = app_with_data
        
        timer = setup_dialog_auto_closer(interval_ms=50)
        
        times = []
        
        try:
            for _ in range(10):
                start = time.perf_counter()
                app.on_ml_filter()
                QApplication.instance().processEvents()
                QTest.qWait(100)
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)
            
            avg_time = np.mean(times)
            print(f"\n📊 ML фильтрация (с диалогами): среднее={avg_time:.2f}ms")
            
            assert avg_time < 1500, f"Среднее время {avg_time:.2f}ms превышает 1500ms"
            
        finally:
            timer.stop()
    
    @pytest.mark.performance
    def test_detect_outliers_button_with_dialog(self, app_with_data):
        """Тест кнопки 'Обнаружить выбросы' с автозакрытием диалогов"""
        app = app_with_data
        
        timer = setup_dialog_auto_closer(interval_ms=50)
        
        times = []
        
        try:
            for _ in range(10):
                start = time.perf_counter()
                app.on_detect_outliers()
                QApplication.instance().processEvents()
                QTest.qWait(100)
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)
            
            avg_time = np.mean(times)
            print(f"\n📊 Обнаружить выбросы (с диалогами): среднее={avg_time:.2f}ms")
            
            assert avg_time < 1000, f"Среднее время {avg_time:.2f}ms превышает 1000ms"
            
        finally:
            timer.stop()
    
    @pytest.mark.performance
    def test_analyze_flow_regime_button_with_dialog(self, app_with_data):
        """Тест кнопки 'Анализ режима течения' с автозакрытием диалогов"""
        app = app_with_data
        
        timer = setup_dialog_auto_closer(interval_ms=50)
        
        times = []
        
        try:
            for _ in range(10):
                start = time.perf_counter()
                app.on_analyze_flow_regime()
                QApplication.instance().processEvents()
                QTest.qWait(100)
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)
            
            avg_time = np.mean(times)
            print(f"\n📊 Анализ режима течения (с диалогами): среднее={avg_time:.2f}ms")
            
            assert avg_time < 1000, f"Среднее время {avg_time:.2f}ms превышает 1000ms"
            
        finally:
            timer.stop()
    
    @pytest.mark.performance
    def test_compute_productivity_button_with_dialog(self, app_with_data):
        """Тест кнопки 'Индекс продуктивности' с автозакрытием диалогов"""
        app = app_with_data
        
        timer = setup_dialog_auto_closer(interval_ms=50)
        
        times = []
        
        try:
            for _ in range(10):
                start = time.perf_counter()
                app.on_compute_productivity()
                QApplication.instance().processEvents()
                QTest.qWait(100)
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)
            
            avg_time = np.mean(times)
            print(f"\n📊 Индекс продуктивности (с диалогами): среднее={avg_time:.2f}ms")
            
            assert avg_time < 1000, f"Среднее время {avg_time:.2f}ms превышает 1000ms"
            
        finally:
            timer.stop()
    
    @pytest.mark.performance
    def test_detect_transitions_button_with_dialog(self, app_with_data):
        """Тест кнопки 'Переходы режимов' с автозакрытием диалогов"""
        app = app_with_data
        
        timer = setup_dialog_auto_closer(interval_ms=50)
        
        times = []
        
        try:
            for _ in range(10):
                start = time.perf_counter()
                app.on_detect_transitions()
                QApplication.instance().processEvents()
                QTest.qWait(100)
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)
            
            avg_time = np.mean(times)
            print(f"\n📊 Переходы режимов (с диалогами): среднее={avg_time:.2f}ms")
            
            assert avg_time < 1000, f"Среднее время {avg_time:.2f}ms превышает 1000ms"
            
        finally:
            timer.stop()


# ============================================================================
# ТЕСТ СТАБИЛЬНОСТИ С ДИАЛОГАМИ
# ============================================================================

class TestDialogClosingMechanism:
    """Тесты механизма автоматического закрытия диалогов"""
    
    @pytest.mark.unit
    def test_dialog_auto_close_mechanism(self, app_with_data, qapp):
        """Проверка, что механизм автозакрытия работает"""
        app = app_with_data
        
        timer = setup_dialog_auto_closer(interval_ms=50)
        
        try:
            # Вызываем операцию, которая показывает диалог
            app.on_plot_dimensionless_selected()
            
            # Ждем немного
            QTest.qWait(200)
            
            # Обрабатываем события
            qapp.processEvents()
            
            # Проверяем, что диалогов не осталось
            open_dialogs = [w for w in QApplication.topLevelWidgets() 
                           if isinstance(w, QMessageBox) and w.isVisible()]
            
            assert len(open_dialogs) == 0, f"Остались открытые диалоги: {len(open_dialogs)}"
            
        finally:
            timer.stop()
    
    @pytest.mark.unit
    def test_multiple_dialogs_close(self, app_with_data, qapp):
        """Проверка закрытия множественных диалогов"""
        app = app_with_data
        
        timer = setup_dialog_auto_closer(interval_ms=50)
        
        try:
            # Вызываем несколько операций подряд
            operations = [
                app.on_plot_dimensionless_selected,
                app.on_ml_filter,
                app.on_detect_outliers,
            ]
            
            for op in operations:
                op()
                qapp.processEvents()
            
            # Ждем закрытия всех диалогов
            QTest.qWait(300)
            qapp.processEvents()
            
            # Проверяем
            open_dialogs = [w for w in QApplication.topLevelWidgets() 
                           if isinstance(w, QMessageBox) and w.isVisible()]
            
            assert len(open_dialogs) == 0, f"Остались открытые диалоги: {len(open_dialogs)}"
            
        finally:
            timer.stop()


# ============================================================================
# НАГРУЗОЧНЫЙ ТЕСТ С ДИАЛОГАМИ
# ============================================================================

class TestLoadWithDialogs:
    """Нагрузочные тесты с реальными диалогами"""
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_100_operations_with_dialogs(self, app_with_data):
        """100 операций с автозакрытием диалогов - проверка стабильности"""
        app = app_with_data
        
        timer = setup_dialog_auto_closer(interval_ms=30)  # Чаще проверяем
        
        failures = 0
        
        try:
            for i in range(100):
                try:
                    app.on_plot_dimensionless_selected()
                    QApplication.instance().processEvents()
                    
                    if i % 10 == 0:
                        QTest.qWait(100)  # Периодически ждем
                        
                except Exception as e:
                    failures += 1
                    print(f"Итерация {i}: {e}")
            
            failure_rate = (failures / 100) * 100
            
            print(f"\n📊 Failure rate (с диалогами): {failure_rate:.2f}%")
            
            assert failure_rate < 1.0, f"Failure rate {failure_rate:.2f}% превышает 1%"
            
        finally:
            timer.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

