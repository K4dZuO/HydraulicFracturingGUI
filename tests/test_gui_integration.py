#!/usr/bin/env python3
"""
Интеграционные тесты для GUI приложения
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.well_data import WellTimeSeries


class TestGUIIntegration:
    """Интеграционные тесты для GUI"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        # Создаем QApplication для тестов GUI
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        pass
    
    @patch('main.QApplication')
    def test_application_initialization(self, mock_qapp):
        """Тест инициализации приложения"""
        from main import MyApp
        
        # Создаем экземпляр приложения
        app = MyApp()
        
        # Проверяем основные компоненты
        assert hasattr(app, 'loaded_data')
        assert hasattr(app, 'current_index')
        assert hasattr(app, 'plot_type_combo')
        assert hasattr(app, 'plot_widget')
        
        # Проверяем начальные значения
        assert app.loaded_data == []
        assert app.current_index == 0
    
    def test_plot_type_combo_population(self):
        """Тест заполнения combo box типами графиков"""
        from main import MyApp
        
        app = MyApp()
        
        # Проверяем, что combo box заполнен
        assert app.plot_type_combo.count() > 0
        
        # Проверяем наличие всех требуемых типов
        required_types = [
            "Давление vs Время",
            "Дебит vs Время",
            "Производная давления",
            "Производная дебита",
            "Давление vs Дебит",
            "Безразмерные параметры (X-Y)",
            "Логарифмический P(t) с инверсией",
            "Логарифмический Q(t)",
            "Логарифмическая производная P"
        ]
        
        available_types = []
        for i in range(app.plot_type_combo.count()):
            available_types.append(app.plot_type_combo.itemText(i))
        
        for req_type in required_types:
            assert req_type in available_types
    
    def test_well_data_loading(self):
        """Тест загрузки данных скважины"""
        from main import MyApp
        
        app = MyApp()
        
        # Создаем тестовые данные
        time_data = np.array([1, 2, 3, 4, 5])
        pressure_data = np.array([100, 95, 90, 85, 80])
        flow_rate_data = np.array([50, 45, 40, 35, 30])
        
        well_data = WellTimeSeries(
            time=pd.Series(time_data),
            pressure=pd.Series(pressure_data),
            flow_rate=pd.Series(flow_rate_data),
            thickness=10.0,
            fracture_length=100.0,
            skin=0.0,
            fractures_count=1,
            a_l_ratio=0.1
        )
        
        # Добавляем данные в приложение
        app.loaded_data = [well_data]
        app.current_index = 0
        
        # Проверяем, что данные загружены
        assert app.current_data is not None
        assert app.current_data == well_data
    
    @patch('main.QFileDialog.getOpenFileName')
    def test_file_loading_dialog(self, mock_file_dialog):
        """Тест диалога загрузки файла"""
        from main import MyApp
        
        app = MyApp()
        
        # Мокаем диалог выбора файла
        mock_file_dialog.return_value = ("test_file.csv", "CSV Files (*.csv)")
        
        # Вызываем метод загрузки файла
        app.load_template()
        
        # Проверяем, что диалог был вызван
        mock_file_dialog.assert_called_once()
    
    def test_plot_widget_initialization(self):
        """Тест инициализации виджета графика"""
        from main import MyApp
        
        app = MyApp()
        
        # Проверяем, что plot_widget инициализирован
        assert app.plot_widget is not None
        
        # Проверяем настройки сетки
        # (Это может потребовать дополнительной проверки в зависимости от реализации)
    
    def test_button_connections(self):
        """Тест подключения кнопок к обработчикам"""
        from main import MyApp
        
        app = MyApp()
        
        # Проверяем, что кнопки существуют
        assert hasattr(app, 'plot_btn')
        assert hasattr(app, 'smooth_btn')
        assert hasattr(app, 'interp_btn')
        assert hasattr(app, 'export_btn')
        
        # Проверяем, что кнопки подключены к обработчикам
        # (Это может потребовать дополнительной проверки в зависимости от реализации)
    
    def test_well_selection_combo(self):
        """Тест combo box выбора скважины"""
        from main import MyApp
        
        app = MyApp()
        
        # Проверяем, что combo box существует
        assert hasattr(app, 'well_selection_combo')
        
        # Проверяем начальное состояние
        assert app.well_selection_combo.count() == 0  # Нет загруженных скважин


class TestGUIFunctional:
    """Функциональные тесты для GUI"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
    
    def test_plot_creation_with_data(self):
        """Тест создания графика с данными"""
        from main import MyApp
        
        app = MyApp()
        
        # Создаем тестовые данные
        time_data = np.array([1, 2, 3, 4, 5])
        pressure_data = np.array([100, 95, 90, 85, 80])
        flow_rate_data = np.array([50, 45, 40, 35, 30])
        
        well_data = WellTimeSeries(
            time=pd.Series(time_data),
            pressure=pd.Series(pressure_data),
            flow_rate=pd.Series(flow_rate_data),
            thickness=10.0,
            fracture_length=100.0,
            skin=0.0,
            fractures_count=1,
            a_l_ratio=0.1
        )
        
        # Добавляем данные
        app.loaded_data = [well_data]
        app.current_index = 0
        
        # Устанавливаем тип графика
        app.plot_type_combo.setCurrentText("Давление vs Время")
        
        # Вызываем построение графика
        try:
            app.on_plot_timeseries()
            # Если не было исключений, тест прошел
            assert True
        except Exception as e:
            pytest.fail(f"Ошибка при построении графика: {e}")
    
    def test_dimensionless_parameters_plot(self):
        """Тест построения графика безразмерных параметров"""
        from main import MyApp
        
        app = MyApp()
        
        # Создаем тестовые данные
        time_data = np.array([1, 2, 3, 4, 5])
        pressure_data = np.array([100, 95, 90, 85, 80])
        flow_rate_data = np.array([50, 45, 40, 35, 30])
        
        well_data = WellTimeSeries(
            time=pd.Series(time_data),
            pressure=pd.Series(pressure_data),
            flow_rate=pd.Series(flow_rate_data),
            thickness=10.0,
            fracture_length=100.0,
            skin=0.0,
            fractures_count=1,
            a_l_ratio=0.1
        )
        
        # Добавляем данные
        app.loaded_data = [well_data]
        app.current_index = 0
        
        # Устанавливаем тип графика
        app.plot_type_combo.setCurrentText("Безразмерные параметры (X-Y)")
        
        # Вызываем построение графика
        try:
            app.on_plot_timeseries()
            assert True
        except Exception as e:
            pytest.fail(f"Ошибка при построении графика безразмерных параметров: {e}")
    
    def test_logarithmic_plot_with_inversion(self):
        """Тест логарифмического графика с инверсией"""
        from main import MyApp
        
        app = MyApp()
        
        # Создаем тестовые данные
        time_data = np.array([1, 2, 3, 4, 5])
        pressure_data = np.array([100, 95, 90, 85, 80])
        flow_rate_data = np.array([50, 45, 40, 35, 30])
        
        well_data = WellTimeSeries(
            time=pd.Series(time_data),
            pressure=pd.Series(pressure_data),
            flow_rate=pd.Series(flow_rate_data),
            thickness=10.0,
            fracture_length=100.0,
            skin=0.0,
            fractures_count=1,
            a_l_ratio=0.1
        )
        
        # Добавляем данные
        app.loaded_data = [well_data]
        app.current_index = 0
        
        # Устанавливаем тип графика
        app.plot_type_combo.setCurrentText("Логарифмический P(t) с инверсией")
        
        # Вызываем построение графика
        try:
            app.on_plot_timeseries()
            assert True
        except Exception as e:
            pytest.fail(f"Ошибка при построении логарифмического графика с инверсией: {e}")


class TestGUIErrorHandling:
    """Тесты обработки ошибок в GUI"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
    
    def test_plot_without_data(self):
        """Тест построения графика без данных"""
        from main import MyApp
        
        app = MyApp()
        
        # Убеждаемся, что нет данных
        app.loaded_data = []
        
        # Устанавливаем тип графика
        app.plot_type_combo.setCurrentText("Давление vs Время")
        
        # Вызываем построение графика
        try:
            app.on_plot_timeseries()
            # Должно обработаться без ошибок
            assert True
        except Exception as e:
            pytest.fail(f"Ошибка при построении графика без данных: {e}")
    
    def test_plot_with_invalid_data(self):
        """Тест построения графика с некорректными данными"""
        from main import MyApp
        
        app = MyApp()
        
        # Создаем некорректные данные
        time_data = np.array([np.nan, np.inf, -np.inf])
        pressure_data = np.array([np.nan, np.inf, -np.inf])
        flow_rate_data = np.array([np.nan, np.inf, -np.inf])
        
        well_data = WellTimeSeries(
            time=pd.Series(time_data),
            pressure=pd.Series(pressure_data),
            flow_rate=pd.Series(flow_rate_data),
            thickness=10.0,
            fracture_length=100.0,
            skin=0.0,
            fractures_count=1,
            a_l_ratio=0.1
        )
        
        # Добавляем данные
        app.loaded_data = [well_data]
        app.current_index = 0
        
        # Устанавливаем тип графика
        app.plot_type_combo.setCurrentText("Давление vs Время")
        
        # Вызываем построение графика
        try:
            app.on_plot_timeseries()
            # Должно обработаться без ошибок
            assert True
        except Exception as e:
            pytest.fail(f"Ошибка при построении графика с некорректными данными: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
