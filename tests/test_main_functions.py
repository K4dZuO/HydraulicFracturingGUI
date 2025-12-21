#!/usr/bin/env python3
"""
Тесты для функций main.py
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from PySide6.QtWidgets import QApplication

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.well_data import WellTimeSeries
from main import MyApp, DEFAULT_K, DEFAULT_MU, DEFAULT_B, DEFAULT_PHI, DEFAULT_C_T


class TestMainFunctions:
    """Тесты для методов класса MyApp"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        # Создаем QApplication для тестов GUI
        if not QApplication.instance():
            self.app_instance = QApplication([])
        else:
            self.app_instance = QApplication.instance()
        
        # Создаем приложение в тестовом режиме
        self.app = MyApp(test_mode=True)
        
        # Создаем тестовые данные
        self.time = np.array([1, 2, 3, 4, 5])
        self.pressure = np.array([100, 95, 90, 85, 80])
        self.flow_rate = np.array([50, 45, 40, 35, 30])
        
        self.well_data = WellTimeSeries(
            time=pd.Series(self.time),
            pressure=pd.Series(self.pressure),
            flow_rate=pd.Series(self.flow_rate),
            thickness=10.0,
            fracture_length=100.0,
            fracture_width=0.01,
            skin=0.0,
            fractures_count=1,
            a_l_ratio=0.1
        )
        self.app.loaded_data = [self.well_data]
        self.app.current_index = 0
    
    def test_get_well_params_defaults(self):
        """Тест создания параметров скважины с значениями по умолчанию"""
        params = self.app._get_params(self.well_data)
        
        assert params['k'] == DEFAULT_K
        assert params['mu'] == DEFAULT_MU
        assert params['B'] == DEFAULT_B
        assert params['phi'] == DEFAULT_PHI
        assert params['c_t'] == DEFAULT_C_T
        assert params['h'] == self.well_data.thickness
        assert params['L'] == self.well_data.fracture_length
        assert params['skin'] == self.well_data.skin
        assert params['N'] == self.well_data.fractures_count
        assert params['a_L'] == self.well_data.a_l_ratio
    
    def test_get_well_params_custom(self):
        """Тест создания параметров скважины с кастомными значениями"""
        custom_k = 5.0
        custom_mu = 2.0
        params = self.app._get_params(self.well_data, default_k=custom_k, default_mu=custom_mu)
        
        assert params['k'] == custom_k
        assert params['mu'] == custom_mu
        assert params['B'] == DEFAULT_B  # остальные по умолчанию
    
    def test_get_quality_label(self):
        """Тест определения качества интерполяции"""
        # Отличное качество
        assert self.app._get_quality_label(0.005) == 'Отлично'
        assert self.app._get_quality_label(0.005, short=True) == 'отлично'
        
        # Хорошее качество
        assert self.app._get_quality_label(0.05) == 'Хорошо'
        assert self.app._get_quality_label(0.05, short=True) == 'хорошо'
        
        # Удовлетворительное качество
        assert self.app._get_quality_label(0.5) == 'Удовлетворительно'
        assert self.app._get_quality_label(0.5, short=True) == 'удовл.'
    
    def test_create_no_interpolation_report(self):
        """Тест создания отчета о том, что интерполяция не требуется"""
        report = self.app._create_no_interpolation_report()
        
        assert "ИНТЕРПОЛЯЦИЯ НЕ ТРЕБУЕТСЯ" in report
        assert "Данные не содержат пропущенных значений" in report
        assert str(len(self.well_data.pressure)) in report
        assert str(len(self.well_data.flow_rate)) in report
    
    def test_show_info_test_mode(self):
        """Тест что show_info не показывается в test_mode"""
        # В test_mode сообщения не должны показываться
        # Просто проверяем, что метод не падает
        self.app.show_info("Тест", "Сообщение")
        # Если бы был показан диалог, это бы вызвало ошибку в тестовом окружении
    
    def test_show_warning_test_mode(self):
        """Тест что show_warning не показывается в test_mode"""
        self.app.show_warning("Тест", "Предупреждение")
    
    def test_show_error_test_mode(self):
        """Тест что show_error не показывается в test_mode"""
        self.app.show_error("Тест", "Ошибка")


class TestMainIntegration:
    """Интеграционные тесты для main.py"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        # Создаем QApplication для тестов GUI
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
    
    @patch('main.QApplication')
    def test_application_startup(self, mock_qapp):
        """Тест запуска приложения"""
        # Мокаем QApplication
        mock_app = Mock()
        mock_qapp.return_value = mock_app
        
        # Импортируем и создаем приложение
        from main import MyApp
        
        # Создаем экземпляр приложения
        app = MyApp()
        
        # Проверяем, что приложение создалось
        assert app is not None
        assert hasattr(app, 'loaded_data')
        assert hasattr(app, 'current_index')
    


class TestMainIntegration:
    """Интеграционные тесты для main.py"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        if not QApplication.instance():
            self.app_instance = QApplication([])
        else:
            self.app_instance = QApplication.instance()
        
        self.app = MyApp(test_mode=True)
    
    def test_application_startup(self):
        """Тест запуска приложения"""
        app = MyApp(test_mode=True)
        
        # Проверяем, что приложение создалось
        assert app is not None
        assert hasattr(app, 'loaded_data')
        assert hasattr(app, 'current_index')
        assert app.test_mode is True
    
    def test_constants_defined(self):
        """Тест что все константы определены"""
        from main import (WINDOW_WIDTH, WINDOW_HEIGHT, TAB_WIDGET_X, TAB_WIDGET_Y,
                          TAB_WIDGET_WIDTH, TAB_WIDGET_HEIGHT, LEFT_PANEL_MAX_WIDTH,
                          RMSE_EXCELLENT_THRESHOLD, RMSE_GOOD_THRESHOLD,
                          DEFAULT_K, DEFAULT_MU, DEFAULT_B, DEFAULT_PHI, DEFAULT_C_T,
                          DEFAULT_OUTLIER_THRESHOLD, DEFAULT_SAVGOL_WINDOW_LENGTH,
                          DEFAULT_SAVGOL_POLYORDER, N_PARAMETERS_PER_POINT,
                          REPORT_SEPARATOR_LENGTH, TYPE_CURVE_N_POINTS,
                          TYPE_CURVE_TIME_MIN, TYPE_CURVE_TIME_MAX, INTERPOLATION_METHODS)
        
        assert isinstance(WINDOW_WIDTH, int)
        assert isinstance(WINDOW_HEIGHT, int)
        assert isinstance(RMSE_EXCELLENT_THRESHOLD, float)
        assert isinstance(RMSE_GOOD_THRESHOLD, float)
        assert isinstance(DEFAULT_K, float)
        assert isinstance(INTERPOLATION_METHODS, tuple)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
