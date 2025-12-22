from PySide6.QtWidgets import (QLabel, QTableView, QApplication, QMainWindow, QFileDialog, QMessageBox, 
                               QComboBox, QSpinBox, QPushButton, QWidget, QVBoxLayout, 
                               QHBoxLayout, QTabWidget, QTextEdit, QGroupBox, QGridLayout, QDialog,
                               QDialogButtonBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QStandardItemModel, QStandardItem

import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from scipy.interpolate import interp1d
import pickle
import os


from ui import Ui_mainWindow

from helpers.parse_well_data import parse_well_data
from helpers.ml_methods import (detect_outliers)
from helpers.dimensionless_analysis import convert_to_dimensionless_curves, get_dimensionless_series, fit_xy_curve_coefficients
from helpers.dimensionless.filtration import SignalFilters, PhysicsConstraints, compute_snr
from helpers.dimensionless_plotting import plot_dimensionless_grouped
from helpers.dimensionless.filtration.utils import select_filter_method
from helpers.dimensionless_interpolating import DimensionlessCurveInterpolator
from helpers.quadratic_regression_model import QuadraticRegressionModel
from helpers.ml_methods import DimensionlessExtrapolator
from helpers.grp_analysis import analyze_flow_regime, compute_productivity_index, detect_flow_regime_transitions, generate_type_curves, match_type_curves

from schemas.well_data import WellTimeSeries

from helpers.input_test import diag_dimensional
from helpers.ui_setup import (
    setup_interface,
    setup_timeseries_tab,
    setup_grp_tab,
    setup_type_curves_tab,
    setup_results_tab,
    setup_data_tab
)

# Импорт констант из config.py
from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    TAB_WIDGET_X, TAB_WIDGET_Y, TAB_WIDGET_WIDTH, TAB_WIDGET_HEIGHT, LEFT_PANEL_MAX_WIDTH,
    RMSE_EXCELLENT_THRESHOLD, RMSE_GOOD_THRESHOLD,
    DEFAULT_K, DEFAULT_MU, DEFAULT_B, DEFAULT_PHI, DEFAULT_C_T,
    DEFAULT_OUTLIER_THRESHOLD, DEFAULT_SAVGOL_WINDOW_LENGTH, DEFAULT_SAVGOL_POLYORDER,
    N_PARAMETERS_PER_POINT, REPORT_SEPARATOR_LENGTH,
    TYPE_CURVE_N_POINTS, TYPE_CURVE_TIME_MIN, TYPE_CURVE_TIME_MAX,
    INTERPOLATION_METHODS
)



class MyApp(QMainWindow, Ui_mainWindow):
    def __init__(self, test_mode: bool = False) -> None:
        super().__init__()
        self.setupUi(self)
        
        self.test_mode = test_mode  # Флаг для отключения сообщений в тестах
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        # Переименуем существующую кнопку и подключим обработчик
        try:
            self.load_template_button.setText("Загрузить основной файл")
        except Exception:
            pass
        self.load_template_button.clicked.connect(self.load_template)
        self.loaded_data = []  # Список WellTimeSeries объектов
        self.current_index = 0  # Индекс текущей скважины
        self.validation_data = []  # Данные для проверки качества интерполяции
        self.last_interpolated_mask_XY = None  # Маска восстановленных точек для выделения на X-Y
        self.last_interpolated_pressure = None  # Интерполированные значения давления (не изменяют исходные данные)
        self.last_extrapolated_XY = None  # Пара экстраполированных X,Y для отображения
        self.last_extrapolation_result = None  # Результат экстраполяции с метриками качества
        self.last_fitted_XY = None  # Подогнанные X,Y с коэффициентами поправки
        self.last_fit_coefficients = None  # Коэффициенты подгонки (a для X, a_y, b, c для Y)
        self.original_calc_XY = None  # Оригинальные расчётные X,Y (до подгонки)
        self.trained_interpolator = None  # Обученный интерполятор (может быть загружен из файла)
        self.trained_interpolator_path = None  # Путь к загруженной модели
        self.trained_interpolation_model = None  # Обученная модель интерполяции (DimensionlessCurveInterpolator)
        self.trained_interpolation_model_path = None  # Путь к загруженной модели интерполяции
        
        # Создаем  интерфейс с вкладками
        setup_interface(self)
        
        # Настраиваем обработчики событий
        self.setup_event_handlers()
    
    @property
    def current_data(self) -> Optional[WellTimeSeries]:
        """Возвращает данные текущей скважины"""
        if self.loaded_data and 0 <= self.current_index < len(self.loaded_data):
            return self.loaded_data[self.current_index]
        return None
    
    @current_data.setter
    def current_data(self, value: Optional[WellTimeSeries]) -> None:
        """Устанавливает данные текущей скважины"""
        if value is None:
            # Если устанавливаем None, не меняем current_index
            # current_data будет None через property getter, если индекс невалидный
            pass
        else:
            # Ищем индекс в loaded_data
            if self.loaded_data:
                try:
                    index = self.loaded_data.index(value)
                    self.current_index = index
                except ValueError:
                    # Если не найдено, добавляем в список
                    self.loaded_data.append(value)
                    self.current_index = len(self.loaded_data) - 1
            else:
                # Если loaded_data пустой, создаём список и добавляем элемент
                self.loaded_data = [value]
                self.current_index = 0

    def show_info(self, title: str, message: str) -> None:
        """Показывает информационное сообщение (только если не test_mode)"""
        if not self.test_mode:
            QMessageBox.information(self, title, message)
    
    def show_warning(self, title: str, message: str) -> None:
        """Показывает предупреждение (только если не test_mode)"""
        if not self.test_mode:
            QMessageBox.warning(self, title, message)
    
    def show_error(self, title: str, message: str) -> None:
        """Показывает сообщение об ошибке (только если не test_mode)"""
        if not self.test_mode:
            QMessageBox.critical(self, title, message)

    def _get_params(self, data_item: WellTimeSeries, 
                        default_k: float = DEFAULT_K, 
                        default_mu: float = DEFAULT_MU, 
                        default_B: float = DEFAULT_B,
                        default_phi: float = DEFAULT_PHI, 
                        default_c_t: float = DEFAULT_C_T) -> Dict[str, Any]:
        """Создает словарь параметров скважины для безразмерного анализа"""
        return {
            'k': default_k,
            'h': data_item.thickness,
            'mu': default_mu,
            'B': default_B,
            'phi': default_phi,
            'c_t': default_c_t,
            'L': data_item.fracture_length,
            'skin': data_item.skin,
            'N': data_item.fractures_count,
            'a_L': data_item.a_l_ratio,
            'dP': data_item.depression,
        }
    
    def _get_quality_label(self, rmse: float, short: bool = False) -> str:
        """Возвращает текстовое описание качества интерполяции на основе RMSE"""
        if rmse < RMSE_EXCELLENT_THRESHOLD:
            return 'отлично' if short else 'Отлично'
        elif rmse < RMSE_GOOD_THRESHOLD:
            return 'хорошо' if short else 'Хорошо'
        else:
            return 'удовл.' if short else 'Удовлетворительно'
    
    def _create_no_interpolation_report(self) -> str:
        """Создает отчет о том, что интерполяция не требуется"""
        separator = "=" * REPORT_SEPARATOR_LENGTH
        report = separator + "\n"
        report += "ИНТЕРПОЛЯЦИЯ НЕ ТРЕБУЕТСЯ\n"
        report += separator + "\n\n"
        report += "Данные не содержат пропущенных значений (NaN).\n"
        report += f"Точек данных по давлению: {len(self.current_data.pressure)}\n"
        report += f"Точек данных по дебиту: {len(self.current_data.flow_rate)}\n"
        report += "\nВсе данные присутствуют, интерполяция не нужна.\n"
        report += separator + "\n"
        return report
    
    def _detect_gaps_by_distance(self, time: np.ndarray, X: Optional[np.ndarray] = None, 
                                   Y: Optional[np.ndarray] = None, threshold_factor: float = 1) -> np.ndarray:
        """
        Определяет пропуски по расстоянию между точками в пространстве времени и/или X-Y.
        Если расстояние между двумя точками больше, чем threshold_factor * среднее расстояние
        между этими точками и их соседями, то между ними пропуск.
        
        Логика: для каждой пары соседних точек (i, i+1) в отсортированном порядке:
        - Вычисляем расстояние по времени: dt_i = time[i+1] - time[i]
        - Вычисляем расстояние в X-Y пространстве: dXY_i = sqrt((X[i+1]-X[i])² + (Y[i+1]-Y[i])²)
        - Вычисляем расстояния до соседей
        - Если dt_i > threshold_factor * среднее(dt_prev, dt_next) ИЛИ 
          dXY_i > threshold_factor * среднее(dXY_prev, dXY_next), то между i и i+1 пропуск
        
        Args:
            time: Временные метки
            X: Безразмерные X-координаты (опционально)
            Y: Безразмерные Y-координаты (опционально)
            threshold_factor: Множитель для определения пропусков (по умолчанию 1)
        
        Returns:
            Маска пропусков (True где есть пропуск - точки между двумя точками с большим расстоянием)
        """
        if len(time) < 3:
            return np.zeros(len(time), dtype=bool)
        
        # Сортируем по времени для анализа расстояний
        time_argsorted = np.argsort(time)
        time_sorted = time[time_argsorted]
        
        # Вычисляем расстояния между соседними точками в отсортированном порядке
        dt = np.diff(time_sorted)
        
        if len(dt) == 0 or np.all(dt <= 0):
            return np.zeros(len(time), dtype=bool)
        
        # Вычисляем расстояния в X-Y пространстве, если координаты предоставлены
        dXY = None
        if X is not None and Y is not None and len(X) == len(time) and len(Y) == len(time):
            X_sorted = X[time_argsorted]
            Y_sorted = Y[time_argsorted]
            # Евклидово расстояние в X-Y пространстве
            dX = np.diff(X_sorted)
            dY = np.diff(Y_sorted)
            dXY = np.sqrt(dX**2 + dY**2)
        
        # Определяем пропуски: если расстояние между двумя точками больше,
        # чем расстояние между этими точками и их соседями
        gap_mask = np.zeros(len(time), dtype=bool)
        
        for i in range(len(time_sorted) - 1):
            dt_i = time_sorted[i + 1] - time_sorted[i]  # Расстояние по времени между текущей парой
            
            # Вычисляем расстояния до соседей по времени
            dt_prev = dt[i - 1] if i > 0 else dt_i
            dt_next = dt[i + 1] if i + 1 < len(dt) else dt_i
            
            # Среднее расстояние до соседей по времени
            if i > 0 and i + 1 < len(dt):
                neighbor_avg_dt = (dt_prev + dt_next) / 2.0
            else:
                neighbor_avg_dt = max(dt_prev, dt_next) if max(dt_prev, dt_next) > 0 else dt_i
            
            # Проверяем расстояние по времени
            is_gap_by_time = dt_i > threshold_factor * neighbor_avg_dt and neighbor_avg_dt > 0
            
            # Проверяем расстояние в X-Y пространстве, если координаты предоставлены
            is_gap_by_xy = False
            if dXY is not None and len(dXY) > i:
                dXY_i = dXY[i]  # Расстояние в X-Y пространстве между текущей парой
                
                # Вычисляем расстояния до соседей в X-Y пространстве
                dXY_prev = dXY[i - 1] if i > 0 else dXY_i
                dXY_next = dXY[i + 1] if i + 1 < len(dXY) else dXY_i
                
                # Среднее расстояние до соседей в X-Y пространстве
                if i > 0 and i + 1 < len(dXY):
                    neighbor_avg_dXY = (dXY_prev + dXY_next) / 2.0
                else:
                    neighbor_avg_dXY = max(dXY_prev, dXY_next) if max(dXY_prev, dXY_next) > 0 else dXY_i
                
                is_gap_by_xy = dXY_i > threshold_factor * neighbor_avg_dXY and neighbor_avg_dXY > 0
            
            # Если расстояние по времени ИЛИ по X-Y превышает порог, то это пропуск
            if is_gap_by_time or is_gap_by_xy:
                # Это пропуск - помечаем все точки между time_argsorted[i] и time_argsorted[i+1]
                # в исходном порядке как пропуски
                start_orig_idx = time_argsorted[i]
                end_orig_idx = time_argsorted[i + 1]
                
                # Находим все индексы между start и end в исходном порядке
                if start_orig_idx < end_orig_idx:
                    # Помечаем все точки между start и end как пропуски
                    gap_mask[start_orig_idx + 1:end_orig_idx] = True
                else:
                    # Если индексы перепутаны (не должно быть при сортировке, но на всякий случай)
                    gap_mask[end_orig_idx + 1:start_orig_idx] = True
        
        return gap_mask
    
    def _get_Y_at_time(self, dim_data, time_array: np.ndarray, t_query: float) -> float:
        """
        Получает Y-координату для заданного времени t_query.
        
        Args:
            dim_data: Объект с атрибутами Y и t (или time)
            time_array: Массив временных меток
            t_query: Запрос времени
        
        Returns:
            Y-координата для времени t_query
        """
        # Проверяем, есть ли прямое соответствие t->Y
        if hasattr(dim_data, 't') and len(dim_data.t) == len(dim_data.Y):
            from scipy.interpolate import interp1d
            f = interp1d(dim_data.t, dim_data.Y, kind='linear', bounds_error=False, fill_value=np.nan)
            return float(f(t_query))
        elif len(time_array) == len(dim_data.Y):
            # Используем time_array как маппинг
            from scipy.interpolate import interp1d
            # Удаляем NaN для интерполяции
            valid_mask = np.isfinite(time_array) & np.isfinite(dim_data.Y)
            if np.sum(valid_mask) < 2:
                # Fallback: используем ближайшее значение
                idx = np.argmin(np.abs(time_array - t_query))
                return float(dim_data.Y[idx]) if idx < len(dim_data.Y) else np.nan
            
            f = interp1d(time_array[valid_mask], dim_data.Y[valid_mask], 
                       kind='linear', bounds_error=False, fill_value=np.nan)
            result = f(t_query)
            return float(result) if np.isfinite(result) else np.nan
        else:
            # Fallback: используем ближайшее значение по времени
            idx = np.argmin(np.abs(time_array - t_query))
            return float(dim_data.Y[idx]) if idx < len(dim_data.Y) else np.nan
    
    def _detect_neighbor_copy(self, pred_values: np.ndarray, original_values: np.ndarray, 
                             gaps_indices: np.ndarray, eps: float = 1e-8) -> float:
        """
        Определяет, копирует ли предсказание значения соседей.
        
        Returns:
            Доля предсказаний, которые равны соседям (должна быть < 0.1)
        """
        if len(pred_values) == 0 or len(gaps_indices) == 0:
            return 0.0
        
        neighbor_copy_count = 0
        for i, idx in enumerate(gaps_indices):
            if i >= len(pred_values):
                continue
            
            pred_val = pred_values[i]
            if not np.isfinite(pred_val):
                continue
            
            # Проверяем левого соседа
            if idx > 0 and np.isfinite(original_values[idx - 1]):
                if np.abs(pred_val - original_values[idx - 1]) <= eps:
                    neighbor_copy_count += 1
                    continue
            
            # Проверяем правого соседа
            if idx < len(original_values) - 1 and np.isfinite(original_values[idx + 1]):
                if np.abs(pred_val - original_values[idx + 1]) <= eps:
                    neighbor_copy_count += 1
                    continue
        
        return neighbor_copy_count / len(pred_values) if len(pred_values) > 0 else 0.0
    
    def _local_interpolation_Y_space(self, Y_grid: np.ndarray, P_values: np.ndarray, 
                                     Y_target: np.ndarray) -> np.ndarray:
        """
        Локальная интерполяция в Y-пространстве с использованием PCHIP (сохраняет монотонность).
        
        Args:
            Y_grid: Сетка Y-координат (должна быть монотонной)
            P_values: Значения давления на Y_grid
            Y_target: Целевые Y-координаты для интерполяции
        
        Returns:
            Интерполированные значения давления
        """
        from scipy.interpolate import PchipInterpolator, interp1d
        
        # Проверяем упорядоченность
        if not np.all(np.diff(Y_grid) >= 0) and not np.all(np.diff(Y_grid) <= 0):
            # Если не монотонна, сортируем
            sort_idx = np.argsort(Y_grid)
            Y_grid = Y_grid[sort_idx]
            P_values = P_values[sort_idx]
        
        # Удаляем NaN
        valid_mask = np.isfinite(Y_grid) & np.isfinite(P_values)
        if np.sum(valid_mask) < 2:
            # Fallback на линейную интерполяцию
            return np.interp(Y_target, Y_grid[valid_mask], P_values[valid_mask])
        
        Y_valid = Y_grid[valid_mask]
        P_valid = P_values[valid_mask]
        
        # Используем PCHIP для сохранения монотонности
        try:
            pchip = PchipInterpolator(Y_valid, P_valid, extrapolate=False)
            result = pchip(Y_target)
            # Для точек вне диапазона используем clamp
            y_min, y_max = Y_valid.min(), Y_valid.max()
            result[Y_target < y_min] = P_valid[Y_valid == y_min][0] if len(P_valid[Y_valid == y_min]) > 0 else np.nan
            result[Y_target > y_max] = P_valid[Y_valid == y_max][-1] if len(P_valid[Y_valid == y_max]) > 0 else np.nan
            return result
        except Exception:
            # Fallback на линейную интерполяцию
            return np.interp(Y_target, Y_valid, P_valid)
    
    def _find_complete_curves(self) -> List[Tuple[int, WellTimeSeries]]:
        """
        Находит кривые без пропусков в загруженных данных.
        
        Returns:
            Список кортежей (индекс, WellTimeSeries) для кривых без пропусков
        """
        complete_curves = []
        for idx, well_data in enumerate(self.loaded_data):
            # Проверяем наличие пропусков
            has_nan = (well_data.pressure.isna().any() or 
                      well_data.flow_rate.isna().any())
            
            # Проверяем большие расстояния (без X-Y для простоты)
            time_gaps = self._detect_gaps_by_distance(
                well_data.time.values,
                threshold_factor=3.0
            )
            
            if not has_nan and not np.any(time_gaps):
                complete_curves.append((idx, well_data))
        
        return complete_curves
    
    def on_train_interpolator(self) -> None:
        """Обучает квадратичную регрессию для подгонки расчётных X-Y под эталонные X-Y из данных на массиве всех скважин"""
        if not self.loaded_data:
            self.show_info("Ошибка", "Нет загруженных данных для обучения")
            return
        
        try:
            # Собираем все расчётные и эталонные X-Y со всех скважин
            X_calc_all = []
            Y_calc_all = []
            X_data_all = []
            Y_data_all = []
            
            # Находим кривые без пропусков и с эталонными X-Y
            complete_curves = self._find_complete_curves()
            
            for idx, well_data in complete_curves:
                # Проверяем наличие эталонных X-Y в данных
                if not (hasattr(well_data, 'X') and well_data.X is not None and
                        hasattr(well_data, 'Y') and well_data.Y is not None):
                    print(f"Пропущена кривая {idx}: отсутствуют эталонные X-Y в данных")
                    continue
                
                try:
                    params = self._get_params(well_data)
                    
                    # Вычисляем расчётные X-Y по формулам
                    dim_data = convert_to_dimensionless_curves(
                        well_data.time, well_data.pressure, well_data.flow_rate,
                        well_data.depression, params, x_mode='alt'
                    )
                    
                    # Получаем эталонные X-Y из данных
                    X_data = well_data.X.values if hasattr(well_data.X, 'values') else np.array(well_data.X)
                    Y_data = well_data.Y.values if hasattr(well_data.Y, 'values') else np.array(well_data.Y)
                    X_calc = np.asarray(dim_data.X).flatten()
                    Y_calc = np.asarray(dim_data.Y).flatten()
                    
                    # Преобразуем в массивы и убираем скаляры
                    X_data = np.asarray(X_data).flatten()
                    Y_data = np.asarray(Y_data).flatten()
                    
                    # Проверяем, что это массивы, а не скаляры
                    if X_data.ndim == 0 or Y_data.ndim == 0 or X_calc.ndim == 0 or Y_calc.ndim == 0:
                        print(f"Пропущена кривая {idx}: X или Y являются скалярами, а не массивами")
                        continue
                    
                    # Проверяем, что длины совпадают
                    min_len = min(len(X_data), len(Y_data), len(X_calc), len(Y_calc))
                    if min_len < 2:
                        print(f"Пропущена кривая {idx}: недостаточно точек ({min_len})")
                        continue
                    
                    # Обрезаем до минимальной длины
                    X_data = X_data[:min_len]
                    Y_data = Y_data[:min_len]
                    X_calc = X_calc[:min_len]
                    Y_calc = Y_calc[:min_len]
                    
                    # Добавляем в общий массив (преобразуем в список для extend)
                    X_calc_all.extend(X_calc.tolist() if hasattr(X_calc, 'tolist') else list(X_calc))
                    Y_calc_all.extend(Y_calc.tolist() if hasattr(Y_calc, 'tolist') else list(Y_calc))
                    X_data_all.extend(X_data.tolist() if hasattr(X_data, 'tolist') else list(X_data))
                    Y_data_all.extend(Y_data.tolist() if hasattr(Y_data, 'tolist') else list(Y_data))
                    
                except Exception as e:
                    print(f"Пропущена кривая {idx} из-за ошибки: {e}")
                    continue
            
            # Проверяем минимальное количество точек (30)
            if len(X_calc_all) < 30:
                self.show_info("Ошибка", 
                    f"Недостаточно точек для обучения. Найдено: {len(X_calc_all)}, требуется минимум 30.\n"
                    f"Убедитесь, что в данных есть эталонные X и Y для скважин без пропусков.")
                return
            
            # Преобразуем в numpy массивы
            X_calc_all = np.array(X_calc_all)
            Y_calc_all = np.array(Y_calc_all)
            X_data_all = np.array(X_data_all)
            Y_data_all = np.array(Y_data_all)
            
            # Обучаем квадратичную регрессию (подгоняем и X, и Y)
            model = QuadraticRegressionModel(alpha=1.0, fit_only_y=False)
            model.fit(X_calc_all, Y_calc_all, X_data_all, Y_data_all)
            
            self.trained_interpolator = model
            self.trained_interpolator_path = None  # Сбрасываем путь, так как модель переобучена
            
            # Получаем метрики
            metrics = model.get_metrics()
            coef = model.get_coefficients()
            
            # Обновляем статус
            if hasattr(self, 'model_status_label'):
                self.model_status_label.setText(f"Обучена на {metrics['n_samples']} точках")
            
            # Формируем сообщение
            msg = f"Модель обучена на {metrics['n_samples']} точках из {len(complete_curves)} скважин.\n\n"
            msg += f"Коэффициенты:\n"
            msg += f"  X: a = {coef['a']:.6f}\n"
            msg += f"  Y: a_y = {coef['a_y']:.6f}, b = {coef['b']:.6f}, c = {coef['c']:.6f}\n\n"
            msg += f"Метрики:\n"
            msg += f"  X: RMSE = {metrics['rmse_x']:.6e}, R² = {metrics['r2_x']:.4f}\n"
            msg += f"  Y: RMSE = {metrics['rmse_y']:.6e}, R² = {metrics['r2_y']:.4f}"
            
            self.show_info("Успех", msg)
            
        except Exception as e:
            self.show_info("Ошибка", f"Не удалось обучить модель: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def on_save_interpolator(self) -> None:
        """Сохраняет обученный интерполятор в файл"""
        if self.trained_interpolator is None:
            self.show_info("Ошибка", "Нет обученной модели для сохранения")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить модель интерполяции", "",
                "Pickle Files (*.pkl);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Добавляем расширение, если его нет
            if not file_path.endswith('.pkl'):
                file_path += '.pkl'
            
            # Сохраняем модель
            with open(file_path, 'wb') as f:
                pickle.dump(self.trained_interpolator, f)
            
            self.trained_interpolator_path = file_path
            
            # Обновляем статус
            if hasattr(self, 'model_status_label'):
                filename = os.path.basename(file_path)
                self.model_status_label.setText(f"Сохранена: {filename}")
            
            self.show_info("Успех", f"Модель сохранена в файл:\n{file_path}")
            
        except Exception as e:
            self.show_info("Ошибка", f"Не удалось сохранить модель: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def on_load_interpolator(self) -> None:
        """Загружает интерполятор из файла"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Загрузить модель интерполяции", "",
                "Pickle Files (*.pkl);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Загружаем модель
            with open(file_path, 'rb') as f:
                interp = pickle.load(f)
            
            # Проверяем, что это правильный тип
            if not isinstance(interp, (DimensionlessCurveInterpolator, QuadraticRegressionModel)):
                self.show_info("Ошибка", "Загруженный файл не является моделью интерполяции")
                return
            
            if not interp.is_fitted:
                self.show_info("Ошибка", "Загруженная модель не обучена")
                return
            
            self.trained_interpolator = interp
            self.trained_interpolator_path = file_path
            
            # Обновляем статус
            if hasattr(self, 'model_status_label'):
                filename = os.path.basename(file_path)
                self.model_status_label.setText(f"Загружена: {filename}")
            
            # Формируем информацию о модели в зависимости от типа
            if isinstance(interp, QuadraticRegressionModel):
                metrics = interp.get_metrics()
                coef = interp.get_coefficients()
                model_info = f"Модель: Квадратичная регрессия (Ridge)\n"
                model_info += f"Обучена на {metrics['n_samples']} точках\n"
                model_info += f"RMSE Y: {metrics['rmse_y']:.6e}, R² Y: {metrics['r2_y']:.4f}"
            else:
                # DimensionlessCurveInterpolator
                model_info = f"Метод: {interp.best_method}\n"
                model_info += f"Обучающих примеров: {interp.param_grid.shape[0] if hasattr(interp, 'param_grid') else 0}"
            
            self.show_info("Успех", 
                f"Модель загружена из файла:\n{file_path}\n\n{model_info}")
            
        except Exception as e:
            self.show_info("Ошибка", f"Не удалось загрузить модель: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def _perform_interpolation(self) -> Tuple[Dict[str, Any], Any]:
        """
        Выполняет интерполяцию безразмерных кривых с правильным маппингом Y->t.
        Использует два режима: локальный (PCHIP) и параметрический (RBF/GP).
        """
        params = self._get_params(self.current_data)
        time_array = self.current_data.time.values
        
        # Конвертируем в безразмерные параметры
        dim_data = convert_to_dimensionless_curves(
            self.current_data.time, self.current_data.pressure, self.current_data.flow_rate, 
            self.current_data.depression, params, x_mode='alt'
        )
        
        # 1. ПРОВЕРКИ ВХОДОВ
        # Убеждаемся, что dim_data.Y и time_array имеют одинаковую длину и порядок
        if len(dim_data.Y) != len(time_array):
            raise ValueError(f"Несоответствие длин: dim_data.Y={len(dim_data.Y)}, time_array={len(time_array)}")
        
        # 2. ОПРЕДЕЛЕНИЕ ПРОПУСКОВ
        pressure_is_nan = self.current_data.pressure.isna()
        time_gaps = self._detect_gaps_by_distance(
            time_array, X=dim_data.X, Y=dim_data.Y, threshold_factor=3.0
        )
        
        # Объединяем маски
        gaps_mask = pressure_is_nan.values if hasattr(pressure_is_nan, 'values') else pressure_is_nan
        gaps_mask = gaps_mask | time_gaps
        
        # Sanity check: если >30% точек помечены как пропуски, используем более мягкий порог
        gap_fraction = np.sum(gaps_mask) / len(gaps_mask) if len(gaps_mask) > 0 else 0
        if gap_fraction > 0.3:
            # Пересчитываем с более мягким порогом
            time_gaps = self._detect_gaps_by_distance(
                time_array, X=dim_data.X, Y=dim_data.Y, threshold_factor=5.0
            )
            gaps_mask = pressure_is_nan.values | time_gaps
            gap_fraction = np.sum(gaps_mask) / len(gaps_mask)
        
        if not np.any(gaps_mask):
            self.last_interpolated_pressure = self.current_data.pressure.copy()
            return {'method': 'none', 'n_points': 0, 'message': 'Нет пропусков для интерполяции'}, None
        
        gaps_indices = np.where(gaps_mask)[0]
        
        # 3. ОПРЕДЕЛЕНИЕ РЕЖИМА ИНТЕРПОЛЯЦИИ
        # Проверяем, есть ли обученная модель (может быть BinaryCurveModel или DimensionlessCurveInterpolator)
        use_trained_model = (self.trained_interpolator is not None and 
                            hasattr(self.trained_interpolator, 'is_fitted') and 
                            self.trained_interpolator.is_fitted)
        use_trained_interpolation_model = (self.trained_interpolation_model is not None and 
                                          hasattr(self.trained_interpolation_model, 'is_fitted') and 
                                          self.trained_interpolation_model.is_fitted)
        use_local_mode = not (use_trained_model or use_trained_interpolation_model)  # Локальный режим, если нет обученной модели
        
        # 4. ИНТЕРПОЛЯЦИЯ X-Y КРИВОЙ (расчётной)
        pressure_restored = self.current_data.pressure.copy()
        
        # Получаем X и Y координаты для пропусков
        X_targets = dim_data.X[gaps_indices].copy()
        Y_targets = dim_data.Y[gaps_indices].copy()
        valid_XY_mask = np.isfinite(X_targets) & np.isfinite(Y_targets)
        
        # Интерполируем X и Y отдельно
        X_values_gaps = None
        Y_values_gaps = None
        
        if use_trained_model:
            # Используем обученную модель для предсказания X-Y кривой
            try:
                if np.sum(valid_XY_mask) > 0:
                    # Предсказываем X-Y кривую для всех точек Y_grid модели
                    XY_pred = self.trained_interpolator.predict(
                        skin=self.current_data.skin,
                        N=self.current_data.fractures_count,
                        a_L=self.current_data.a_l_ratio
                    )
        
                    # XY_pred должен быть DataFrame с колонками 'X' и 'Y' или Series с MultiIndex
                    if isinstance(XY_pred, pd.DataFrame) and 'X' in XY_pred.columns and 'Y' in XY_pred.columns:
                        Y_grid_model = XY_pred.index.values
                        X_pred_model = XY_pred['X'].values
                        Y_pred_model = XY_pred['Y'].values
                    elif hasattr(XY_pred, 'index'):
                        # Если это Series с X и Y как значениями, нужно извлечь их
                        Y_grid_model = XY_pred.index.values
                        # Предполагаем, что XY_pred содержит X значения, а Y - это индекс
                        X_pred_model = XY_pred.values
                        Y_pred_model = Y_grid_model
                    else:
                        # Fallback на локальную интерполяцию
                        use_local_mode = True
                        X_values_gaps = None
                        Y_values_gaps = None
                    
                    if not use_local_mode:
                        from scipy.interpolate import interp1d
                        # Интерполируем X и Y для пропусков
                        interp_X = interp1d(Y_grid_model, X_pred_model, kind='linear',
                                           bounds_error=False, fill_value=np.nan)
                        interp_Y = interp1d(Y_grid_model, Y_pred_model, kind='linear',
                                           bounds_error=False, fill_value=np.nan)
                        
                        Y_targets_valid = Y_targets[valid_XY_mask]
                        X_values_gaps = interp_X(Y_targets_valid)
                        Y_values_gaps = interp_Y(Y_targets_valid)
                else:
                    use_local_mode = True
                    X_values_gaps = None
                    Y_values_gaps = None
            except Exception as e:
                # Если ошибка при использовании обученной модели, переключаемся на локальный режим
                print(f"Ошибка при использовании обученной модели: {e}")
                use_local_mode = True
                X_values_gaps = None
                Y_values_gaps = None
        elif use_trained_interpolation_model:
            # Используем DimensionlessCurveInterpolator или SimpleRBFInterpolator
            try:
                if np.sum(valid_XY_mask) > 0:
                    # Получаем параметры скважины
                    skin = self.current_data.skin if hasattr(self.current_data, 'skin') else 0.0
                    N = self.current_data.fractures_count if hasattr(self.current_data, 'fractures_count') else 1
                    a_L = self.current_data.a_l_ratio if hasattr(self.current_data, 'a_l_ratio') else 0.1
                    
                    # Предсказываем P_D кривую
                    P_D_pred = self.trained_interpolation_model.predict(skin, N, a_L)
                    
                    if isinstance(P_D_pred, pd.Series) and len(P_D_pred) > 0:
                        # P_D_pred - это Series с индексом Y и значениями P_D
                        Y_grid_model = P_D_pred.index.values
                        P_D_values = P_D_pred.values
                        
                        # Преобразуем P_D в X и Y
                        # X = P_D (безразмерное давление)
                        # Y уже есть в индексе
                        X_pred_model = P_D_values
                        Y_pred_model = Y_grid_model
                        
                        from scipy.interpolate import interp1d
                        # Интерполируем X и Y для пропусков
                        interp_X = interp1d(Y_grid_model, X_pred_model, kind='linear',
                                           bounds_error=False, fill_value=np.nan)
                        interp_Y = interp1d(Y_grid_model, Y_pred_model, kind='linear',
                                           bounds_error=False, fill_value=np.nan)
                        
                        Y_targets_valid = Y_targets[valid_XY_mask]
                        X_values_gaps = interp_X(Y_targets_valid)
                        Y_values_gaps = interp_Y(Y_targets_valid)
                    else:
                        # Fallback на локальную интерполяцию
                        use_local_mode = True
                        X_values_gaps = None
                        Y_values_gaps = None
                else:
                    use_local_mode = True
                    X_values_gaps = None
                    Y_values_gaps = None
            except Exception as e:
                # Если ошибка при использовании модели интерполяции, переключаемся на локальный режим
                print(f"Ошибка при использовании модели интерполяции: {e}")
                use_local_mode = True
                X_values_gaps = None
                Y_values_gaps = None
        
        if use_local_mode or X_values_gaps is None or Y_values_gaps is None:
            # Локальная интерполяция X-Y кривой в Y-пространстве
            # ВАЖНО: Используем только ВАЛИДНЫЕ (не пропущенные) точки для обучения интерполятора
            valid_mask = ~gaps_mask  # Инвертируем маску пропусков
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) < 2:
                # Недостаточно точек для интерполяции
                self.last_interpolated_pressure = pressure_restored
                return {
                    'method': 'none', 
                    'n_points': 0, 
                    'message': f'Недостаточно валидных точек для интерполяции: {len(valid_indices)}'
                }, None
            
            # X, Y для валидных точек (в исходном порядке времени)
            X_valid = dim_data.X[valid_indices]
            Y_valid = dim_data.Y[valid_indices]
            
            if np.sum(valid_XY_mask) > 0:
                # Сортируем валидные точки по Y для интерполяции
                sort_idx_valid = np.argsort(Y_valid)
                Y_valid_sorted = Y_valid[sort_idx_valid]
                X_valid_sorted = X_valid[sort_idx_valid]
                
                Y_targets_valid = Y_targets[valid_XY_mask]
                
                # Интерполируем X и Y отдельно
                X_values_gaps = self._local_interpolation_Y_space(
                    Y_valid_sorted, X_valid_sorted, Y_targets_valid
                )
                Y_values_gaps = Y_targets_valid  # Y уже известен для пропусков
            else:
                X_values_gaps = np.full(np.sum(valid_XY_mask), np.nan)
                Y_values_gaps = np.full(np.sum(valid_XY_mask), np.nan)
        
        # Восстанавливаем давление из интерполированных X-Y
        if X_values_gaps is None or Y_values_gaps is None:
            # Если не удалось получить предсказания, возвращаем исходные данные
            self.last_interpolated_pressure = pressure_restored
            return {
                'method': 'none',
                'n_points': 0,
                'message': 'Не удалось получить предсказания X-Y для пропусков'
            }, None
        
        # Восстанавливаем давление из X-Y используя обратные формулы
        # X = (0.00864 * k * h * Δp) / (μ * B * Q) => Δp = (X * μ * B * Q) / (0.00864 * k * h)
        # Y = (Q * B * t) / (24 * φ * c_t * h * L² * Δp_i) => Δp_i = (Q * B * t) / (24 * φ * c_t * h * L² * Y)
        # Для восстановления давления используем формулу: P = P_i - Δp_i (где P_i - начальное давление)
        
        params = self._get_params(self.current_data)
        k = params.get('k', 1.0)
        h = params.get('h', 10.0)
        mu = params.get('mu', 1.0)
        B = params.get('B', 1.0)
        phi = params.get('phi', 0.1)
        c_t = params.get('c_t', 1e-4)
        L = params.get('L', 100.0)
        
        # Получаем время и дебит для пропусков
        time_gaps = time_array[gaps_indices[valid_XY_mask]]
        flow_rate_gaps = self.current_data.flow_rate.iloc[gaps_indices[valid_XY_mask]].values
        
        # Используем средний дебит, если есть пропуски в дебите
        if np.any(~np.isfinite(flow_rate_gaps)) or np.any(flow_rate_gaps == 0):
            flow_rate_mean = self.current_data.flow_rate[np.isfinite(self.current_data.flow_rate) & (self.current_data.flow_rate > 0)].mean()
            if not np.isfinite(flow_rate_mean) or flow_rate_mean == 0:
                flow_rate_mean = 1.0
            flow_rate_gaps = np.where((np.isfinite(flow_rate_gaps) & (flow_rate_gaps > 0)), 
                                      flow_rate_gaps, flow_rate_mean)
        
        # Восстанавливаем Δp_i из Y: Δp_i = (Q * B * t) / (24 * φ * c_t * h * L² * Y)
        delta_p_i_gaps = (flow_rate_gaps * B * time_gaps) / (24 * phi * c_t * h * L**2 * Y_values_gaps)
        delta_p_i_gaps = np.where(np.isfinite(delta_p_i_gaps) & (delta_p_i_gaps > 0), 
                                 delta_p_i_gaps, dim_data.delta_p_i)
        
        # Восстанавливаем Δp из X: Δp = (X * μ * B * Q) / (0.00864 * k * h)
        delta_p_gaps = (X_values_gaps * mu * B * flow_rate_gaps) / (0.00864 * k * h)
        delta_p_gaps = np.where(np.isfinite(delta_p_gaps) & (delta_p_gaps > 0), 
                               delta_p_gaps, np.nan)
        
        # Восстанавливаем давление: P = P_initial - Δp_i (используем начальное давление из данных)
        P_initial = self.current_data.pressure.iloc[0] if len(self.current_data.pressure) > 0 else 0.0
        if not np.isfinite(P_initial):
            # Если начальное давление неизвестно, используем первое валидное
            valid_pressure = self.current_data.pressure[np.isfinite(self.current_data.pressure)]
            P_initial = valid_pressure.iloc[0] if len(valid_pressure) > 0 else 0.0
        
        pressure_values_gaps_physical = P_initial - delta_p_i_gaps
        
        # 5. БЕЗОПАСНАЯ ВСТАВКА ПРЕДСКАЗАНИЙ
        # ВАЖНО: Маппим предсказания обратно на правильные индексы пропусков
        filled_count = 0
        valid_gap_idx = 0  # Индекс в массиве предсказаний для валидных X-Y
        
        for i, idx_time in enumerate(gaps_indices):
            # Проверяем, есть ли валидное предсказание для этого пропуска
            if valid_XY_mask[i] and valid_gap_idx < len(pressure_values_gaps_physical):
                pred_value = pressure_values_gaps_physical[valid_gap_idx]
                if np.isfinite(pred_value) and pred_value > 0:
                    # ВСТАВЛЯЕМ ТОЧНО В ПОЗИЦИЮ ПРОПУСКА
                    pressure_restored.iloc[idx_time] = pred_value
                    filled_count += 1
                valid_gap_idx += 1
        
        # 6. ЗАЩИТА ОТ КОПИРОВАНИЯ СОСЕДЕЙ
        # Собираем заполненные значения для проверки
        filled_values = []
        filled_indices = []
        for i, idx_time in enumerate(gaps_indices):
            if idx_time < len(pressure_restored) and np.isfinite(pressure_restored.iloc[idx_time]):
                # Проверяем, что это действительно заполненное значение (не исходное)
                if gaps_mask[idx_time]:  # Это был пропуск
                    filled_values.append(pressure_restored.iloc[idx_time])
                    filled_indices.append(idx_time)
        
        if len(filled_values) > 0:
            neighbor_copy_frac = self._detect_neighbor_copy(
                np.array(filled_values),
                self.current_data.pressure.values,
                np.array(filled_indices)
            )
        else:
            neighbor_copy_frac = 0.0
        
        # Если >95% предсказаний равны соседям, переключаемся на альтернативный метод
        if neighbor_copy_frac > 0.95 and len(filled_values) > 0:
            # Используем более сглаживающий метод для X-Y интерполяции
            from scipy.interpolate import UnivariateSpline
            valid_mask = ~gaps_mask
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) >= 3:
                try:
                    X_valid = dim_data.X[valid_indices]
                    Y_valid = dim_data.Y[valid_indices]
                    sort_idx = np.argsort(Y_valid)
                    Y_valid_sorted = Y_valid[sort_idx]
                    X_valid_sorted = X_valid[sort_idx]
                    
                    # Интерполируем X с помощью сплайна
                    spline_X = UnivariateSpline(Y_valid_sorted, X_valid_sorted, s=len(Y_valid_sorted))
                    X_values_gaps_smooth = spline_X(Y_targets[valid_XY_mask])
                    
                    # Восстанавливаем давление из сглаженных X-Y
                    time_gaps_smooth = time_array[gaps_indices[valid_XY_mask]]
                    flow_rate_gaps_smooth = self.current_data.flow_rate.iloc[gaps_indices[valid_XY_mask]].values
                    if np.any(~np.isfinite(flow_rate_gaps_smooth)) or np.any(flow_rate_gaps_smooth == 0):
                        flow_rate_mean = self.current_data.flow_rate[np.isfinite(self.current_data.flow_rate) & (self.current_data.flow_rate > 0)].mean()
                        if not np.isfinite(flow_rate_mean) or flow_rate_mean == 0:
                            flow_rate_mean = 1.0
                        flow_rate_gaps_smooth = np.where((np.isfinite(flow_rate_gaps_smooth) & (flow_rate_gaps_smooth > 0)), 
                                                          flow_rate_gaps_smooth, flow_rate_mean)
                    
                    delta_p_i_gaps_smooth = (flow_rate_gaps_smooth * B * time_gaps_smooth) / (24 * phi * c_t * h * L**2 * Y_values_gaps)
                    delta_p_i_gaps_smooth = np.where(np.isfinite(delta_p_i_gaps_smooth) & (delta_p_i_gaps_smooth > 0), 
                                                     delta_p_i_gaps_smooth, dim_data.delta_p_i)
                    
                    P_initial = self.current_data.pressure.iloc[0] if len(self.current_data.pressure) > 0 else 0.0
                    if not np.isfinite(P_initial):
                        valid_pressure = self.current_data.pressure[np.isfinite(self.current_data.pressure)]
                        P_initial = valid_pressure.iloc[0] if len(valid_pressure) > 0 else 0.0
                    
                    pressure_values_gaps_smooth = P_initial - delta_p_i_gaps_smooth
                    
                    # Перезаписываем только если новые значения отличаются от соседей
                    valid_gap_idx = 0
                    for i, idx_time in enumerate(gaps_indices):
                        if valid_XY_mask[i] and valid_gap_idx < len(pressure_values_gaps_smooth):
                            if np.isfinite(pressure_values_gaps_smooth[valid_gap_idx]) and pressure_values_gaps_smooth[valid_gap_idx] > 0:
                                pressure_restored.iloc[idx_time] = pressure_values_gaps_smooth[valid_gap_idx]
                            valid_gap_idx += 1
                except Exception as e:
                    print(f"Ошибка при сглаживании X-Y интерполяции: {e}")
                    pass  # Оставляем предыдущие значения
        
        # 7. ФИЗИЧЕСКИЕ ОГРАНИЧЕНИЯ И ПОСТ-ОБРАБОТКА
        # Clip: P >= 0
        pressure_restored = pressure_restored.clip(lower=0.0)
        
        # Пересчитываем dP если нужно
        if hasattr(self.current_data, 'depression'):
            depression_restored = self.current_data.depression.copy()
            # dP[i] = P[i] - P[i-1] для измененных точек
            for idx_time in gaps_indices:
                if idx_time > 0 and np.isfinite(pressure_restored.iloc[idx_time]) and np.isfinite(pressure_restored.iloc[idx_time - 1]):
                    depression_restored.iloc[idx_time] = pressure_restored.iloc[idx_time] - pressure_restored.iloc[idx_time - 1]
        
        # Легкое сглаживание на локальном окне для избежания шагов
        if filled_count > 0:
            from scipy.signal import savgol_filter
            try:
                window = min(5, len(pressure_restored) // 10)
                if window >= 3 and window % 2 == 1:
                    smoothed = savgol_filter(pressure_restored.values, window, 2)
                    # Применяем только к заполненным точкам
                    for idx_time in gaps_indices:
                        if idx_time < len(smoothed):
                            # Смешиваем 80% сглаженного + 20% исходного для мягкости
                            pressure_restored.iloc[idx_time] = 0.8 * smoothed[idx_time] + 0.2 * pressure_restored.iloc[idx_time]
            except Exception:
                pass
        
        # 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
        self.last_interpolated_pressure = pressure_restored
        
        # Формируем информацию о результатах
        if use_trained_model and not use_local_mode:
            method_used = 'trained_model'
            if isinstance(self.trained_interpolator, QuadraticRegressionModel):
                # Для QuadraticRegressionModel
                best_method = 'quadratic_regression'
                metrics = self.trained_interpolator.get_metrics()
                rmse_scores = {'quadratic_regression': metrics['rmse_y']}
                n_samples = metrics['n_samples']
            else:
                # Для BinaryCurveModel или другого типа
                best_method = getattr(self.trained_interpolator, 'best_method', 'unknown')
                rmse_scores = getattr(self.trained_interpolator, 'rmse_scores', {})
                n_samples = self.trained_interpolator.param_grid.shape[0] if hasattr(self.trained_interpolator, 'param_grid') else 0
        elif use_trained_interpolation_model and not use_local_mode:
            # Для DimensionlessCurveInterpolator или SimpleRBFInterpolator
            method_used = 'trained_interpolation_model'
            best_method = getattr(self.trained_interpolation_model, 'best_method', 'rbf')
            rmse_scores = getattr(self.trained_interpolation_model, 'rmse_scores', {})
            if hasattr(self.trained_interpolation_model, 'param_grid'):
                n_samples = self.trained_interpolation_model.param_grid.shape[0]
            elif hasattr(self.trained_interpolation_model, 'n_wells'):
                n_samples = self.trained_interpolation_model.n_wells
            else:
                n_samples = 0
        else:
            method_used = 'local_pchip'
            best_method = 'local_pchip'
            rmse_scores = {'local_pchip': 0.0}
            n_samples = 1
        
        interp_info = {
            'method': method_used,
            'best_method': best_method,
            'n_samples': n_samples,
            'n_points': filled_count,  # Количество заполненных точек
            'n_gaps_detected': len(gaps_indices),
            'gap_fraction': gap_fraction,
            'neighbor_copy_fraction': neighbor_copy_frac,
            'message': f'Заполнено {filled_count} из {len(gaps_indices)} пропусков',
            'rmse_scores': rmse_scores
        }
        
        # Если есть эталонные данные, рассчитываем метрики относительно эталона
        if hasattr(self, 'validation_data') and self.validation_data and len(self.validation_data) > 0:
            try:
                ref_item = self.validation_data[0]
                # Проверяем, что это действительно эталонные данные
                if (ref_item is not None and 
                    hasattr(ref_item, 'time') and hasattr(ref_item, 'pressure') and
                    (ref_item is not self.current_data or 
                     len(ref_item.pressure) != len(self.current_data.pressure) or
                     not np.allclose(ref_item.pressure.values, self.current_data.pressure.values, 
                                   rtol=1e-3, equal_nan=True))):
                    
                    # Конвертируем эталон в безразмерные параметры
                    ref_params = self._get_params(ref_item)
                    ref_dim = convert_to_dimensionless_curves(
                        ref_item.time, ref_item.pressure, ref_item.flow_rate, ref_item.depression, ref_params, x_mode='alt'
                    )
                    
                    # Рассчитываем RMSE для заполненных точек
                    if filled_count > 0:
                        # Сравниваем только заполненные точки
                        pred_pressure = pressure_restored.values[gaps_indices[:filled_count]]
                        ref_pressure = ref_item.pressure.values[gaps_indices[:filled_count]]
                        
                        valid_mask = np.isfinite(pred_pressure) & np.isfinite(ref_pressure)
                        if np.sum(valid_mask) > 0:
                            rmse = np.sqrt(np.mean((pred_pressure[valid_mask] - ref_pressure[valid_mask])**2))
                            interp_info['reference_metrics'] = {'rmse': rmse}
            except Exception as e:
                # Если не удалось рассчитать метрики, просто пропускаем
                print(f"Не удалось рассчитать метрики относительно эталона: {e}")
        
        # Сохраняем информацию для отображения в резюме графика
        self.last_interpolation_info = interp_info
        
        return interp_info, dim_data
    
    def _create_interpolation_report(self, n_nan_pressure: int, n_nan_flow: int, 
                                    interp_info: Dict[str, Any]) -> str:
        """Создает отчет о результатах интерполяции"""
        method_names = {
            'linear': 'Линейная регрессия',
            'rbf': 'RBF интерполяция (Thin Plate Spline)',
            'gp': 'Гауссовский процесс'
        }
        
        # Определяем, что было интерполировано
        # Интерполируется безразмерная кривая pD(Y), из которой затем восстанавливаются X и Y
        n_interpolated_points = max(n_nan_pressure, n_nan_flow)  # Количество точек с пропусками
        
        separator = "=" * REPORT_SEPARATOR_LENGTH
        report = separator + "\n"
        report += "РЕЗУЛЬТАТЫ ИНТЕРПОЛЯЦИИ БЕЗРАЗМЕРНЫХ КРИВЫХ\n"
        report += separator + "\n\n"
        
        # Что было интерполировано
        report += "Интерполировано:\n"
        report += "  ✓ Безразмерная кривая pD(Y)\n"
        if n_interpolated_points > 0:
            report += f"  ✓ Восстановлено {n_interpolated_points} точек безразмерной кривой\n"
            report += "  ✓ Из восстановленной кривой рассчитаны X (фильтрационный) и Y (ёмкостной) параметры\n"
        report += "\n"
        
        # Исходные данные
        report += "Исходные данные:\n"
        report += f"  Всего точек: {len(self.current_data.time)}\n"
        if n_interpolated_points > 0:
            coverage = (1 - n_interpolated_points / len(self.current_data.time)) * 100
            report += f"  Полнота данных: {coverage:.1f}%\n"
            report += f"  Восстановлено точек безразмерной кривой: {n_interpolated_points} ({n_interpolated_points/len(self.current_data.time)*100:.1f}%)\n"
        report += "\n"
        
        # Параметры скважины
        report += "Параметры скважины (использованы для интерполяции):\n"
        report += f"  Skin: {self.current_data.skin:.4f}\n"
        report += f"  N (кол-во трещин): {self.current_data.fractures_count}\n"
        report += f"  a/L: {self.current_data.a_l_ratio:.4f}\n"
        report += f"  L (длина трещины): {self.current_data.fracture_length:.2f} м\n"
        report += f"  h (толщина пласта): {self.current_data.thickness:.2f} м\n"
        report += "\n"
        
        # Параметры интерполяции
        report += "Параметры метода:\n"
        if 'n_samples' in interp_info:
            report += f"  Обучающих примеров: {interp_info['n_samples']}\n"
        if 'n_points' in interp_info:
            report += f"  Заполнено точек: {interp_info['n_points']}\n"
        if 'n_gaps_detected' in interp_info:
            report += f"  Обнаружено пропусков: {interp_info['n_gaps_detected']}\n"
        if 'gap_fraction' in interp_info:
            report += f"  Доля пропусков: {interp_info['gap_fraction']*100:.1f}%\n"
        if 'neighbor_copy_fraction' in interp_info:
            report += f"  Доля копирования соседей: {interp_info['neighbor_copy_fraction']*100:.1f}%\n"
        report += "\n"
        
        # Сравнение методов (только если есть несколько методов)
        if 'rmse_scores' in interp_info and len(interp_info['rmse_scores']) > 1:
            report += "СРАВНЕНИЕ МЕТОДОВ ИНТЕРПОЛЯЦИИ\n"
        report += "-" * REPORT_SEPARATOR_LENGTH + "\n"
        report += f"{'Метод':<42} {'RMSE':<12} {'Статус'}\n"
        report += "-" * REPORT_SEPARATOR_LENGTH + "\n"
        
        # Сортируем методы по RMSE
        sorted_methods = sorted(interp_info['rmse_scores'].items(), key=lambda x: x[1])
        
        for i, (method, rmse) in enumerate(sorted_methods, 1):
            method_display = method_names.get(method, method)
            marker = "✓ ВЫБРАН" if method == interp_info.get('best_method') else f"#{i}"
            report += f"{method_display:<42} {rmse:<12.3f} {marker}\n"
        
        report += "-" * REPORT_SEPARATOR_LENGTH + "\n\n"
        
        # Итоговая информация
        best_method = interp_info.get('best_method', interp_info.get('method', 'unknown'))
        best_rmse = interp_info.get('rmse_scores', {}).get(best_method, 0) if 'rmse_scores' in interp_info else 0
        report += "ИТОГОВЫЙ РЕЗУЛЬТАТ:\n"
        report += f"  Метод: {method_names.get(best_method, best_method)}\n"
        if best_rmse > 0:
            report += f"  RMSE: {best_rmse:.6e}\n"
        quality = self._get_quality_label(best_rmse)
        report += f"  Качество: {quality}\n"
        if 'message' in interp_info:
            report += f"  {interp_info['message']}\n"
        
        # Если есть метрики относительно эталона, добавляем их
        if 'reference_metrics' in interp_info:
            ref_metrics = interp_info['reference_metrics']
            report += "\n"
            report += "МЕТРИКИ ОТНОСИТЕЛЬНО ЭТАЛОНА (на случайных точках):\n"
            report += "-" * REPORT_SEPARATOR_LENGTH + "\n"
            report += f"  RMSE: {ref_metrics.get('rmse', 0):.6e}\n"
            report += f"  MAE: {ref_metrics.get('mae', 0):.6e}\n"
            report += f"  MAPE: {ref_metrics.get('mape', 0):.2f}%\n"
            report += f"  R²: {ref_metrics.get('r2', 0):.6f}\n"
            report += f"  Максимальная ошибка: {ref_metrics.get('max_error', 0):.6e}\n"
            report += f"  Медианная ошибка: {ref_metrics.get('median_error', 0):.6e}\n"
            report += f"  Средняя ошибка: {ref_metrics.get('mean_error', 0):.6e}\n"
            report += f"  MSE: {ref_metrics.get('mse', 0):.6e}\n"
        
        # Дополнительные метрики качества
        report += "\nОЦЕНКА КАЧЕСТВА ИНТЕРПОЛЯЦИИ:\n"
        report += "-" * REPORT_SEPARATOR_LENGTH + "\n"
        if 'n_samples' in interp_info:
            report += f"  Обучающих примеров: {interp_info['n_samples']}\n"
        if 'n_points' in interp_info:
            report += f"  Точек на кривой: {interp_info['n_points']}\n"
        if 'best_method' in interp_info:
            report += f"  Выбранный метод: {method_names.get(interp_info['best_method'], interp_info['best_method'])}\n"
        
        # Стабильность интерполяции
        if len(interp_info['rmse_scores']) > 1:
            rmse_values = list(interp_info['rmse_scores'].values())
            rmse_std = np.std(rmse_values) if rmse_values else 0
            rmse_mean = np.mean(rmse_values) if rmse_values else 0
            report += f"  Средний RMSE по всем методам: {rmse_mean:.6e}\n"
            report += f"  Стандартное отклонение RMSE: {rmse_std:.6e}\n"
            if rmse_std > 0:
                stability = "высокая" if rmse_std / rmse_mean < 0.1 else "средняя" if rmse_std / rmse_mean < 0.3 else "низкая"
                report += f"  Стабильность: {stability}\n"
        
        report += "\n" + separator + "\n"
        
        return report
    
    def _create_extrapolation_report(self, result: Dict[str, Any]) -> str:
        """Создает отчет о результатах экстраполяции"""
        separator = "=" * REPORT_SEPARATOR_LENGTH
        report = separator + "\n"
        report += "РЕЗУЛЬТАТЫ ЭКСТРАПОЛЯЦИИ БЕЗРАЗМЕРНЫХ КРИВЫХ X-Y\n"
        report += separator + "\n\n"
        
        # Метаданные
        meta = result.get('meta', {})
        df_pred = result.get('df_pred', pd.DataFrame())
        df_ref = result.get('df_ref', None)
        rmse = result.get('rmse', {})
        
        # Основная информация
        report += "ПАРАМЕТРЫ ЭКСТРАПОЛЯЦИИ:\n"
        report += f"  Метод: {meta.get('method', 'unknown')}\n"
        report += f"  Исторических точек: {meta.get('n_historical_points', 0)}\n"
        report += f"  Экстраполированных точек: {meta.get('n_future_points', 0)}\n"
        report += f"  Шаг времени (dt): {meta.get('dt', 0):.4f} ч\n"
        
        if 'time_range_historical' in meta:
            t_min, t_max = meta['time_range_historical']
            report += f"  Диапазон исторических данных: {t_min:.2f} - {t_max:.2f} ч\n"
        
        if 'time_range_predicted' in meta:
            t_min, t_max = meta['time_range_predicted']
            report += f"  Диапазон экстраполированных данных: {t_min:.2f} - {t_max:.2f} ч\n"
        
        report += "\n"
        
        # Train/validation split информация
        if 'train_split' in meta:
            train_split = meta['train_split']
            n_train = meta.get('n_train_points', 0)
            n_val = meta.get('n_val_points', 0)
            split_pct = int(train_split * 100)
            report += f"  Разделение данных: {split_pct}% train / {100-split_pct}% validation\n"
            report += f"  Точки train: {n_train}, validation: {n_val}\n"
            report += "\n"
        
        # Оценка качества на validation window
        validation_metrics = result.get('validation_metrics', {})
        if validation_metrics:
            report += "ОЦЕНКА КАЧЕСТВА НА VALIDATION WINDOW:\n"
            report += "-" * REPORT_SEPARATOR_LENGTH + "\n"
            
            if 'RMSE_X_val' in validation_metrics:
                report += f"  RMSE_X_val: {validation_metrics['RMSE_X_val']:.6e}\n"
                report += f"  RMSE_Y_val: {validation_metrics['RMSE_Y_val']:.6e}\n"
                report += f"  RMSE_mean: {validation_metrics['RMSE_mean']:.6e}\n"
            
            if 'MAE_X_val' in validation_metrics:
                report += f"  MAE_X_val: {validation_metrics['MAE_X_val']:.6e}\n"
                report += f"  MAE_Y_val: {validation_metrics['MAE_Y_val']:.6e}\n"
            
            if 'MAPE_mean' in validation_metrics:
                report += f"  MAPE_mean: {validation_metrics['MAPE_mean']:.2f}%\n"
            
            if 'R2_mean' in validation_metrics:
                report += f"  R²_mean: {validation_metrics['R2_mean']:.4f}\n"
            
            if 'Max_error' in validation_metrics:
                report += f"  Max_error: {validation_metrics['Max_error']:.6e}\n"
            
            report += "\n"
        
        # Оценка качества экстраполяции (обратная совместимость)
        report += "ОЦЕНКА КАЧЕСТВА ЭКСТРАПОЛЯЦИИ:\n"
        report += "-" * REPORT_SEPARATOR_LENGTH + "\n"
        
        if rmse:
            if isinstance(rmse, dict):
                rmse_x = rmse.get('X', 0)
                rmse_y = rmse.get('Y', 0)
                rmse_mean = rmse.get('mean', 0)
                
                report += f"  RMSE по X: {rmse_x:.6e}\n"
                report += f"  RMSE по Y: {rmse_y:.6e}\n"
                report += f"  Средний RMSE: {rmse_mean:.6e}\n"
                
                # Оценка качества
                quality_x = self._get_quality_label(rmse_x)
                quality_y = self._get_quality_label(rmse_y)
                quality_mean = self._get_quality_label(rmse_mean)
                
                report += f"  Качество по X: {quality_x}\n"
                report += f"  Качество по Y: {quality_y}\n"
                report += f"  Общее качество: {quality_mean}\n"
            else:
                report += f"  RMSE: {rmse:.6e}\n"
                quality = self._get_quality_label(rmse)
                report += f"  Качество: {quality}\n"
        else:
            report += "  RMSE не рассчитан (нет эталонных данных)\n"
        
        # Оценка физичности хвоста (ERI)
        tail_metrics = result.get('tail_metrics', {})
        eri = result.get('ERI', None)
        
        if tail_metrics or eri is not None:
            report += "\nОЦЕНКА ФИЗИЧНОСТИ ХВОСТА (ERI):\n"
            report += "-" * REPORT_SEPARATOR_LENGTH + "\n"
            
            if eri is not None:
                report += f"  ERI (Extrapolation Reliability Index): {eri:.4f}\n"
                if eri >= 0.8:
                    eri_quality = "отличная"
                elif eri >= 0.6:
                    eri_quality = "хорошая"
                elif eri >= 0.4:
                    eri_quality = "удовлетворительная"
                else:
                    eri_quality = "требует улучшения"
                report += f"  Качество экстраполяции: {eri_quality}\n"
                report += "\n"
            
            if tail_metrics:
                report += "  Компоненты ERI:\n"
                report += f"    Stability score: {tail_metrics.get('stability_score', 0):.4f}\n"
                report += f"    Physics slope score: {tail_metrics.get('physics_slope_score', 0):.4f}\n"
                report += f"    Physics curvature score: {tail_metrics.get('physics_curvature_score', 0):.4f}\n"
                report += f"    Mass balance score: {tail_metrics.get('mass_balance_score', 0):.4f}\n"
                report += f"    Smoothness score: {tail_metrics.get('smoothness_score', 0):.4f}\n"
                report += f"    Lipschitz score: {tail_metrics.get('lipschitz_score', 0):.4f}\n"
                report += f"    Ensemble score: {tail_metrics.get('ensemble_score', 0):.4f}\n"
        
        # Стабильность (обратная совместимость)
        if 'stability' in meta:
            stability = meta['stability']
            stability_ru = "хорошая" if stability == 'good' else "требует улучшения"
            report += f"\n  Стабильность (legacy): {stability_ru}\n"
        
        # Статистика по экстраполированным данным
        if not df_pred.empty and 'X' in df_pred.columns and 'Y' in df_pred.columns:
            report += "\nСТАТИСТИКА ЭКСТРАПОЛИРОВАННЫХ ДАННЫХ:\n"
            report += "-" * REPORT_SEPARATOR_LENGTH + "\n"
            
            X_values = df_pred['X'].dropna()
            Y_values = df_pred['Y'].dropna()
            
            if len(X_values) > 0:
                report += f"  X: min={X_values.min():.6e}, max={X_values.max():.6e}, mean={X_values.mean():.6e}\n"
            if len(Y_values) > 0:
                report += f"  Y: min={Y_values.min():.6e}, max={Y_values.max():.6e}, mean={Y_values.mean():.6e}\n"
        
        # Сравнение с эталонными данными (если есть)
        if df_ref is not None and not df_ref.empty and not df_pred.empty:
            report += "\nСРАВНЕНИЕ С ЭТАЛОННЫМИ ДАННЫМИ:\n"
            report += "-" * REPORT_SEPARATOR_LENGTH + "\n"
            
            if 'X' in df_ref.columns and 'Y' in df_ref.columns:
                # Находим общие временные точки
                common_times = np.intersect1d(df_ref['t'].values, df_pred['t'].values)
                if len(common_times) > 0:
                    ref_common = df_ref[df_ref['t'].isin(common_times)].sort_values('t')
                    pred_common = df_pred[df_pred['t'].isin(common_times)].sort_values('t')
                    
                    X_diff = np.abs(ref_common['X'].values - pred_common['X'].values)
                    Y_diff = np.abs(ref_common['Y'].values - pred_common['Y'].values)
                    
                    report += f"  Средняя абсолютная ошибка по X: {np.mean(X_diff):.6e}\n"
                    report += f"  Средняя абсолютная ошибка по Y: {np.mean(Y_diff):.6e}\n"
                    report += f"  Максимальная ошибка по X: {np.max(X_diff):.6e}\n"
                    report += f"  Максимальная ошибка по Y: {np.max(Y_diff):.6e}\n"
        
        report += "\n" + separator + "\n"
        
        return report
    
    def on_train_interpolator_model(self) -> None:
        """Обучает модель интерполяции DimensionlessCurveInterpolator на массиве всех скважин"""
        if not self.loaded_data:
            self.show_info("Ошибка", "Нет загруженных данных для обучения")
            return
        
        try:
            # Собираем данные для обучения интерполятора
            param_grid = []  # Параметры (skin, N, a/L) для каждой скважины
            P_curves_list = []  # Кривые давления (pD) для каждой скважины
            Y_grids_list = []  # Сетки Y для каждой скважины
            
            for idx, well_data in enumerate(self.loaded_data):
                # Проверяем тип данных
                if not isinstance(well_data, WellTimeSeries):
                    print(f"Пропущена скважина {idx}: неверный тип данных (ожидается WellTimeSeries, получен {type(well_data)})")
                    continue
                try:
                    # Проверяем, что well_data - это WellTimeSeries
                    if not hasattr(well_data, 'pressure') or not hasattr(well_data, 'time'):
                        print(f"Пропущена скважина {idx}: отсутствуют необходимые атрибуты")
                        continue
                    
                    # Проверяем, что это не строка или другой неподходящий тип
                    if isinstance(well_data, str) or not isinstance(well_data, WellTimeSeries):
                        print(f"Пропущена скважина {idx}: неверный тип данных")
                        continue
                    
                    params = self._get_params(well_data)
                    
                    # Правильно извлекаем данные - проверяем, что это Series или массив
                    if hasattr(well_data.time, 'values'):
                        time_series = well_data.time.values
                    elif isinstance(well_data.time, (list, tuple, np.ndarray)):
                        time_series = np.array(well_data.time)
                    else:
                        print(f"Пропущена скважина {idx}: неверный тип time")
                        continue
                    
                    if hasattr(well_data.pressure, 'values'):
                        pressure_series = well_data.pressure.values
                    elif isinstance(well_data.pressure, (list, tuple, np.ndarray)):
                        pressure_series = np.array(well_data.pressure)
                    else:
                        print(f"Пропущена скважина {idx}: неверный тип pressure")
                        continue
                    
                    if hasattr(well_data.flow_rate, 'values'):
                        flow_rate_series = well_data.flow_rate.values
                    elif isinstance(well_data.flow_rate, (list, tuple, np.ndarray)):
                        flow_rate_series = np.array(well_data.flow_rate)
                    else:
                        print(f"Пропущена скважина {idx}: неверный тип flow_rate")
                        continue
                    
                    if well_data.depression is not None:
                        if hasattr(well_data.depression, 'values'):
                            depression_series = well_data.depression.values
                        elif isinstance(well_data.depression, (list, tuple, np.ndarray)):
                            depression_series = np.array(well_data.depression)
                        else:
                            depression_series = None
                    else:
                        depression_series = None
                    
                    # Преобразуем в Series для совместимости
                    import pandas as pd
                    time_pd = pd.Series(time_series)
                    pressure_pd = pd.Series(pressure_series)
                    flow_rate_pd = pd.Series(flow_rate_series)
                    depression_pd = pd.Series(depression_series) if depression_series is not None else None
                    
                    # Вычисляем безразмерные параметры
                    dim_data = convert_to_dimensionless_curves(
                        time_pd, pressure_pd, flow_rate_pd,
                        depression_pd, params, x_mode='alt'
                    )
                    
                    # Вычисляем pD (безразмерное давление)
                    pressure_values = np.asarray(pressure_series).flatten()
                    
                    if len(pressure_values) == 0:
                        print(f"Пропущена скважина {idx}: пустой массив давления")
                        continue
                    
                    delta_p_i = float(np.max(pressure_values)) - float(np.min(pressure_values))
                    if delta_p_i == 0 or not np.isfinite(delta_p_i):
                        delta_p_i = 1.0
                    
                    pD = pressure_values / delta_p_i
                    Y = dim_data.Y
                    
                    # Убираем NaN и Inf
                    valid_mask = np.isfinite(pD) & np.isfinite(Y) & (Y > 0)
                    if np.sum(valid_mask) < 10:  # Минимум 10 точек
                        continue
                    
                    pD_valid = pD[valid_mask]
                    Y_valid = Y[valid_mask]
                    
                    # Сортируем по Y
                    sort_idx = np.argsort(Y_valid)
                    Y_sorted = Y_valid[sort_idx]
                    pD_sorted = pD_valid[sort_idx]
                    
                    # Добавляем параметры скважины (проверяем наличие атрибутов)
                    if not (hasattr(well_data, 'skin') and hasattr(well_data, 'fractures_count') and hasattr(well_data, 'a_l_ratio')):
                        print(f"Пропущена скважина {idx}: отсутствуют параметры скважины")
                        continue
                    
                    skin_val = float(well_data.skin) if hasattr(well_data, 'skin') else 0.0
                    fractures_val = int(well_data.fractures_count) if hasattr(well_data, 'fractures_count') else 1
                    a_l_val = float(well_data.a_l_ratio) if hasattr(well_data, 'a_l_ratio') else 0.5
                    
                    param_grid.append([skin_val, fractures_val, a_l_val])
                    P_curves_list.append(pD_sorted)
                    Y_grids_list.append(Y_sorted)
                    
                except Exception as e:
                    print(f"Пропущена скважина {idx} из-за ошибки: {e}")
                    continue
            
            if len(param_grid) < 1:
                self.show_info("Ошибка", 
                    f"Недостаточно скважин для обучения. Найдено: {len(param_grid)}, требуется минимум 1.")
                return
            
            # Создаём общую сетку Y (объединяем все Y и берём уникальные значения)
            all_Y = np.concatenate(Y_grids_list)
            Y_grid = np.unique(np.sort(all_Y))
            
            # Если сетка слишком большая, делаем её более разреженной
            if len(Y_grid) > 1000:
                # Берём логарифмически равномерную сетку
                log_Y_min = np.log10(np.min(Y_grid[Y_grid > 0]))
                log_Y_max = np.log10(np.max(Y_grid))
                Y_grid = np.logspace(log_Y_min, log_Y_max, 500)
            
            # Интерполируем кривые pD на общую сетку Y
            P_curves_interp = []
            from scipy.interpolate import interp1d
            
            for i, (Y_orig, P_orig) in enumerate(zip(Y_grids_list, P_curves_list)):
                try:
                    # Интерполируем на общую сетку
                    interp_func = interp1d(Y_orig, P_orig, kind='linear', 
                                         bounds_error=False, fill_value=np.nan)
                    P_interp = interp_func(Y_grid)
                    P_curves_interp.append(P_interp)
                except Exception as e:
                    print(f"Ошибка интерполяции для скважины {i}: {e}")
                    # Используем NaN для этой кривой
                    P_curves_interp.append(np.full(len(Y_grid), np.nan))
            
            # Преобразуем в numpy массивы
            param_grid = np.array(param_grid)  # Форма: (n_wells, 3)
            P_curves = np.array(P_curves_interp)  # Форма: (n_wells, n_points)
            
            n_wells = len(param_grid)
            
            # Ветвление: если меньше 10 скважин, используем упрощённый RBF
            if n_wells < 10:
                # Обучаем RBF напрямую на имеющихся скважинах
                from scipy.interpolate import RBFInterpolator
                
                # Создаём упрощённую модель для малого количества скважин
                class SimpleRBFInterpolator:
                    """Упрощённый RBF интерполятор для малого количества скважин"""
                    def __init__(self, param_grid, Y_grid, P_curves):
                        self.param_grid = param_grid
                        self.Y_grid = Y_grid
                        self.P_curves = P_curves
                        self.rbf_models = []  # RBF модели для каждой точки Y
                        self.is_fitted = True
                        self.best_method = "rbf"
                        self.n_wells = len(param_grid)
                        
                        # Специальный случай: одна скважина - просто возвращаем её кривую
                        if self.n_wells == 1:
                            # Сохраняем кривую единственной скважины
                            self.single_curve = P_curves[0, :]
                            return
                        
                        # Обучаем RBF для каждой точки Y_grid
                        for i in range(len(Y_grid)):
                            y_values = P_curves[:, i]
                            valid_mask = ~np.isnan(y_values)
                            
                            if np.sum(valid_mask) < 1:
                                # Нет данных для этой точки
                                self.rbf_models.append(None)
                                continue
                            
                            param_valid = param_grid[valid_mask]
                            y_values_valid = y_values[valid_mask]
                            
                            # Если только одна скважина с данными, просто возвращаем её значение
                            if len(param_valid) == 1:
                                self.rbf_models.append(y_values_valid[0])  # Сохраняем значение
                                continue
                            
                            try:
                                # Пытаемся обучить RBF (нужно минимум 2 точки)
                                if len(param_valid) >= 2:
                                    # Для 2-3 точек используем линейную регрессию
                                    if len(param_valid) < 4:
                                        from sklearn.linear_model import LinearRegression
                                        lr_model = LinearRegression().fit(param_valid, y_values_valid)
                                        self.rbf_models.append(lr_model)
                                    else:
                                        # Для 4+ точек используем RBF
                                        rbf_model = RBFInterpolator(param_valid, y_values_valid, kernel='thin_plate_spline')
                                        self.rbf_models.append(rbf_model)
                                else:
                                    # Fallback: используем среднее значение
                                    self.rbf_models.append(np.mean(y_values_valid))
                            except Exception:
                                # Fallback на линейную регрессию при ошибке
                                try:
                                    from sklearn.linear_model import LinearRegression
                                    lr_model = LinearRegression().fit(param_valid, y_values_valid)
                                    self.rbf_models.append(lr_model)
                                except Exception:
                                    # Если и это не работает, используем среднее
                                    self.rbf_models.append(np.mean(y_values_valid))
                    
                    def predict(self, skin, N, a_L):
                        """Предсказание для заданных параметров"""
                        # Специальный случай: одна скважина - возвращаем её кривую
                        if self.n_wells == 1:
                            return pd.Series(self.single_curve, index=self.Y_grid, 
                                            name=f"P_D(s={skin}, N={N}, a/L={a_L})")
                        
                        param = np.array([[skin, N, a_L]])
                        predictions = []
                        
                        for i, model in enumerate(self.rbf_models):
                            if model is None:
                                predictions.append(np.nan)
                            elif isinstance(model, (int, float, np.number)):
                                # Просто значение (для случая с одной скважиной)
                                predictions.append(float(model))
                            elif isinstance(model, RBFInterpolator):
                                try:
                                    pred = model(param)[0]
                                    predictions.append(pred)
                                except Exception:
                                    predictions.append(np.nan)
                            else:
                                # LinearRegression или другой sklearn модель
                                try:
                                    pred = model.predict(param)[0]
                                    predictions.append(pred)
                                except Exception:
                                    predictions.append(np.nan)
                        
                        return pd.Series(predictions, index=self.Y_grid, 
                                        name=f"P_D(s={skin}, N={N}, a/L={a_L})")
                
                model = SimpleRBFInterpolator(param_grid, Y_grid, P_curves)
                method_info = f"RBF (упрощённый режим для {n_wells} скважин)"
            else:
                # Обычный режим: используем DimensionlessCurveInterpolator
                model = DimensionlessCurveInterpolator(methods=("linear", "rbf"))
                model.fit(param_grid, Y_grid, P_curves)
                method_info = f"Лучший метод: {model.best_method}"
            
            self.trained_interpolation_model = model
            self.trained_interpolation_model_path = None  # Сбрасываем путь
            
            # Обновляем статус
            if hasattr(self, 'interpolar_model_status_label'):
                n_points = len(Y_grid)
                self.interpolar_model_status_label.setText(
                    f"Обучена на {n_wells} скважинах, {n_points} точках"
                )
            
            # Формируем сообщение
            msg = f"Модель интерполяции обучена на {n_wells} скважинах.\n"
            msg += f"Сетка Y: {len(Y_grid)} точек\n"
            msg += f"{method_info}\n"
            if hasattr(model, 'rmse_scores') and model.rmse_scores:
                msg += f"RMSE: {model.rmse_scores.get(model.best_method, 'N/A')}"
            
            self.show_info("Успех", msg)
            
        except Exception as e:
            self.show_info("Ошибка", f"Не удалось обучить модель интерполяции: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def on_save_interpolator_model(self) -> None:
        """Сохраняет обученную модель интерполяции в файл"""
        if self.trained_interpolation_model is None:
            self.show_info("Ошибка", "Нет обученной модели интерполяции для сохранения")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить модель интерполяции", "",
                "Pickle Files (*.pkl);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Добавляем расширение, если его нет
            if not file_path.endswith('.pkl'):
                file_path += '.pkl'
            
            # Сохраняем модель
            with open(file_path, 'wb') as f:
                pickle.dump(self.trained_interpolation_model, f)
            
            self.trained_interpolation_model_path = file_path
            
            # Обновляем статус
            if hasattr(self, 'interpolar_model_status_label'):
                filename = os.path.basename(file_path)
                self.interpolar_model_status_label.setText(f"Сохранена: {filename}")
            
            self.show_info("Успех", f"Модель интерполяции сохранена в файл:\n{file_path}")
            
        except Exception as e:
            self.show_info("Ошибка", f"Не удалось сохранить модель интерполяции: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def on_load_interpolator_model(self) -> None:
        """Загружает модель интерполяции из файла"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Загрузить модель интерполяции", "",
                "Pickle Files (*.pkl);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Загружаем модель
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            
            # Проверяем, что это правильный тип
            # Может быть DimensionlessCurveInterpolator или SimpleRBFInterpolator
            if not (isinstance(model, DimensionlessCurveInterpolator) or 
                    (hasattr(model, 'is_fitted') and hasattr(model, 'predict') and hasattr(model, 'best_method'))):
                self.show_info("Ошибка", "Загруженный файл не является моделью интерполяции")
                return
            
            if not model.is_fitted:
                self.show_info("Ошибка", "Загруженная модель не обучена")
                return
            
            self.trained_interpolation_model = model
            self.trained_interpolation_model_path = file_path
            
            # Обновляем статус
            if hasattr(self, 'interpolar_model_status_label'):
                filename = os.path.basename(file_path)
                self.interpolar_model_status_label.setText(f"Загружена: {filename}")
            
            # Формируем информацию о модели
            model_info = f"Метод: {model.best_method}\n"
            if hasattr(model, 'param_grid') and model.param_grid is not None:
                model_info += f"Обучающих примеров: {model.param_grid.shape[0]}\n"
            if hasattr(model, 'Y_grid') and model.Y_grid is not None:
                model_info += f"Точек в сетке Y: {len(model.Y_grid)}"
            
            self.show_info("Успех", 
                f"Модель интерполяции загружена из файла:\n{file_path}\n\n{model_info}")
            
        except Exception as e:
            self.show_info("Ошибка", f"Не удалось загрузить модель интерполяции: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
    def setup_event_handlers(self) -> None:
        """Настройка обработчиков событий"""
        # Временные ряды
        self.plot_btn.clicked.connect(self.on_plot_dimensionless_selected)
        self.interp_btn.clicked.connect(self.on_interpolate_data)
        self.ml_filter_btn.clicked.connect(self.on_ml_filter)
        self.outlier_btn.clicked.connect(self.on_detect_outliers)
        self.export_btn.clicked.connect(self.on_export_data)
        self.load_validation_button.clicked.connect(self.load_validation_file)
        self.extrapolate_btn.clicked.connect(self.on_extrapolate_xy)
        self.fit_xy_btn.clicked.connect(self.on_fit_xy_curve)
        
        # Управление моделью апроксимации (оригинальный кусок)
        self.approx_train_model_btn.clicked.connect(self.on_train_interpolator)
        self.approx_save_model_btn.clicked.connect(self.on_save_interpolator)
        self.approx_load_model_btn.clicked.connect(self.on_load_interpolator)
        
        # Управление моделью интерполяции
        self.interpolar_train_model_btn.clicked.connect(self.on_train_interpolator_model)
        self.interpolar_save_model_btn.clicked.connect(self.on_save_interpolator_model)
        self.interpolar_load_model_btn.clicked.connect(self.on_load_interpolator_model)
        
        
        # Кнопка сброса графиков
        self.reset_plots_btn.clicked.connect(self.on_reset_plots)
        
        # обновление данных при смене 
        self.well_combo_dim.currentIndexChanged.connect(self.on_well_changed)
        
        # Привязываем чекбоксы к перестройке графика
        self.cb_dim_pD.stateChanged.connect(self.on_checkbox_toggled)
        self.cb_dim_dpD.stateChanged.connect(self.on_checkbox_toggled)
        self.cb_dim_tD.stateChanged.connect(self.on_checkbox_toggled)
        self.cb_dim_CD.stateChanged.connect(self.on_checkbox_toggled)
        self.cb_XY_plot.stateChanged.connect(self.on_checkbox_toggled)
        self.cb_calc_XY.stateChanged.connect(self.on_checkbox_toggled)
        self.cb_type_gry.stateChanged.connect(self.on_checkbox_toggled)
        self.cb_type_cinco.stateChanged.connect(self.on_checkbox_toggled)
        self.cb_type_valko.stateChanged.connect(self.on_checkbox_toggled)
        self.cb_gfunc.stateChanged.connect(self.on_checkbox_toggled)
        self.cb_mbt.stateChanged.connect(self.on_checkbox_toggled)
        self.cb_real_p.stateChanged.connect(self.on_checkbox_toggled)
        self.cb_real_q.stateChanged.connect(self.on_checkbox_toggled)
        
        # Анализ ГРП
        self.flow_regime_btn.clicked.connect(self.on_analyze_flow_regime)
        self.productivity_btn.clicked.connect(self.on_compute_productivity_index)
        self.transitions_btn.clicked.connect(self.on_detect_flow_regime_transitions)
        
        # Эталонные кривые
        self.bilinear_btn.clicked.connect(self.on_plot_bilinear)
        self.linear_btn.clicked.connect(self.on_plot_linear)
        self.pseudoradial_btn.clicked.connect(self.on_plot_pseudoradial)
        self.match_curves_btn.clicked.connect(self.on_match_type_curves)
        
        # Результаты
        self.export_report_btn.clicked.connect(self.on_export_report)

        self.data_tab = setup_data_tab(self)
        self.tab_widget.addTab(self.data_tab, "Загруженные данные")

    def _load_data_file(self, target: str = 'main') -> None:
        """
        Общий метод загрузки файла данных.
        
        Args:
            target: 'main' - загрузить в loaded_data, 'validation' - загрузить в validation_data
        """
        dialog_title = "Загрузить файл данных" if target == 'main' else "Загрузить файл для проверки"
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, dialog_title, "./", 
                                                 "Data Files (*.csv *.parquet);;CSV Files (*.csv);;Parquet Files (*.parquet);;All Files (*)")
        if not file_path:
            return

        # Парсим данные скважины
        try:
            data, error_msg = parse_well_data(file_path)
            if error_msg is not None:
                self.show_warning("Ошибка загрузки", error_msg)
                return
        except Exception as e:
            self.show_warning("Ошибка загрузки", f"Неожиданная ошибка при загрузке файла: {str(e)}")
            return

        if target == 'main':
            self.loaded_data = data
            # Устанавливаем current_index только если данные не пустые
            if data and len(data) > 0:
                self.current_index = 0
            else:
                self.current_index = -1
            # Очищаем validation_data при загрузке новых основных данных
            self.validation_data = []
            # Очищаем интерполированные значения при загрузке новых данных
            self.last_interpolated_mask_XY = None
            self.last_interpolated_pressure = None
            self.last_extrapolated_XY = None
            self.last_extrapolation_result = None
            
            # Обновляем список выбора скважин
            self.update_well_selection()
            
            # Обновляем отображение параметров ГРП
            self.update_grp_parameters()
            
            # Обновляем вкладку с загруженными данными
            self.update_data_tab()
            
            # Обновляем старые поля для совместимости
            current_item = self.current_data
            if current_item:
                self.update_interface_parameters()
            
            # Показываем информацию о загруженных данных
            total_points = sum(len(item.time) for item in data)
            self.show_info("Данные загружены", 
                          f"Загружено {len(data)} групп данных\n"
                          f"Всего измерений: {total_points}\n"
                          f"Текущая скважина: {self.well_combo_dim.currentText()}")
            
            # Запускаем диагностику асинхронно
            QTimer.singleShot(100, lambda: self.run_data_diagnostics(file_path))
            
            # Обновляем комбо в новой вкладке
            self.well_combo_dim.clear()
            for i, item in enumerate(self.loaded_data):
                self.well_combo_dim.addItem(f"Скважина {i+1} (Skin={item.skin:.2f})")
        else:  # target == 'validation'
            self.validation_data = data
            
            # Показываем информацию о загруженных данных
            total_points = sum(len(item.time) for item in data)
            self.show_info("Файл для проверки загружен", 
                          f"Загружено {len(data)} групп данных\n"
                          f"Всего измерений: {total_points}\n"
                          f"Данные будут использованы как эталон для оценки качества интерполяции")

    def load_template(self) -> None:
        """Загрузка CSV файла с данными разведки месторождений"""
        self._load_data_file(target='main')

    def load_validation_file(self) -> None:
        """Загрузка файла для проверки качества интерполяции (эталонные данные)"""
        self._load_data_file(target='validation')

    def run_data_diagnostics(self, file_path: str) -> None:
        """Запускает диагностику загруженных данных"""
        try:
            print("\n" + "="*60)
            print(f"ДИАГНОСТИКА ДАННЫХ: {file_path}")
            print("="*60)
            
            # Читаем CSV для диагностики
            df = pd.read_csv(file_path)
            
            # Конвертируем в формат для diag_dimensional если нужно
            # Функция ожидает колонки: 'X', 'Y', 'P', 'Q', 't'
            if not all(col in df.columns for col in ['X', 'Y', 'P', 'Q', 't']):
                # Преобразуем из формата WellTimeSeries
                if self.loaded_data and len(self.loaded_data) > 0:
                    item = self.loaded_data[0]
                    # Конвертируем в безразмерные параметры
                    params = self._get_params(item)
                    dim_data = convert_to_dimensionless_curves(
                        item.time, item.pressure, item.flow_rate, item.depression, params, x_mode='alt'
                    )
                    
                    # Создаем DataFrame в нужном формате
                    df_diag = pd.DataFrame({
                        'X': dim_data.X,
                        'Y': dim_data.Y,
                        'P': dim_data.pressure,
                        'Q': dim_data.flow_rate,
                        'dP': dim_data.dP,
                        't': item.time
                    })
                else:
                    print("⚠️ Не удалось преобразовать данные для диагностики")
                    return
            else:
                df_diag = df
            
            # Запускаем диагностику (вывод идет в консоль)
            diag_dimensional(df_diag)
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"⚠️ Ошибка диагностики: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_interface_parameters(self) -> None:
        self.thickness_doubleSpinBox.setValue(self.current_data.thickness)
        self.skin_doubleSpinBox.setValue(self.current_data.skin)
        self.width_doubleSpinBox.setValue(self.current_data.fracture_width)
        self.n_spinBox.setValue(self.current_data.fractures_count)
        self.aL_doubleSpinBox.setValue(self.current_data.a_l_ratio)
        
    def update_data_tab(self) -> None:
        """Обновляет таблицу на вкладке 'Загруженные данные'"""
        if not self.loaded_data:
            self.data_info_label.setText("Нет загруженных данных")
            self.data_table.setModel(None)
            return

        # Преобразуем данные текущей скважины в DataFrame
        current_item = self.current_data
        if current_item is None:
            # Очищаем таблицу, если нет данных
            model = QStandardItemModel(0, 3)
            model.setHorizontalHeaderLabels(["time", "pressure", "rate"])
            self.data_table.setModel(model)
            return
        
        df = pd.DataFrame({
            "Время t, ч": current_item.time,
            "Давление P, кгс/см²": current_item.pressure,
            "Депрессия, кгс/см²": current_item.depression,
            "Поток Q, м³/сут": current_item.flow_rate,
            "X": current_item.X,
            "Y": current_item.Y,
        })

        # Создаем модель для QTableView
        model = QStandardItemModel(df.shape[0], df.shape[1])
        model.setHorizontalHeaderLabels(df.columns)

        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                item = QStandardItem(str(df.iat[row, col]))
                item.setEditable(False)
                model.setItem(row, col, item)

        self.data_table.setModel(model)
        self.data_info_label.setText(f"Отображены данные скважины {self.current_index + 1} — {len(df)} строк")

    def update_well_selection(self) -> None:
        """Обновляет список выбора скважин"""
        self.well_combo_dim.clear()
        
        for i, item in enumerate(self.loaded_data):
            item_name = f"Скважина {i+1} (Skin={item.skin:.3f}, N={item.fractures_count}, a/L={item.a_l_ratio})"
            self.well_combo_dim.addItem(item_name)
        
        if self.loaded_data:
            self.well_combo_dim.setCurrentIndex(self.current_index)
    
    
    def on_well_changed(self, index: int) -> None:
        """Обработка смены выбранной скважины."""
        if self.loaded_data and 0 <= index < len(self.loaded_data):
            self.current_index = index
        else:
            # Если индекс невалидный, устанавливаем 0 или оставляем как есть
            if self.loaded_data:
                self.current_index = 0
            else:
                self.current_index = -1
        
        self.last_interpolated_mask_XY = None
        self.last_interpolated_pressure = None
        self.last_extrapolated_XY = None
        self.last_extrapolation_result = None
        self.last_fitted_XY = None
        self.last_fit_coefficients = None
        self.original_calc_XY = None
        self.last_filter_info = None
        self.reset_plots()
        
        self.update_interface_parameters()
        self.update_grp_parameters()
        self.update_data_tab()
        
    
    def update_grp_parameters(self) -> None:
        """Обновляет отображение параметров ГРП"""
        current_item = self.current_data
        if current_item is None:
            return
            
        self.skin_value_label.setText(f"Skin: {current_item.skin:.3f}")
        self.thickness_value_label.setText(f"Толщина: {current_item.thickness:.1f} м")
        self.fractures_value_label.setText(f"Трещины: {current_item.fractures_count}")
        self.fracture_width_label.setText(f"Ширина трещины: {current_item.fracture_width:.3f} м")
        self.fracture_length_label.setText(f"Длина трещины: {current_item.fracture_length:.1f} м")
        self.al_ratio_label.setText(f"a/L: {current_item.a_l_ratio:.3f}")
    
    def on_interpolate_data(self) -> None:
        """Интерполяция данных безразмерных кривых"""
        if self.current_data is None:
            return
        
        # Проверяем, есть ли пропуски в данных
        has_nan = (self.current_data.pressure.isna().any() or 
                   self.current_data.flow_rate.isna().any())
        
        if not has_nan:
            # Данные уже полные, интерполяция не требуется
            report = self._create_no_interpolation_report()
            
            if hasattr(self, 'results_text'):
                self.results_text.setPlainText(report)
            
            self.show_info("Интерполяция", 
                          "Данные уже полные, интерполяция не требуется")
            return
        
        try:
            # Определяем пропуски: NaN и большие расстояния между точками
            pressure_is_nan = self.current_data.pressure.isna()
            flow_rate_is_nan = self.current_data.flow_rate.isna()
            
            # Определяем пропуски по расстоянию между точками (по времени И по X-Y координатам)
            # Сначала получаем безразмерные координаты для определения пропусков
            params_temp = self._get_params(self.current_data)
            dim_data_temp = convert_to_dimensionless_curves(
                self.current_data.time, self.current_data.pressure, self.current_data.flow_rate, 
                self.current_data.depression, params_temp, x_mode='alt'
            )
            time_gaps = self._detect_gaps_by_distance(
                self.current_data.time.values,
                X=dim_data_temp.X,
                Y=dim_data_temp.Y,
                threshold_factor=1.0
            )
            
            # Объединяем маски: пропуски = NaN ИЛИ большие расстояния
            pressure_gaps_mask = (pressure_is_nan.values if hasattr(pressure_is_nan, 'values') else pressure_is_nan) | time_gaps
            flow_gaps_mask = (flow_rate_is_nan.values if hasattr(flow_rate_is_nan, 'values') else flow_rate_is_nan) | time_gaps
            
            n_nan_pressure = int(np.sum(pressure_gaps_mask))
            n_nan_flow = int(np.sum(flow_gaps_mask))
            
            # Сохраняем маску пропусков для подсветки на X-Y графике
            try:
                self.last_interpolated_mask_XY = pressure_gaps_mask.copy()
            except Exception:
                self.last_interpolated_mask_XY = None

            # Выполняем интерполяцию
            interp_info, _ = self._perform_interpolation()
            
            # Формируем отчет
            report = self._create_interpolation_report(n_nan_pressure, n_nan_flow, interp_info)
            
            # Выводим в текстовое поле результатов
            if hasattr(self, 'results_text'):
                self.results_text.setPlainText(report)
            
            self.on_plot_dimensionless_selected()
            
            # Показываем информационное сообщение
            method_names = {
                'linear': 'Линейная регрессия',
                'rbf': 'RBF интерполяция (Thin Plate Spline)',
                'gp': 'Гауссовский процесс',
                'quadratic_regression': 'Квадратичная регрессия',
                'local_pchip': 'Локальная PCHIP интерполяция',
                'trained_model': 'Обученная модель',
                'trained_interpolation_model': 'Модель интерполяции'
            }
            best_rmse = interp_info['rmse_scores'].get(interp_info['best_method'], 0)
            quality = self._get_quality_label(best_rmse)
            self.show_info("Интерполяция", 
                          f"Данные интерполированы методом безразмерных кривых\n\n"
                          f"Выбранный метод: {method_names.get(interp_info['best_method'], interp_info['best_method'])}\n"
                          f"RMSE: {best_rmse:.3f}\n"
                          f"Качество: {quality}")
        except Exception as e:
            self.show_warning("Ошибка", f"Ошибка интерполяции: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def reset_plots(self) -> None:
        """Сброс всех графиков и чекбоксов"""
        # Очищаем график
        if hasattr(self, 'dimensionless_plot'):
            self.dimensionless_plot.clear()
            self.dimensionless_plot.setLabel('bottom', 'X (безразмерный фильтрационный параметр)')
            self.dimensionless_plot.setLabel('left', 'Безразмерный параметр')
            self.dimensionless_plot.setTitle("Безразмерные кривые МГРП")
            self.dimensionless_plot.showGrid(x=True, y=True)
        
        # Снимаем все чекбоксы
        if hasattr(self, 'cb_dim_pD'):
            self.cb_dim_pD.setChecked(False)
        if hasattr(self, 'cb_dim_dpD'):
            self.cb_dim_dpD.setChecked(False)
        if hasattr(self, 'cb_dim_tD'):
            self.cb_dim_tD.setChecked(False)
        if hasattr(self, 'cb_dim_CD'):
            self.cb_dim_CD.setChecked(False)
        if hasattr(self, 'cb_XY_plot'):
            self.cb_XY_plot.setChecked(False)
        if hasattr(self, 'cb_calc_XY'):
            self.cb_calc_XY.setChecked(False)
        if hasattr(self, 'cb_type_gry'):
            self.cb_type_gry.setChecked(False)
        if hasattr(self, 'cb_type_cinco'):
            self.cb_type_cinco.setChecked(False)
        if hasattr(self, 'cb_type_valko'):
            self.cb_type_valko.setChecked(False)
        if hasattr(self, 'cb_gfunc'):
            self.cb_gfunc.setChecked(False)
        if hasattr(self, 'cb_mbt'):
            self.cb_mbt.setChecked(False)
        if hasattr(self, 'cb_real_p'):
            self.cb_real_p.setChecked(False)
        if hasattr(self, 'cb_real_q'):
            self.cb_real_q.setChecked(False)
        
        # Очищаем отчёт
        if hasattr(self, 'text_report'):
            self.text_report.clear()
        
        # Сбрасываем внутренние состояния подсветки/экстраполяции/фильтрации
        self.last_interpolated_mask_XY = None
        self.last_interpolated_pressure = None
        self.last_extrapolated_XY = None
        self.last_extrapolation_result = None
        self.last_fitted_XY = None
        self.last_fit_coefficients = None
        self.original_calc_XY = None
        self.last_filter_info = None
        
        
    def on_reset_plots(self) -> None:
        """Сброс графиков с уведомлением"""
        self.reset_plots()
        if not self.test_mode:
            self.show_info("Графики очищены", "Все графики и чекбоксы сброшены")


    def on_extrapolate_xy(self) -> None:
        """Экстраполяция и наложение X–Y кривой на график на основе экстраполированных размерных параметров."""
        if self.current_data is None:
            return
        try:
            # Готовим параметры скважины
            params = self._get_params(self.current_data)
            
            # Преобразуем WellTimeSeries в DataFrame для нового экстраполятора
            n_points = len(self.current_data.time)
            df = pd.DataFrame({
                't': self.current_data.time.values,
                'P': self.current_data.pressure.values,
                'dP': self.current_data.depression.values,
                'Q': self.current_data.flow_rate.values,
                'Skin': [self.current_data.skin] * n_points,  # Статичные параметры для всех строк
                'h': [self.current_data.thickness] * n_points,
                'N': [self.current_data.fractures_count] * n_points,
                'W': [self.current_data.fracture_width] * n_points,
                'L': [self.current_data.fracture_length] * n_points,
                'a/L': [self.current_data.a_l_ratio] * n_points
            })
            
            extrapolator = DimensionlessExtrapolator()
            result = extrapolator.run(
                df=df,
                well_params=params,
                n_future=n_points//2,
                method="adaptive",
                check_rmse=True  # Проверяем RMSE для оценки качества
            )
            
            # Извлекаем экстраполированные X и Y (новый формат или старый для обратной совместимости)
            if 'X_ext' in result and 'Y_ext' in result:
                X_ext = result['X_ext']
                Y_ext = result['Y_ext']
            else:
                df_pred = result['df_pred']
                X_ext = df_pred['X'].values
                Y_ext = df_pred['Y'].values
            
            self.last_extrapolated_XY = (X_ext, Y_ext)
            self.last_extrapolation_result = result  # Сохраняем результат для отчёта
            
            # Перестраиваем график с наложением экстраполяции
            self.on_plot_dimensionless_selected()
            
            # Формируем и выводим отчёт об экстраполяции
            extrapolation_report = self._create_extrapolation_report(result)
            if hasattr(self, 'results_text'):
                # Добавляем отчёт к существующему тексту
                current_text = self.results_text.toPlainText()
                if current_text:
                    self.results_text.setPlainText(current_text + "\n\n" + extrapolation_report)
                else:
                    self.results_text.setPlainText(extrapolation_report)
            
            # Показываем информацию о результатах
            meta = result.get('meta', {})
            method_used = meta.get('method', 'unknown')
            n_points = meta.get('n_future_points', 0)
            self.show_info("Экстраполяция", 
                         f"Экстраполированная кривая X–Y добавлена на график\n"
                         f"Метод: {method_used}, точек: {n_points}")
        except Exception as e:
            import traceback
            self.show_warning("Ошибка экстраполяции", 
                            f"Не удалось выполнить экстраполяцию: {str(e)}\n{traceback.format_exc()}")
    
    def on_fit_xy_curve(self) -> None:
        """Подгонка расчётной кривой X-Y к эталонной из данных."""
        if self.current_data is None:
            self.show_warning("Ошибка", "Нет данных для подгонки")
            return
        
        # Проверяем наличие X и Y в данных
        if not (hasattr(self.current_data, 'X') and self.current_data.X is not None and
                hasattr(self.current_data, 'Y') and self.current_data.Y is not None):
            self.show_warning("Ошибка", "В данных отсутствуют X и Y. Невозможно выполнить подгонку.")
            return
        
        try:
            # Получаем параметры скважины
            params = self._get_params(self.current_data)
            
            # Вычисляем расчётные X и Y
            dim_data = convert_to_dimensionless_curves(
                self.current_data.time, self.current_data.pressure, self.current_data.flow_rate, self.current_data.depression, params, x_mode='alt'
            )
            
            # Сохраняем оригинальные расчётные значения
            self.original_calc_XY = (dim_data.X.copy(), dim_data.Y.copy())
            
            # Получаем эталонные значения из данных
            X_data = self.current_data.X.values
            Y_data = self.current_data.Y.values
            X_calc = dim_data.X
            Y_calc = dim_data.Y
            
            # Используем обученную модель, если она есть
            if (self.trained_interpolator is not None and 
                isinstance(self.trained_interpolator, QuadraticRegressionModel) and
                self.trained_interpolator.is_fitted):
                # Используем обученную модель для подгонки
                X_fitted, Y_fitted = self.trained_interpolator.predict(X_calc, Y_calc)
                
                # Вычисляем метрики
                mask = np.isfinite(X_data) & np.isfinite(Y_data) & np.isfinite(X_fitted) & np.isfinite(Y_fitted)
                if np.any(mask):
                    rmse_y = np.sqrt(np.mean((Y_data[mask] - Y_fitted[mask]) ** 2))
                    rmse_x = np.sqrt(np.mean((X_data[mask] - X_fitted[mask]) ** 2))
                    
                    y_mean = np.mean(Y_data[mask])
                    ss_tot = np.sum((Y_data[mask] - y_mean) ** 2)
                    r2_y = 1 - np.sum((Y_data[mask] - Y_fitted[mask]) ** 2) / ss_tot if ss_tot > 0 else 0.0
                    
                    y_range = np.max(Y_data[mask]) - np.min(Y_data[mask])
                    accuracy = max(0, (1 - rmse_y / y_range) * 100) if y_range > 0 else 0.0
                else:
                    rmse_y = np.inf
                    rmse_x = np.inf
                    r2_y = 0.0
                    accuracy = 0.0
                
                coef = self.trained_interpolator.get_coefficients()
                
                # Формируем результат в формате fit_xy_curve_coefficients
                fit_result = {
                    'a': coef['a'],
                    'a_y': coef['a_y'],
                    'b': coef['b'],
                    'c': coef['c'],
                    'rmse': rmse_y,
                    'rmse_before': rmse_y,  # Для обученной модели это не применимо
                    'accuracy': accuracy,
                    'r2': r2_y,
                    'X_fitted': X_fitted,
                    'Y_fitted': Y_fitted,
                    'c_clipped': False,
                    'fallback_used': False
                }
            else:
                # Используем старый метод подгонки для одной скважины
                fit_result = fit_xy_curve_coefficients(X_data, Y_data, X_calc, Y_calc, fit_only_y=False)
            
            # Сохраняем результаты (коэффициенты будут применяться автоматически при построении графиков)
            self.last_fitted_XY = (fit_result['X_fitted'], fit_result['Y_fitted'])
            self.last_fit_coefficients = {
                'a': fit_result['a'],  # Коэффициент для X
                'a_y': fit_result.get('a_y', 1.0),  # Коэффициент для Y (линейный член)
                'b': fit_result['b'],  # Свободный член для Y
                'c': fit_result.get('c', 0.0),  # Квадратичный коэффициент для Y
                'rmse': fit_result['rmse'],
                'rmse_before': fit_result.get('rmse_before', fit_result['rmse']),
                'accuracy': fit_result['accuracy'],
                'r2': fit_result['r2'],
                'c_clipped': fit_result.get('c_clipped', False),
                'fallback_used': fit_result.get('fallback_used', False)
            }
            
            # Формируем отчёт
            report = "=" * 60 + "\n"
            report += "ПОДГОНКА РАСЧЁТНОЙ КРИВОЙ X-Y\n"
            report += "=" * 60 + "\n\n"
            
            # Указываем, используется ли обученная модель
            if (self.trained_interpolator is not None and 
                isinstance(self.trained_interpolator, QuadraticRegressionModel) and
                self.trained_interpolator.is_fitted):
                metrics = self.trained_interpolator.get_metrics()
                report += f"ℹ️ Использована обученная модель (квадратичная регрессия с регуляризацией)\n"
                report += f"   Обучена на {metrics['n_samples']} точках из всех скважин\n\n"
            else:
                report += f"ℹ️ Использована локальная подгонка для текущей скважины\n\n"
            
            report += f"✅ ЛУЧШИЕ КОЭФФИЦИЕНТЫ:\n"
            report += f"   a (по X) = {fit_result['a']:.4g}\n"
            report += f"   b (по Y) = {fit_result['b']:.4g}\n"
            c_val = fit_result.get('c', 0.0)
            report += f"   c (квадратичный) = {c_val:.4g}\n"
            if fit_result.get('c_clipped', False):
                report += f"   ⚠️ c был ограничен до [-0.2, 0.2]\n"
            if fit_result.get('fallback_used', False):
                report += f"   ⚠️ Использован fallback (c=0) из-за ухудшения RMSE\n"
            report += f"   RMSE до подгонки = {fit_result.get('rmse_before', fit_result['rmse']):.4e}\n"
            report += f"   RMSE после подгонки = {fit_result['rmse']:.4e}\n"
            report += f"   Точность = {fit_result['accuracy']:.2f}%\n"
            report += f"   R² = {fit_result['r2']:.4f}\n\n"
            report += "📘 Итоговая аппроксимирующая формула:\n"
            report += f"   X_fit = {fit_result['a']:.3g} * (0.00864 * k * h * ΔP / (μ * B * Q))\n"
            if abs(c_val) < 1e-10:
                report += f"   Y_fit = {fit_result['b']:.3g} * (Q * B * t / (24 * φ * ct * h * L² * ΔP))\n"
            else:
                report += f"   Y_fit = {fit_result['b']:.3g} * Y + {c_val:.3g} * Y²\n"
            report += "=" * 60 + "\n"
            report += "ℹ️ Коэффициенты сохранены и будут применяться к расчётной кривой\n"
            report += "   при каждом построении графика до ручной очистки.\n"
            report += "=" * 60 + "\n"
            
            # Выводим отчёт
            if hasattr(self, 'text_report'):
                self.text_report.setText(report)
            
            # Перестраиваем график с подогнанными данными
            self.on_plot_dimensionless_selected()
            
            c_val = fit_result.get('c', 0.0)
            model_info = ""
            if (self.trained_interpolator is not None and 
                isinstance(self.trained_interpolator, QuadraticRegressionModel) and
                self.trained_interpolator.is_fitted):
                model_info = "\n(Использована обученная модель на массиве всех скважин)"
            else:
                model_info = "\n(Использована локальная подгонка для текущей скважины)"
            
            self.show_info("Подгонка выполнена", 
                         f"Коэффициенты: a={fit_result['a']:.3g}, b={fit_result['b']:.3g}, c={c_val:.3g}\n"
                         f"Точность: {fit_result['accuracy']:.2f}%\n"
                         f"Коэффициенты сохранены и будут применяться автоматически.{model_info}")
            
        except Exception as e:
            import traceback
            self.show_warning("Ошибка подгонки", 
                            f"Не удалось выполнить подгонку: {str(e)}\n{traceback.format_exc()}")
    
    def on_plot_dimensionless_selected(self) -> None:
        """Обработка нажатия на кнопку 'Построить график'."""
        self.dimensionless_plot.clear()
        # Не очищаем text_report полностью, чтобы сохранить отчёт о подгонке

        current_item = self.current_data
        if current_item is None:
            # Добавляем отладочную информацию
            debug_info = f"❌ Нет данных для построения.\n"
            debug_info += f"loaded_data: {len(self.loaded_data) if self.loaded_data else 0} элементов\n"
            debug_info += f"current_index: {self.current_index}\n"
            if self.loaded_data:
                debug_info += f"Доступные индексы: 0-{len(self.loaded_data) - 1}"
            self.text_report.setText(debug_info)
            return

        try:
            # Параметры скважины
            params = self._get_params(current_item)
            
            # 1️⃣ Конвертация в безразмерные параметры
            # Обрабатываем случай, когда depression может быть None
            depression = getattr(current_item, 'depression', None)
            try:
                dim_data = convert_to_dimensionless_curves(
                    current_item.time, current_item.pressure, current_item.flow_rate, depression, params, x_mode='alt'
                )
            except Exception as e:
                # Логируем ошибку
                try:
                    from helpers.math_error_logger import log_computation_error
                    log_computation_error(
                        subsystem="gui",
                        method="on_plot_dimensionless_selected",
                        exception=e,
                        data_volume=len(current_item.time) if current_item.time is not None else None,
                        data_quality=1.0 - (np.sum(np.isnan(current_item.pressure)) / len(current_item.pressure)) if current_item.pressure is not None and len(current_item.pressure) > 0 else None,
                        context={"has_depression": depression is not None}
                    )
                except Exception:
                    pass
                raise

            # 2️⃣ Определяем, какие группы графиков выбраны
            checked_groups = {
                # Реальные параметры
                'real_params': self.cb_real_p.isChecked() or self.cb_real_q.isChecked(),
                'cb_real_p': self.cb_real_p.isChecked(),
                'cb_real_q': self.cb_real_q.isChecked(),
                
                # Безразмерные кривые
                'dimensionless': (self.cb_dim_pD.isChecked() or self.cb_dim_dpD.isChecked() or 
                                 self.cb_dim_tD.isChecked() or self.cb_dim_CD.isChecked()),
                'cb_dim_pD': self.cb_dim_pD.isChecked(),
                'cb_dim_dpD': self.cb_dim_dpD.isChecked(),
                'cb_dim_tD': self.cb_dim_tD.isChecked(),
                'cb_dim_CD': self.cb_dim_CD.isChecked(),
                'cb_XY_plot': hasattr(self, 'cb_XY_plot') and self.cb_XY_plot.isChecked(),
                'cb_calc_XY': hasattr(self, 'cb_calc_XY') and self.cb_calc_XY.isChecked(),
                
                # Типовые кривые
                'type_curves': (self.cb_type_gry.isChecked() or self.cb_type_cinco.isChecked() or 
                               self.cb_type_valko.isChecked()),
                'cb_type_gry': self.cb_type_gry.isChecked(),
                'cb_type_cinco': self.cb_type_cinco.isChecked(),
                'cb_type_valko': self.cb_type_valko.isChecked(),
                
                # Специальные пространства (если есть)
                'special': (hasattr(self, 'cb_gfunc') and self.cb_gfunc.isChecked()) or \
                          (hasattr(self, 'cb_mbt') and self.cb_mbt.isChecked()),
                'cb_gfunc': hasattr(self, 'cb_gfunc') and self.cb_gfunc.isChecked(),
                'cb_mbt': hasattr(self, 'cb_mbt') and self.cb_mbt.isChecked(),
            }
            
            # Проверяем, выбрано ли что-то для отображения
            # cb_calc_XY может быть выбран независимо от других графиков
            has_any_selected = (checked_groups.get('real_params', False) or
                              checked_groups.get('dimensionless', False) or
                              checked_groups.get('cb_XY_plot', False) or
                              checked_groups.get('cb_calc_XY', False) or
                              checked_groups.get('type_curves', False) or
                              checked_groups.get('special', False))
            
            if not has_any_selected:
                self.text_report.setText("⚠️ Выберите хотя бы один график для отображения в чекбоксах.")
                return
            
            # 3️⃣ Подготовка данных для валидации
            # validation_data передается только если это действительно эталонные данные для валидации,
            # а не просто текущие данные с пропусками
            validation_data = None
            # Проверяем, что validation_data содержит эталонные данные (не текущие данные)
            if self.validation_data and len(self.validation_data) > 0 and current_item is not None:
                try:
                    ref_item = self.validation_data[0]
                    # Строгая проверка: убеждаемся, что это не те же данные, что и current_item
                    # Проверяем по нескольким критериям:
                    # 1. Это не тот же объект (обязательно)
                    # 2. И хотя бы одно из условий различия:
                    #    - Разные временные ряды (по длине или значениям)
                    #    - Разные параметры скважины
                    #    - Разные данные давления/дебита
                    is_different = (
                        ref_item is not current_item and
                        (
                            # Разные временные ряды
                            (len(ref_item.time) != len(current_item.time) or
                             not np.array_equal(ref_item.time.values, current_item.time.values)) or
                            # Разные параметры скважины
                            (ref_item.skin != current_item.skin or
                             ref_item.fractures_count != current_item.fractures_count or
                             ref_item.a_l_ratio != current_item.a_l_ratio) or
                            # Разные данные давления/дебита
                            (len(ref_item.pressure) != len(current_item.pressure) or
                             not np.allclose(ref_item.pressure.values, current_item.pressure.values, 
                                           rtol=1e-3, equal_nan=True)) or
                            (len(ref_item.flow_rate) != len(current_item.flow_rate) or
                             not np.allclose(ref_item.flow_rate.values, current_item.flow_rate.values,
                                           rtol=1e-3, equal_nan=True))
                        )
                    )
                    
                    if is_different:
                        ref_params = self._get_params(ref_item)
                        ref_dim = convert_to_dimensionless_curves(
                            ref_item.time, ref_item.pressure, ref_item.flow_rate, ref_item.depression, ref_params, x_mode='alt'
                        )
                        validation_data = {'ref_dim': ref_dim}
                    else:
                        # Это те же данные - не используем как эталон
                        validation_data = None
                except Exception as e:
                    # Не выводим ошибку, просто не используем validation_data
                    validation_data = None
            
            # 4️⃣ Используем новую функцию для отображения сгруппированных графиков
            # ВАЖНО: Всегда используем исходные X и Y из данных (current_item), если они есть
            # X-Y кривые должны вычисляться на основе исходных данных, без изменений
            # Пересчитанные X и Y из dim_data используются только для отображения расчётных кривых (через чекбокс)
            X_data = None
            Y_data = None
            # Всегда используем исходные X и Y из данных, если они есть
            X_data = current_item.X if hasattr(current_item, 'X') and current_item.X is not None else None
            Y_data = current_item.Y if hasattr(current_item, 'Y') and current_item.Y is not None else None
            
            show_calc_XY = hasattr(self, 'cb_calc_XY') and self.cb_calc_XY.isChecked()
            
            # Если были сохранены коэффициенты подгонки, применяем их к расчётной кривой
            # Коэффициенты применяются автоматически при каждом построении графика
            if self.last_fit_coefficients is not None and self.original_calc_XY is not None:
                # Восстанавливаем оригинальные расчётные значения
                dim_data.X = self.original_calc_XY[0].copy()
                dim_data.Y = self.original_calc_XY[1].copy()
                
                # Применяем сохранённые коэффициенты
                a = self.last_fit_coefficients['a']  # Коэффициент для X
                a_y = self.last_fit_coefficients.get('a_y', 1.0)  # Коэффициент для Y (линейный член)
                b = self.last_fit_coefficients['b']  # Свободный член для Y
                c = self.last_fit_coefficients.get('c', 0.0)  # Квадратичный коэффициент для Y
                
                # Применяем преобразования: X_fit = a * X, Y_fit = a_y * Y + b + c * Y^2
                dim_data.X = dim_data.X * a
                
                # Применяем квадратичное преобразование к Y: Y_fit = a_y * Y + b + c * Y^2
                dim_data.Y = a_y * dim_data.Y + b + c * (dim_data.Y ** 2)
                
                # Обновляем last_fitted_XY для совместимости
                self.last_fitted_XY = (dim_data.X.copy(), dim_data.Y.copy())
            
            # Применяем коэффициенты подгонки к экстраполированным X и Y, если они есть
            if self.last_extrapolated_XY is not None and self.last_fit_coefficients is not None:
                X_ext, Y_ext = self.last_extrapolated_XY
                a = self.last_fit_coefficients['a']  # Коэффициент для X
                a_y = self.last_fit_coefficients.get('a_y', 1.0)  # Коэффициент для Y (линейный член)
                b = self.last_fit_coefficients['b']  # Свободный член для Y
                c = self.last_fit_coefficients.get('c', 0.0)  # Квадратичный коэффициент для Y
                
                # Применяем те же коэффициенты к экстраполированным значениям
                X_ext = X_ext * a
                Y_ext = a_y * Y_ext + b + c * (Y_ext ** 2)
                
                # Обновляем экстраполированные значения с применёнными коэффициентами
                self.last_extrapolated_XY = (X_ext.copy(), Y_ext.copy())

            # Если X и Y отсутствуют в данных, просто не будем их использовать для графика "X-Y (из данных)"
            # Расчётные X и Y можно отображать независимо через чекбокс "Отобразить расчётные X и Y"
            
            plot_dimensionless_grouped(
                plot_widget=self.dimensionless_plot,
                dim_data=dim_data,
                time=current_item.time,
                pressure=current_item.pressure,
                flow_rate=current_item.flow_rate,
                checked_groups=checked_groups,
                validation_data=validation_data,
                X_data=X_data,
                Y_data=Y_data,
                show_calculated_XY=show_calc_XY,
                interpolated_mask_XY=(self.last_interpolated_mask_XY if self.last_interpolated_mask_XY is not None else None),
                extrapolated_XY=(self.last_extrapolated_XY if self.last_extrapolated_XY is not None else None)
            )
            
            self.text_report.append("✅ График построен успешно")
            
            # Добавляем краткое резюме, если была проведена интерполяция
            if hasattr(self, 'last_interpolation_info') and self.last_interpolation_info:
                interp_info = self.last_interpolation_info
                method_names = {
                    'linear': 'Линейная регрессия',
                    'rbf': 'RBF',
                    'gp': 'GP'
                }
                best_rmse = interp_info['rmse_scores'].get(interp_info['best_method'], 0)
                quality = self._get_quality_label(best_rmse, short=True)
                self.text_report.append(
                    f"📊 Интерполяция: {method_names.get(interp_info['best_method'], interp_info['best_method'])}, "
                    f"RMSE={best_rmse:.6e} ({quality})"
                )
                
                # Добавляем метрики относительно эталона, если они есть
                if 'reference_metrics' in interp_info:
                    ref_metrics = interp_info['reference_metrics']
                    mse = ref_metrics.get('mse', 0)
                    mae = ref_metrics.get('mae', 0)
                    r2 = ref_metrics.get('r2', 0)
                    self.text_report.append(
                        f"   Метрики (эталон): MSE={mse:.6e}, MAE={mae:.6e}, R²={r2:.4f}"
                    )
            
            # Добавляем краткое резюме, если была проведена фильтрация
            if hasattr(self, 'last_filter_info') and self.last_filter_info:
                filter_info = self.last_filter_info
                quality = self._get_quality_label(filter_info['rmse'], short=True)
                self.text_report.append(
                    f"📊 Фильтрация: {filter_info['method_name']}, "
                    f"RMSE={filter_info['rmse']:.6e} ({quality}), "
                    f"точность={filter_info['accuracy']:.2f}%"
                )
                self.text_report.append(
                    f"   SNR: {filter_info['snr_before']:.2f} → {filter_info['snr_after']:.2f} дБ "
                    f"({filter_info['snr_improvement']:+.2f} дБ)"
                )
            
            # Добавляем краткое резюме, если была проведена экстраполяция
            if hasattr(self, 'last_extrapolation_result') and self.last_extrapolation_result:
                result = self.last_extrapolation_result
                meta = result.get('meta', {})
                rmse = result.get('rmse', {})
                
                method_used = meta.get('method', 'unknown')
                n_points = meta.get('n_future_points', 0)
                
                self.text_report.append(
                    f"Экстраполяция: метод={method_used}, точек={n_points}"
                )
                
                if rmse:
                    if isinstance(rmse, dict):
                        rmse_mean = rmse.get('mean', 0)
                        quality = self._get_quality_label(rmse_mean, short=True)
                        self.text_report.append(
                            f"   RMSE: X={rmse.get('X', 0):.6e}, Y={rmse.get('Y', 0):.6e}, "
                            f"среднее={rmse_mean:.6e} ({quality})"
                        )
                    else:
                        quality = self._get_quality_label(rmse, short=True)
                        self.text_report.append(
                            f"   RMSE={rmse:.6e} ({quality})"
                        )
            
        except Exception as e:
            self.text_report.setText(f"Ошибка построения графика: {str(e)}")
            import traceback
            print(traceback.format_exc())      

    def on_checkbox_toggled(self):
        """Вызывается при изменении состояния любого чекбокса"""
        self.on_plot_dimensionless_selected() 

    def on_ml_filter(self) -> None:
        """ML-фильтрация данных - применяется к исходным P, Q, dP"""
        if self.current_data is None:
            return
        
        try:
            # СТРОГОЕ УСЛОВИЕ: Фильтрация применяется к исходным данным (P, Q, dP), а не к безразмерным кривым
            time = self.current_data.time.values
            pressure = self.current_data.pressure.values
            flow_rate = self.current_data.flow_rate.values
            depression = self.current_data.depression.values
            
            # Удаляем NaN значения для фильтрации
            valid_mask = (np.isfinite(time) & np.isfinite(pressure) & 
                         np.isfinite(flow_rate) & np.isfinite(depression))
            
            if not np.any(valid_mask):
                self.show_warning("Ошибка", "Нет валидных данных для фильтрации")
                return
            
            time_clean = time[valid_mask]
            pressure_clean = pressure[valid_mask]
            flow_rate_clean = flow_rate[valid_mask]
            depression_clean = depression[valid_mask]
            
            # Сортируем по времени для правильной фильтрации
            sort_idx = np.argsort(time_clean)
            time_sorted = time_clean[sort_idx]
            pressure_sorted = pressure_clean[sort_idx]
            flow_rate_sorted = flow_rate_clean[sort_idx]
            depression_sorted = depression_clean[sort_idx]
            
            # Вычисляем исходные метрики для давления
            snr_before = compute_snr(pressure_sorted)
            
            # Автоматический выбор и применение фильтра к исходным данным
            # Используем время как координату x
            filtered_pressure_sorted = SignalFilters.denoise(pressure_sorted, method=None, x=time_sorted)
            filtered_flow_rate_sorted = SignalFilters.denoise(flow_rate_sorted, method=None, x=time_sorted)
            filtered_depression_sorted = SignalFilters.denoise(depression_sorted, method=None, x=time_sorted)
            
            # Определяем, какой метод был выбран автоматически
            selected_method = select_filter_method(pressure_sorted, time_sorted)
            
            # Применяем физические ограничения к отфильтрованным данным
            # Для давления: монотонность не критична, но ограничиваем кривизну
            filtered_pressure_sorted = PhysicsConstraints.enforce_all(
                filtered_pressure_sorted,
                x=time_sorted,
                monotonic=False,  # Давление может колебаться
                limit_curvature=True,
                remove_oscillations=True,
                asymptotic_fix=True
            )
            
            # Для депрессии: должна быть неотрицательной и не возрастающей
            filtered_depression_sorted = PhysicsConstraints.enforce_all(
                filtered_depression_sorted,
                x=time_sorted,
                monotonic=True,  # Депрессия должна быть не возрастающей
                limit_curvature=True,
                remove_oscillations=True,
                asymptotic_fix=True
            )
            
            # Восстанавливаем исходный порядок
            filtered_pressure = np.zeros_like(pressure_clean)
            filtered_flow_rate = np.zeros_like(flow_rate_clean)
            filtered_depression = np.zeros_like(depression_clean)
            
            filtered_pressure[sort_idx] = filtered_pressure_sorted
            filtered_flow_rate[sort_idx] = filtered_flow_rate_sorted
            filtered_depression[sort_idx] = filtered_depression_sorted
            
            # Восстанавливаем полные массивы с исходными индексами
            filtered_pressure_full = self.current_data.pressure.copy()
            filtered_flow_rate_full = self.current_data.flow_rate.copy()
            filtered_depression_full = self.current_data.depression.copy()
            
            # Создаем массив для восстановления исходного порядка
            valid_indices = np.where(valid_mask)[0]
            filtered_pressure_full.iloc[valid_indices] = filtered_pressure
            filtered_flow_rate_full.iloc[valid_indices] = filtered_flow_rate
            filtered_depression_full.iloc[valid_indices] = filtered_depression
            
            # Обновляем исходные данные
            self.current_data.pressure = filtered_pressure_full
            self.current_data.flow_rate = filtered_flow_rate_full
            self.current_data.depression = filtered_depression_full
            
            # Вычисляем метрики после фильтрации
            snr_after = compute_snr(filtered_pressure)
            snr_improvement = snr_after - snr_before
            
            # Вычисляем RMSE и другие метрики относительно исходных данных
            rmse = np.sqrt(np.mean((pressure_clean - filtered_pressure) ** 2))
            mae = np.mean(np.abs(pressure_clean - filtered_pressure))
            
            # Вычисляем относительную ошибку и точность
            pressure_range = np.max(pressure_clean) - np.min(pressure_clean)
            relative_error = (rmse / pressure_range * 100) if pressure_range > 0 else 0.0
            accuracy = max(0, 100 - relative_error)
            
            # Если есть эталонные данные, рассчитываем метрики относительно эталона
            reference_metrics = None
            if hasattr(self, 'validation_data') and self.validation_data and len(self.validation_data) > 0:
                try:
                    ref_item = self.validation_data[0]
                    # Проверяем, что это действительно эталонные данные
                    if (ref_item is not None and 
                        hasattr(ref_item, 'time') and hasattr(ref_item, 'pressure') and
                        (ref_item is not self.current_data or 
                         len(ref_item.pressure) != len(self.current_data.pressure) or
                         not np.allclose(ref_item.pressure.values, self.current_data.pressure.values, 
                                       rtol=1e-3, equal_nan=True))):
                        
                        # Получаем эталонные данные
                        ref_time = ref_item.time.values
                        ref_pressure = ref_item.pressure.values
                        ref_flow_rate = ref_item.flow_rate.values
                        ref_depression = ref_item.depression.values
                        
                        # Удаляем NaN из эталона
                        ref_valid_mask = (np.isfinite(ref_time) & np.isfinite(ref_pressure) & 
                                         np.isfinite(ref_flow_rate) & np.isfinite(ref_depression))
                        
                        if np.any(ref_valid_mask):
                            ref_time_clean = ref_time[ref_valid_mask]
                            ref_pressure_clean = ref_pressure[ref_valid_mask]
                            
                            # Интерполируем отфильтрованные данные на временные точки эталона
                            from scipy.interpolate import interp1d
                            
                            # Сортируем для интерполяции
                            time_sorted_idx = np.argsort(time_clean)
                            ref_time_sorted_idx = np.argsort(ref_time_clean)
                            
                            # Интерполируем отфильтрованное давление на эталонные временные точки
                            interp_func = interp1d(
                                time_clean[time_sorted_idx], 
                                filtered_pressure[time_sorted_idx],
                                kind='linear',
                                bounds_error=False,
                                fill_value='extrapolate'
                            )
                            filtered_pressure_on_ref = interp_func(ref_time_clean[ref_time_sorted_idx])
                            
                            # Вычисляем метрики относительно эталона
                            ref_pressure_sorted = ref_pressure_clean[ref_time_sorted_idx]
                            
                            # RMSE относительно эталона
                            ref_rmse = np.sqrt(np.mean((ref_pressure_sorted - filtered_pressure_on_ref) ** 2))
                            
                            # MAE относительно эталона
                            ref_mae = np.mean(np.abs(ref_pressure_sorted - filtered_pressure_on_ref))
                            
                            # R² относительно эталона
                            ss_res = np.sum((ref_pressure_sorted - filtered_pressure_on_ref) ** 2)
                            ss_tot = np.sum((ref_pressure_sorted - np.mean(ref_pressure_sorted)) ** 2)
                            ref_r2 = 1 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 1e-12 else 0.0
                            
                            # Относительная ошибка относительно эталона
                            ref_range = np.max(ref_pressure_sorted) - np.min(ref_pressure_sorted)
                            ref_relative_error = (ref_rmse / ref_range * 100) if ref_range > 0 else 0.0
                            ref_accuracy = max(0, 100 - ref_relative_error)
                            
                            reference_metrics = {
                                'rmse': ref_rmse,
                                'mae': ref_mae,
                                'r2': ref_r2,
                                'relative_error': ref_relative_error,
                                'accuracy': ref_accuracy,
                                'n_points': len(ref_pressure_sorted)
                            }
                except Exception as e:
                    # Если не удалось рассчитать метрики, просто пропускаем
                    print(f"Не удалось рассчитать метрики фильтрации относительно эталона: {e}")
            
            # Сохраняем информацию о фильтрации для вывода в отчёте
            method_names = {
                'lowess': 'LOWESS (RLOESS)',
                'savgol': 'Savitzky-Golay',
                'gaussian': 'Gaussian',
                'kalman': 'Kalman',
                'log_domain': 'Log-domain',
                'hybrid': 'Hybrid'
            }
            
            self.last_filter_info = {
                'method': selected_method,
                'method_name': method_names.get(selected_method, selected_method),
                'snr_before': snr_before,
                'snr_after': snr_after,
                'snr_improvement': snr_improvement,
                'rmse': rmse,
                'mae': mae,
                'relative_error': relative_error,
                'accuracy': accuracy,
                'n_points': len(pressure_clean),
                'reference_metrics': reference_metrics  # Метрики относительно эталона
            }
            
            # Формируем отчёт
            report = self._create_filter_report()
            self.text_report.setText(report)
            
            # Обновляем график
            self.on_plot_dimensionless_selected()
            
            # Показываем информационное сообщение
            quality = self._get_quality_label(rmse)
            self.show_info("ML фильтрация", 
                          f"Фильтрация завершена\n\n"
                          f"Метод: {method_names.get(selected_method, selected_method)}\n"
                          f"Точность: {accuracy:.2f}%\n"
                          f"RMSE: {rmse:.6e} ({quality})\n"
                          f"Улучшение SNR: {snr_improvement:+.2f} дБ")
            
        except Exception as e:
            self.show_warning("Ошибка", f"Ошибка ML фильтрации: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def _create_filter_report(self) -> str:
        """Создаёт отчёт о результатах фильтрации"""
        if not hasattr(self, 'last_filter_info') or not self.last_filter_info:
            return "Фильтрация не выполнялась"
        
        info = self.last_filter_info
        quality = self._get_quality_label(info['rmse'])
        
        report = f"Результаты фильтрации\n"
        report += f"{'=' * 50}\n\n"
        report += f"Метод фильтрации: {info['method_name']} (автоматический выбор)\n"
        report += f"Количество точек: {info['n_points']}\n\n"
        report += f"Метрики качества:\n"
        report += f"  • SNR до фильтрации: {info['snr_before']:.2f} дБ\n"
        report += f"  • SNR после фильтрации: {info['snr_after']:.2f} дБ\n"
        report += f"  • Улучшение SNR: {info['snr_improvement']:+.2f} дБ\n\n"
        report += f"Точность фильтрации:\n"
        report += f"  • RMSE: {info['rmse']:.6e} ({quality})\n"
        report += f"  • MAE: {info['mae']:.6e}\n"
        report += f"  • Относительная ошибка: {info['relative_error']:.2f}%\n"
        report += f"  • Точность: {info['accuracy']:.2f}%\n\n"
        
        # Если есть метрики относительно эталона, добавляем их
        if 'reference_metrics' in info and info['reference_metrics'] is not None:
            ref_metrics = info['reference_metrics']
            report += f"МЕТРИКИ ОТНОСИТЕЛЬНО ЭТАЛОНА:\n"
            report += f"  • RMSE (эталон): {ref_metrics.get('rmse', 0):.6e}\n"
            report += f"  • MAE (эталон): {ref_metrics.get('mae', 0):.6e}\n"
            report += f"  • R² (эталон): {ref_metrics.get('r2', 0):.4f}\n"
            report += f"  • Относительная ошибка (эталон): {ref_metrics.get('relative_error', 0):.2f}%\n"
            report += f"  • Точность (эталон): {ref_metrics.get('accuracy', 0):.2f}%\n"
            report += f"  • Количество точек сравнения: {ref_metrics.get('n_points', 0)}\n\n"
        
        if info['accuracy'] >= 95:
            report += f"✅ Отличная точность фильтрации!\n"
        elif info['accuracy'] >= 90:
            report += f"✓ Хорошая точность фильтрации\n"
        else:
            report += f"⚠ Точность ниже ожидаемой\n"
        
        return report
            
    def on_detect_outliers(self) -> None:
        """Обнаружение выбросов"""
        if self.current_data is None:
            return

        try:
            outliers = detect_outliers(self.current_data.pressure, 'iqr', threshold=DEFAULT_OUTLIER_THRESHOLD)
            outlier_count = outliers.sum()
            
            self.show_info("Обнаружение выбросов", f"Найдено {outlier_count} выбросов")
            
            # Обновляем график
            self.on_plot_dimensionless_selected()
        except Exception as e:
            self.show_warning("Ошибка", f"Ошибка обнаружения выбросов: {str(e)}")
    
    def on_export_data(self) -> None:
        """Экспорт данных"""
        if self.current_data is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Экспорт данных", "", "CSV файлы (*.csv)")

        if file_path:
            try:
                # Создаем DataFrame для экспорта
                export_df = pd.DataFrame({
                    'Time': self.current_data.time,
                    'Pressure': self.current_data.pressure,
                    'FlowRate': self.current_data.flow_rate
                })
                export_df.to_csv(file_path, index=False)
                self.show_info("Экспорт", "Данные успешно экспортированы")
            except Exception as e:
                self.show_warning("Ошибка экспорта", f"Не удалось экспортировать данные: {str(e)}")
    
    def on_analyze_flow_regime(self) -> None:
        """Анализ режима течения"""
        if self.current_data is None:
            return
            
        analysis = analyze_flow_regime(self.current_data)
        
        result_text = f"""
Анализ режима течения:
Тип режима: {analysis.regime_type}
Уверенность: {analysis.confidence:.2f}
Характерное время: {analysis.characteristic_time or 'Не определено'}

Параметры:
{chr(10).join([f'{k}: {v}' for k, v in analysis.parameters.items()])}
"""
        self.show_info("Анализ режима течения", result_text)

    def on_compute_productivity_index(self) -> None:
        """Вычисление индекса продуктивности"""
        if self.current_data is None:
            return

        productivity = compute_productivity_index(self.current_data)

        result_text = f"""
Индекс продуктивности скважины:
Индекс продуктивности: {productivity['productivity_index']:.4f}
Эффективность ГРП: {productivity['fracture_efficiency']:.4f}
Средний дебит: {productivity['average_flow_rate']:.2f} м³/сут
Среднее давление: {productivity['average_pressure']:.2f} атм
Влияние скин-эффекта: {productivity['skin_impact']:.4f}
Общая длина трещин: {productivity['total_fracture_length']:.2f} м
"""

        self.show_info("Индекс продуктивности", result_text)

    def on_detect_flow_regime_transitions(self) -> None:
        """Обнаружение переходов режимов"""
        if self.current_data is None:
            return

        transitions = detect_flow_regime_transitions(self.current_data)

        if transitions:
            result_text = f"Найдено {len(transitions)} переходов режимов течения:\n\n"
            for i, trans in enumerate(transitions, 1):
                result_text += f"Переход {i}:\n"
                result_text += f"Время: {trans['time']:.2f} ч\n"
                result_text += f"Давление: {trans['pressure']:.2f} атм\n"
                result_text += f"Дебит: {trans['flow_rate']:.2f} м³/сут\n"
                result_text += f"Изменение производной: {trans['derivative_change']:.4f}\n\n"
        else:
            result_text = "Переходы режимов течения не обнаружены"

        self.show_info("Переходы режимов", result_text)

    def on_plot_bilinear(self) -> None:
        """Построение кривой билинейного течения"""
        if self.current_data is None or self.type_curves_widget is None:
            return

        time_range = np.logspace(TYPE_CURVE_TIME_MIN, TYPE_CURVE_TIME_MAX, TYPE_CURVE_N_POINTS)
        curves = generate_type_curves(self.current_data.skin, self.current_data.fractures_count,
                                     self.current_data.a_l_ratio, time_range)

        self.type_curves_widget.clear()
        time_curve, pressure_curve = curves['bilinear']
        self.type_curves_widget.plot(time_curve, pressure_curve, pen='b', name='Билинейное течение')
        self.type_curves_widget.setLabel('left', 'Давление, атм')
        self.type_curves_widget.setLabel('bottom', 'Время, ч')
        self.type_curves_widget.setTitle('Билинейное течение')

    def on_plot_linear(self) -> None:
        """Построение кривой линейного течения"""
        if self.current_data is None or self.type_curves_widget is None:
            return

        time_range = np.logspace(TYPE_CURVE_TIME_MIN, TYPE_CURVE_TIME_MAX, TYPE_CURVE_N_POINTS)
        curves = generate_type_curves(self.current_data.skin, self.current_data.fractures_count,
                                     self.current_data.a_l_ratio, time_range)

        self.type_curves_widget.clear()
        time_curve, pressure_curve = curves['linear']
        self.type_curves_widget.plot(time_curve, pressure_curve, pen='g', name='Линейное течение')
        self.type_curves_widget.setLabel('left', 'Давление, атм')
        self.type_curves_widget.setLabel('bottom', 'Время, ч')
        self.type_curves_widget.setTitle('Линейное течение')

    def on_plot_pseudoradial(self) -> None:
        """Построение кривой псевдорадиального течения"""
        if self.current_data is None or self.type_curves_widget is None:
            return

        time_range = np.logspace(TYPE_CURVE_TIME_MIN, TYPE_CURVE_TIME_MAX, TYPE_CURVE_N_POINTS)
        curves = generate_type_curves(self.current_data.skin, self.current_data.fractures_count,
                                     self.current_data.a_l_ratio, time_range)

        self.type_curves_widget.clear()
        time_curve, pressure_curve = curves['pseudoradial']
        self.type_curves_widget.plot(time_curve, pressure_curve, pen='r', name='Псевдорадиальное течение')
        self.type_curves_widget.setLabel('left', 'Давление, атм')
        self.type_curves_widget.setLabel('bottom', 'Время, ч')
        self.type_curves_widget.setTitle('Псевдорадиальное течение')

    def on_match_type_curves(self) -> None:
        """Сопоставление с эталонными кривыми"""
        if self.current_data is None:
            return

        match_result = match_type_curves(self.current_data)

        result_text = f"""
Сопоставление с эталонными кривыми:
Лучшее совпадение: {match_result.curve_type}
Качество совпадения: {match_result.match_quality:.2f}

Оцененные параметры:
{chr(10).join([f'{k}: {v}' for k, v in match_result.estimated_parameters.items()])}
"""

        self.show_info("Сопоставление кривых", result_text)

    def on_export_report(self) -> None:
        """Экспорт отчета"""
        if self.current_data is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Экспорт отчета", "", "Текстовые файлы (*.txt)")

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("ОТЧЕТ АНАЛИЗА ДАННЫХ ГРП\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Скважина: Skin={self.current_data.skin:.3f}, "
                           f"N={self.current_data.fractures_count}, "
                           f"a/L={self.current_data.a_l_ratio:.3f}\n\n")

                    # Анализ режима течения
                    analysis = analyze_flow_regime(self.current_data)
                    f.write(f"Режим течения: {analysis.regime_type}\n")
                    f.write(f"Уверенность: {analysis.confidence:.2f}\n")
                    f.write(f"Характерное время: {analysis.characteristic_time or 'Не определено'}\n\n")

                    # Индекс продуктивности
                    productivity = compute_productivity_index(self.current_data)
                    f.write("Индекс продуктивности:\n")
                    f.write(f"  Индекс продуктивности: {productivity['productivity_index']:.4f}\n")
                    f.write(f"  Эффективность ГРП: {productivity['fracture_efficiency']:.4f}\n")
                    f.write(f"  Средний дебит: {productivity['average_flow_rate']:.2f} м³/сут\n")
                    f.write(f"  Среднее давление: {productivity['average_pressure']:.2f} атм\n\n")

                self.show_info("Экспорт отчета", "Отчет успешно экспортирован")
            except Exception as e:
                self.show_warning("Ошибка экспорта", f"Не удалось экспортировать отчет: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec())
