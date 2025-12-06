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


from ui import Ui_mainWindow

from helpers.parse_well_data import parse_well_data
from helpers.ml_methods import (detect_outliers)
from helpers.dimensionless_analysis import convert_to_dimensionless_curves, get_dimensionless_series, fit_xy_curve_coefficients
from helpers.dimensionless.filtration import SignalFilters, PhysicsConstraints, compute_snr
from helpers.dimensionless_plotting import plot_dimensionless_grouped
from helpers.dimensionless.filtration.utils import select_filter_method
from helpers.dimensionless_interpolating import DimensionlessCurveInterpolator
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
        
        # Создаем  интерфейс с вкладками
        setup_interface(self)
        
        # Добавляем отчёт в centralwidget под параметрами (вне вкладок)
        report_group = QGroupBox("Отчёт", self.centralwidget)
        report_group.setGeometry(20, 330, 300, 400)  # Под параметрами (последний на y=300)
        report_layout = QVBoxLayout(report_group)
        self.text_report = QTextEdit(report_group)
        self.text_report.setReadOnly(True)
        self.text_report.setPlaceholderText("Здесь появится отчёт...")
        self.text_report.setMinimumHeight(150)
        self.text_report.setMaximumHeight(500)
        report_layout.addWidget(self.text_report)
        
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
    
    def _perform_interpolation(self) -> Tuple[Dict[str, Any], Any]:
        """Выполняет интерполяцию безразмерных кривых"""
        params = self._get_params(self.current_data)
        
        # Конвертируем в безразмерные параметры
        dim_data = convert_to_dimensionless_curves(
            self.current_data.time, self.current_data.pressure, self.current_data.flow_rate, self.current_data.depression, params, x_mode='alt'
        )
        
        # Для интерполяции используем текущие параметры
        param_grid = np.array([[self.current_data.skin, self.current_data.fractures_count, self.current_data.a_l_ratio]])
        Y_grid = dim_data.Y
        P_curves = np.asarray([dim_data.pressure / (dim_data.delta_p_i if dim_data.delta_p_i != 0 else 1.0)])
        
        # Используем интерполятор безразмерных кривых для восстановления пропусков
        interp = DimensionlessCurveInterpolator(methods=INTERPOLATION_METHODS)
        interp.fit(param_grid, Y_grid, P_curves)
        
        # Предсказываем для тех же параметров (восстанавливаем пропуски)
        pred_series = interp.predict(
            skin=self.current_data.skin,
            N=self.current_data.fractures_count,
            a_L=self.current_data.a_l_ratio
        )
        
        # Восстанавливаем физические величины из безразмерных
        # pred_series имеет индекс Y_grid, нужно интерполировать обратно на time
        pressure_values = pred_series.values if hasattr(pred_series, 'values') else np.asarray(pred_series)
        
        # Определяем маску пропусков в исходных данных (до интерполяции)
        pressure_is_nan = self.current_data.pressure.isna()
        
        # Проверяем, что pred_series содержит достаточно точек
        if len(pressure_values) != len(self.current_data.time):
            # Если количество точек не совпадает, интерполируем на временную сетку
            Y_grid_values = pred_series.index.values
            # Интерполируем pD обратно на Y из dim_data, затем на time
            if len(pressure_values) > 1 and len(dim_data.Y) > 1:
                # Создаем интерполятор для маппинга Y_grid -> Y из данных
                interp_pd = interp1d(Y_grid_values, pressure_values, kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
                # Интерполируем на Y из данных
                pD_interp = interp_pd(dim_data.Y)
                pressure_values = pD_interp
            elif len(pressure_values) == 1:
                # Если только одна точка, дублируем её для всех временных точек
                pressure_values = np.full(len(self.current_data.time), pressure_values[0])
        
        # Убеждаемся, что pressure_values имеет правильную длину
        if len(pressure_values) != len(self.current_data.time):
            # Если все еще не совпадает, используем линейную интерполяцию по времени
            # Используем доступные точки для интерполяции
            valid_indices = np.arange(len(pressure_values))
            interp_temp = interp1d(valid_indices, pressure_values, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            target_indices = np.linspace(0, len(pressure_values) - 1, len(self.current_data.time))
            pressure_values = interp_temp(target_indices)
        
        # Восстанавливаем давление: заполняем только пропуски точно в точках времени t
        pressure_restored = self.current_data.pressure.copy()
        if pressure_is_nan.any():
            # Преобразуем pressure_values в массив
            pressure_values_array = np.asarray(pressure_values)
            
            # Убеждаемся, что длина совпадает
            if len(pressure_values_array) != len(self.current_data.time):
                # Если не совпадает, интерполируем точно на временную сетку
                # Используем Y_grid для интерполяции обратно на time
                if len(pressure_values_array) > 1 and len(dim_data.Y) > 1:
                    # Интерполируем pD по Y, затем маппим на time через Y
                    Y_grid_for_interp = pred_series.index.values if hasattr(pred_series, 'index') else np.arange(len(pressure_values_array))
                    interp_pd_final = interp1d(Y_grid_for_interp, pressure_values_array, kind='linear',
                                              bounds_error=False, fill_value='extrapolate')
                    # Интерполируем на Y из dim_data (который соответствует time)
                    pressure_values_array = interp_pd_final(dim_data.Y)
            
            # Создаем массив восстановленных значений давления
            pressure_interpolated = pressure_values_array * dim_data.delta_p_i
            
            # Заполняем только пропуски интерполированными значениями точно в точках времени t
            # Используем .values для маски, чтобы получить numpy array
            nan_mask = pressure_is_nan.values if hasattr(pressure_is_nan, 'values') else pressure_is_nan
            # Убеждаемся, что индексы совпадают
            if len(pressure_interpolated) == len(nan_mask):
                pressure_restored.loc[pressure_is_nan] = pressure_interpolated[nan_mask]
            else:
                # Если длины не совпадают, используем прямое индексирование
                pressure_restored.iloc[pressure_is_nan.values] = pressure_interpolated[pressure_is_nan.values]
        
        # ВАЖНО: НЕ изменяем исходные данные! Интерполированные значения используются только для отображения
        # Сохраняем интерполированные значения отдельно для использования в графиках
        # self.current_data.pressure остается неизменным - это исходные данные
        self.last_interpolated_pressure = pressure_restored  # Сохраняем для отображения, но не изменяем исходные данные
        
        # Получаем информацию о результатах интерполяции
        interp_info = interp.get_interpolation_info()
        
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
                    
                    # Получаем предсказанные значения в безразмерных координатах
                    Y_pred = dim_data.Y
                    P_pred = dim_data.pressure / (dim_data.delta_p_i if dim_data.delta_p_i != 0 else 1.0)
                    
                    # Эталонные значения
                    Y_ref = ref_dim.Y
                    P_ref = ref_dim.pressure / (ref_dim.delta_p_i if ref_dim.delta_p_i != 0 else 1.0)
                    
                    # Рассчитываем метрики с использованием случайных точек эталона
                    reference_metrics = interp.compare_with_reference(
                        Y_pred=Y_pred, P_pred=P_pred,
                        Y_ref=Y_ref, P_ref=P_ref,
                        n_random_points=100
                    )
                    
                    # Добавляем метрики в interp_info
                    interp_info['reference_metrics'] = reference_metrics
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
        report += f"  Обучающих примеров: {interp_info['n_samples']}\n"
        report += f"  Точек на безразмерной кривой: {interp_info['n_points']}\n\n"
        
        # Сравнение методов
        report += "СРАВНЕНИЕ МЕТОДОВ ИНТЕРПОЛЯЦИИ\n"
        report += "-" * REPORT_SEPARATOR_LENGTH + "\n"
        report += f"{'Метод':<42} {'RMSE':<12} {'Статус'}\n"
        report += "-" * REPORT_SEPARATOR_LENGTH + "\n"
        
        # Сортируем методы по RMSE
        sorted_methods = sorted(interp_info['rmse_scores'].items(), key=lambda x: x[1])
        
        for i, (method, rmse) in enumerate(sorted_methods, 1):
            method_display = method_names.get(method, method)
            marker = "✓ ВЫБРАН" if method == interp_info['best_method'] else f"#{i}"
            report += f"{method_display:<42} {rmse:<12.3f} {marker}\n"
        
        report += "-" * REPORT_SEPARATOR_LENGTH + "\n\n"
        
        # Итоговая информация
        best_rmse = interp_info['rmse_scores'].get(interp_info['best_method'], 0)
        report += "ИТОГОВЫЙ РЕЗУЛЬТАТ:\n"
        report += f"  Метод: {method_names.get(interp_info['best_method'], interp_info['best_method'])}\n"
        report += f"  RMSE: {best_rmse:.6e}\n"
        quality = self._get_quality_label(best_rmse)
        report += f"  Качество: {quality}\n"
        
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
            # Подсчитываем количество пропусков
            pressure_is_nan_mask = self.current_data.pressure.isna().values if hasattr(self.current_data.pressure, 'isna') else None
            n_nan_pressure = int(np.nansum(pressure_is_nan_mask)) if pressure_is_nan_mask is not None else self.current_data.pressure.isna().sum()
            n_nan_flow = self.current_data.flow_rate.isna().sum()
            
            # Сохраняем маску пропусков для подсветки на X-Y графике
            try:
                self.last_interpolated_mask_XY = pressure_is_nan_mask.copy() if pressure_is_nan_mask is not None else None
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
                'gp': 'Гауссовский процесс'
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
            
            # Проверяем, нужно ли подгонять только Y
            fit_only_y = hasattr(self, 'cb_fit_only_y') and self.cb_fit_only_y.isChecked()
            
            # Выполняем подгонку
            fit_result = fit_xy_curve_coefficients(X_data, Y_data, X_calc, Y_calc, fit_only_y=fit_only_y)
            
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
            report += f"✅ ЛУЧШИЕ КОЭФФИЦИЕНТЫ:\n"
            if fit_only_y:
                report += f"   a (по X) = {fit_result['a']:.4g} (фиксирован)\n"
            else:
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
            if fit_only_y:
                report += f"   X_fit = 0.00864 * k * h * ΔP / (μ * B * Q) (без изменений)\n"
            else:
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
            self.show_info("Подгонка выполнена", 
                         f"Коэффициенты: a={fit_result['a']:.3g}, b={fit_result['b']:.3g}, c={c_val:.3g}\n"
                         f"Точность: {fit_result['accuracy']:.2f}%\n"
                         f"Коэффициенты сохранены и будут применяться автоматически.")
            
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
            dim_data = convert_to_dimensionless_curves(
                current_item.time, current_item.pressure, current_item.flow_rate, current_item.depression, params, x_mode='alt'
            )

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
                # (но только если fit_only_y не было, иначе a = 1.0 уже установлен)
                if not (hasattr(self, 'cb_fit_only_y') and self.cb_fit_only_y.isChecked()):
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
                if not (hasattr(self, 'cb_fit_only_y') and self.cb_fit_only_y.isChecked()):
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
        """ML-фильтрация данных"""
        if self.current_data is None:
            return
        
        try:
            # Получаем параметры скважины
            params = self._get_params(self.current_data)
            
            # Конвертируем в безразмерные параметры
            dim_data = convert_to_dimensionless_curves(
                self.current_data.time,
                self.current_data.pressure,
                self.current_data.flow_rate,
                self.current_data.depression,
                params,
                x_mode='alt'
            )
            
            # Получаем безразмерные кривые
            series = get_dimensionless_series(dim_data)
            Y = series['Y']
            pD_original = series['pD']
            
            # Удаляем NaN значения для фильтрации
            valid_mask = np.isfinite(Y) & np.isfinite(pD_original)
            if not np.any(valid_mask):
                self.show_warning("Ошибка", "Нет валидных данных для фильтрации")
                return
            
            Y_clean = Y[valid_mask]
            pD_clean = pD_original[valid_mask]
            
            # Вычисляем исходные метрики
            snr_before = compute_snr(pD_clean)
            
            # Автоматический выбор и применение фильтра
            filtered_pD = SignalFilters.denoise(pD_clean, method=None, x=Y_clean)
            
            # Определяем, какой метод был выбран автоматически
            selected_method = select_filter_method(pD_clean, Y_clean)
            
            # Применяем физические ограничения
            filtered_pD = PhysicsConstraints.enforce_all(
                filtered_pD,
                x=Y_clean,
                monotonic=True,
                limit_curvature=True,
                remove_oscillations=True,
                asymptotic_fix=True
            )
            
            # Вычисляем метрики после фильтрации
            snr_after = compute_snr(filtered_pD)
            snr_improvement = snr_after - snr_before
            
            # Вычисляем RMSE и другие метрики
            rmse = np.sqrt(np.mean((pD_clean - filtered_pD) ** 2))
            mae = np.mean(np.abs(pD_clean - filtered_pD))
            
            # Вычисляем относительную ошибку и точность
            pD_range = np.max(pD_clean) - np.min(pD_clean)
            relative_error = (rmse / pD_range * 100) if pD_range > 0 else 0.0
            accuracy = max(0, 100 - relative_error)
            
            # Восстанавливаем давление из отфильтрованного pD
            delta_p_i = dim_data.delta_p_i
            if delta_p_i > 0:
                try:
                    interp_func = interp1d(Y_clean, filtered_pD, kind='linear', 
                                         bounds_error=False, fill_value='extrapolate')
                    pD_filtered_interp = interp_func(Y)
                    
                    pressure_initial = self.current_data.pressure.iloc[0] if len(self.current_data.pressure) > 0 else 0
                    delta_p_filtered = pD_filtered_interp * delta_p_i
                    pressure_filtered = pressure_initial - delta_p_filtered
                    
                    self.current_data.pressure = pd.Series(
                        pressure_filtered, 
                        index=self.current_data.pressure.index
                    )
                except Exception as interp_error:
                    print(f"Предупреждение: не удалось восстановить давление: {interp_error}")
            
            # Сохраняем информацию о фильтрации для вывода в отчёте
            method_names = {
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
                'n_points': len(Y_clean)
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
