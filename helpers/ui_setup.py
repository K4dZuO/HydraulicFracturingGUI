"""
Модуль для настройки UI компонентов приложения.
Содержит функции для создания вкладок и виджетов.
"""

from typing import TYPE_CHECKING
from PySide6.QtWidgets import (QLabel, QTableView, QWidget, QVBoxLayout, 
                               QHBoxLayout, QTabWidget, QTextEdit, QGroupBox, 
                               QGridLayout, QCheckBox, QPushButton, QComboBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem
import pyqtgraph as pg

from config import (TAB_WIDGET_X, TAB_WIDGET_Y, TAB_WIDGET_WIDTH, 
                   TAB_WIDGET_HEIGHT, LEFT_PANEL_MAX_WIDTH)

if TYPE_CHECKING:
    from main import MyApp


def setup_interface(app: 'MyApp') -> None:
    """Создает интерфейс с вкладками для анализа ГРП"""
    # Основной контейнер с вкладками
    app.tab_widget = QTabWidget(app.centralwidget)
    app.tab_widget.setGeometry(TAB_WIDGET_X, TAB_WIDGET_Y, TAB_WIDGET_WIDTH, TAB_WIDGET_HEIGHT)
    
    # Вкладка 1: Временные ряды
    app.timeseries_tab = setup_timeseries_tab(app)
    app.tab_widget.addTab(app.timeseries_tab, "Безразмерные кривые")
    
    # Вкладка 2: Анализ ГРП
    app.grp_tab = QWidget()
    app.tab_widget.addTab(app.grp_tab, "Анализ ГРП")
    setup_grp_tab(app)
    
    # Вкладка 3: Эталонные кривые
    app.type_curves_tab = QWidget()
    app.tab_widget.addTab(app.type_curves_tab, "Эталонные кривые")
    setup_type_curves_tab(app)
    
    # Вкладка 4: Результаты анализа
    app.results_tab = QWidget()
    app.tab_widget.addTab(app.results_tab, "Результаты")
    setup_results_tab(app)


def setup_timeseries_tab(app: 'MyApp') -> QWidget:
    """Вкладка для безразмерных графиков в стиле Kappa Sapphire."""
    tab = QWidget()
    
    # ОСНОВНОЙ ЛАЙАУТ С РАЗДЕЛЕНИЕМ ПО ГОРИЗОНТАЛИ
    main_layout = QHBoxLayout(tab)
    main_layout.setContentsMargins(5, 5, 5, 5)  # минимальные отступы

    # --- ЛЕВАЯ ПАНЕЛЬ: управление ---
    left_panel = QWidget()
    left_layout = QVBoxLayout(left_panel)
    left_layout.setContentsMargins(0, 0, 5, 0)
    left_panel.setMaximumWidth(LEFT_PANEL_MAX_WIDTH)  # Возвращаем обычную ширину

    # --- Загрузка файлов ---
    load_group = QGroupBox("Загрузка данных")
    load_layout = QGridLayout(load_group)
    app.load_validation_button = QPushButton("Загрузить файл для проверки")
    load_layout.addWidget(app.load_validation_button, 0, 0)
    left_layout.addWidget(load_group)
    
    # --- Управление моделью интерполяции ---
    model_group = QGroupBox("Модель интерполяции")
    model_layout = QGridLayout(model_group)
    app.train_model_btn = QPushButton("Обучить модель")
    app.save_model_btn = QPushButton("Сохранить модель")
    app.load_model_btn = QPushButton("Загрузить модель")
    app.model_status_label = QLabel("Модель не обучена")
    model_layout.addWidget(app.train_model_btn, 0, 0)
    model_layout.addWidget(app.save_model_btn, 0, 1)
    model_layout.addWidget(app.load_model_btn, 1, 0)
    model_layout.addWidget(app.model_status_label, 1, 1)
    left_layout.addWidget(model_group)

    # --- Кнопки анализа СВЕРХУ ---
    buttons_group = QGroupBox("Управление")
    buttons_layout = QGridLayout(buttons_group)
    app.plot_btn = QPushButton("Построить график")
    app.interp_btn = QPushButton("Интерполяция")
    app.ml_filter_btn = QPushButton("ML фильтрация")
    app.outlier_btn = QPushButton("Обнаружить выбросы")
    app.export_btn = QPushButton("Экспорт данных")
    app.extrapolate_btn = QPushButton("Экстраполировать X-Y")
    app.fit_xy_btn = QPushButton("Подогнать расчётную кривую")
    app.cb_fit_only_y = QCheckBox("Подгонять Y")

    buttons_layout.addWidget(app.plot_btn, 0, 0)
    buttons_layout.addWidget(app.interp_btn, 0, 1)
    buttons_layout.addWidget(app.ml_filter_btn, 1, 0)
    buttons_layout.addWidget(app.outlier_btn, 1, 1)
    buttons_layout.addWidget(app.export_btn, 2, 0)
    buttons_layout.addWidget(app.extrapolate_btn, 2, 1)
    buttons_layout.addWidget(app.fit_xy_btn, 3, 0)
    buttons_layout.addWidget(app.cb_fit_only_y, 3, 1)
    left_layout.addWidget(buttons_group)

    # --- Выбор скважины ---
    well_group = QGroupBox("Выбор скважины")
    well_layout = QVBoxLayout(well_group)
    app.well_combo_dim = QComboBox()
    well_layout.addWidget(app.well_combo_dim)
    left_layout.addWidget(well_group)

    # --- Панель чекбоксов ---
    controls_group = QGroupBox("Отображение (группы)")
    controls_layout = QVBoxLayout(controls_group)

    # Реальные параметры
    app.grp_real = QGroupBox("Реальные параметры")
    real_lay = QVBoxLayout(app.grp_real)
    app.cb_real_p = QCheckBox("P(t)")
    app.cb_real_q = QCheckBox("Q(t)")
    real_lay.addWidget(app.cb_real_p)
    real_lay.addWidget(app.cb_real_q)
    controls_layout.addWidget(app.grp_real)

    # Безразмерные (log-log)
    app.grp_dim = QGroupBox("Безразмерные (log-log)")
    dim_lay = QVBoxLayout(app.grp_dim)
    app.cb_dim_pD = QCheckBox("pD(Y)")
    app.cb_dim_dpD = QCheckBox("dpD/dlogY")
    app.cb_dim_tD = QCheckBox("tD (≈Y)")
    app.cb_dim_CD = QCheckBox("CD (ёмкость)")
    app.cb_XY_plot = QCheckBox("X-Y график (из данных)")
    app.cb_calc_XY = QCheckBox("Отобразить расчётные X и Y")
    dim_lay.addWidget(app.cb_dim_pD)
    dim_lay.addWidget(app.cb_dim_dpD)
    dim_lay.addWidget(app.cb_dim_tD)
    dim_lay.addWidget(app.cb_dim_CD)
    dim_lay.addWidget(app.cb_XY_plot)
    dim_lay.addWidget(app.cb_calc_XY)
    controls_layout.addWidget(app.grp_dim)

    # Спец-пространства и типовые кривые
    app.grp_special = QGroupBox("Спец-пространства/типовые")
    sp_lay = QVBoxLayout(app.grp_special)
    app.cb_gfunc = QCheckBox("G-функция (Nolte)")
    app.cb_mbt = QCheckBox("Время материального баланса")
    app.cb_type_gry = QCheckBox("Билинейный режим")
    app.cb_type_cinco = QCheckBox("Линейный режим течения")
    app.cb_type_valko = QCheckBox("Псевдорадиальный режим")
    sp_lay.addWidget(app.cb_gfunc)
    sp_lay.addWidget(app.cb_mbt)
    sp_lay.addWidget(app.cb_type_gry)
    sp_lay.addWidget(app.cb_type_cinco)
    sp_lay.addWidget(app.cb_type_valko)
    controls_layout.addWidget(app.grp_special)

    # По умолчанию НЕ включаем никакие графики
    # app.cb_dim_pD.setChecked(True)  # Убрано автоматическое включение
    left_layout.addWidget(controls_group)
    
    # Добавляем кнопку сброса графиков
    reset_button_layout = QHBoxLayout()
    app.reset_plots_btn = QPushButton("Очистить график")
    app.reset_plots_btn.setStyleSheet("""
        QPushButton {
            background-color: #e74c3c;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #c0392b;
        }
        QPushButton:pressed {
            background-color: #a93226;
        }
    """)
    reset_button_layout.addWidget(app.reset_plots_btn)
    left_layout.addLayout(reset_button_layout)

    # Растягиваем вниз
    left_layout.addStretch()

    # --- ПРАВАЯ ПАНЕЛЬ: график ---
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0)
    
    # График занимает всё доступное пространство
    app.dimensionless_plot = pg.PlotWidget()
    app.dimensionless_plot.showGrid(x=True, y=True)
    
    app.dimensionless_plot.setLabel('bottom', 'X  безразмерный фильтрационный параметр')  # ← СНИЗУ X
    app.dimensionless_plot.setLabel('left', '•	Y  безразмерный ёмкостной параметр')  # ← СЛЕВА Y
    app.dimensionless_plot.setTitle("Безразмерные кривые МГРП")
    
    right_layout.addWidget(app.dimensionless_plot)

    # --- Собираем основной интерфейс ---
    main_layout.addWidget(left_panel)
    main_layout.addWidget(right_panel, stretch=1)  # график растягивается

    tab.setLayout(main_layout)
    return tab


def setup_grp_tab(app: 'MyApp') -> None:
    """Настройка вкладки анализа ГРП"""
    layout = QVBoxLayout(app.grp_tab)
    
    # Панель параметров ГРП
    params_group = QGroupBox("Параметры ГРП")
    params_layout = QGridLayout(params_group)
    
    # Отображение параметров из загруженных данных
    app.skin_value_label = QLabel("Skin: -")
    app.thickness_value_label = QLabel("Толщина: -")
    app.fractures_value_label = QLabel("Трещины: -")
    app.fracture_width_label = QLabel("Ширина трещины: -")
    app.fracture_length_label = QLabel("Длина трещины: -")
    app.al_ratio_label = QLabel("a/L: -")
    
    params_layout.addWidget(app.skin_value_label, 0, 0)
    params_layout.addWidget(app.thickness_value_label, 0, 1)
    params_layout.addWidget(app.fractures_value_label, 0, 2)
    params_layout.addWidget(app.fracture_width_label, 1, 0)
    params_layout.addWidget(app.fracture_length_label, 1, 1)
    params_layout.addWidget(app.al_ratio_label, 1, 2)
    
    layout.addWidget(params_group)
    
    # Кнопки анализа
    analysis_group = QGroupBox("Анализ")
    analysis_layout = QHBoxLayout(analysis_group)
    
    app.flow_regime_btn = QPushButton("Анализ режима течения")
    app.productivity_btn = QPushButton("Индекс продуктивности")
    app.transitions_btn = QPushButton("Переходы режимов")
    
    analysis_layout.addWidget(app.flow_regime_btn)
    analysis_layout.addWidget(app.productivity_btn)
    analysis_layout.addWidget(app.transitions_btn)
    
    layout.addWidget(analysis_group)
    
    # Область для результатов
    app.grp_results_text = QTextEdit()
    app.grp_results_text.setMaximumHeight(300)
    layout.addWidget(app.grp_results_text)


def setup_type_curves_tab(app: 'MyApp') -> None:
    """Настройка вкладки эталонных кривых"""
    layout = QVBoxLayout(app.type_curves_tab)
    
    # Панель управления
    controls_group = QGroupBox("Эталонные кривые")
    controls_layout = QHBoxLayout(controls_group)
    
    app.bilinear_btn = QPushButton("Билинейное течение")
    app.linear_btn = QPushButton("Линейное течение")
    app.pseudoradial_btn = QPushButton("Псевдорадиальное течение")
    app.match_curves_btn = QPushButton("Сопоставить с данными")
    
    controls_layout.addWidget(app.bilinear_btn)
    controls_layout.addWidget(app.linear_btn)
    controls_layout.addWidget(app.pseudoradial_btn)
    controls_layout.addWidget(app.match_curves_btn)
    
    layout.addWidget(controls_group)
    
    # График эталонных кривых
    if pg is not None:
        app.type_curves_widget = pg.PlotWidget()
        app.type_curves_widget.setLabel('left', 'Дебит, м³/сут')
        app.type_curves_widget.setLabel('bottom', 'Время, ч')
        app.type_curves_widget.setLogMode(True, True)  # Log-log масштаб
        app.type_curves_widget.showGrid(x=True, y=True)
        layout.addWidget(app.type_curves_widget)
    else:
        app.type_curves_widget = None
        layout.addWidget(QLabel("pyqtgraph не установлен"))


def setup_results_tab(app: 'MyApp') -> None:
    """Настройка вкладки результатов анализа"""
    layout = QVBoxLayout(app.results_tab)
    
    # Область для отображения результатов
    app.results_text = QTextEdit()
    app.results_text.setReadOnly(True)
    layout.addWidget(app.results_text)
    
    # Кнопка экспорта отчета
    app.export_report_btn = QPushButton("Экспорт отчета")
    layout.addWidget(app.export_report_btn)


def setup_data_tab(app: 'MyApp') -> QWidget:
    """Вкладка для просмотра загруженных данных (только чтение)"""
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Таблица для отображения данных
    app.data_table = QTableView()
    app.data_table.setAlternatingRowColors(True)
    app.data_table.setSelectionBehavior(app.data_table.SelectionBehavior.SelectRows)
    app.data_table.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(app.data_table)

    # Текстовое поле для информации (например, имя файла или статистика)
    app.data_info_label = QLabel("Здесь появится информация о загруженных данных")
    app.data_info_label.setAlignment(Qt.AlignLeft)
    layout.addWidget(app.data_info_label)

    tab.setLayout(layout)
    return tab

