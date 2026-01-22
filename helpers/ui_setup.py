"""
Модуль для наполнения UI компонентов приложения.
Работает ТОЛЬКО с элементами, созданными в main_ui.ui
"""

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTableView
)
import pyqtgraph as pg

if TYPE_CHECKING:
    from main import MyApp


def setup_interface(app: 'MyApp') -> None:
    """
    Инициализация динамических элементов UI:
    - графики (pyqtgraph)
    - таблица данных
    """

    setup_timeseries_tab(app)
    setup_type_curves_tab(app)
    setup_data_tab(app)


# ------------------------------------------------------------------
# Безразмерные кривые
# ------------------------------------------------------------------

def setup_timeseries_tab(app: 'MyApp') -> None:
    # placeholders
    controls_placeholder = app.findChild(
        QWidget, "timeseries_controls_placeholder"
    )
    plot_placeholder = app.findChild(
        QWidget, "dimensionless_plot_placeholder"
    )

    # --- Левая панель управления ---
    # ВАЖНО: layout уже есть в .ui
    controls_layout = controls_placeholder.layout()
    if controls_layout is None:
        controls_layout = QVBoxLayout(controls_placeholder)

    # Все кнопки УЖЕ существуют в .ui,
    # здесь мы их не создаём, только используем позже в handlers

    # --- График ---
    plot_layout = plot_placeholder.layout()
    if plot_layout is None:
        plot_layout = QVBoxLayout(plot_placeholder)

    app.dimensionless_plot = pg.PlotWidget()
    app.dimensionless_plot.showGrid(x=True, y=True)
    app.dimensionless_plot.setLabel(
        'bottom', 'X безразмерный фильтрационный параметр'
    )
    app.dimensionless_plot.setLabel(
        'left', 'Y безразмерный ёмкостной параметр'
    )
    app.dimensionless_plot.setTitle("Безразмерные кривые МГРП")

    plot_layout.addWidget(app.dimensionless_plot)


# ------------------------------------------------------------------
# Эталонные кривые
# ------------------------------------------------------------------

def setup_type_curves_tab(app: 'MyApp') -> None:
    placeholder = app.findChild(
        QWidget, "type_curves_plot_placeholder"
    )

    layout = placeholder.layout()
    if layout is None:
        layout = QVBoxLayout(placeholder)

    app.type_curves_widget = pg.PlotWidget()
    app.type_curves_widget.setLogMode(True, True)
    app.type_curves_widget.showGrid(x=True, y=True)
    # app.type_curves_widget.setLabel('left', 'Дебит, м³/сут')
    # app.type_curves_widget.setLabel('bottom', 'Время, ч')

    layout.addWidget(app.type_curves_widget)


# ------------------------------------------------------------------
# Таблица данных (вкладка уже есть в .ui)
# ------------------------------------------------------------------

def setup_data_tab(app: 'MyApp') -> None:
    """
    Наполнение вкладки "Загруженные данные"
    """

    data_tab = app.findChild(QWidget, "data_tab")
    if data_tab is None:
        # если вкладку временно уберёшь из .ui — код не упадёт
        return

    layout = data_tab.layout()
    if layout is None:
        layout = QVBoxLayout(data_tab)

    app.data_table = QTableView()
    app.data_table.setAlternatingRowColors(True)
    app.data_table.setSelectionBehavior(
        QTableView.SelectionBehavior.SelectRows
    )
    app.data_table.horizontalHeader().setStretchLastSection(True)

    layout.addWidget(app.data_table)
