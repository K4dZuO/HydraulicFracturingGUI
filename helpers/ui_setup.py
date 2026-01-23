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

import pyqtgraph as pg
from PySide6.QtWidgets import QGraphicsScene

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout


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


def attach_pg_to_widget(container: QWidget) -> pg.PlotItem:
    """
    Встраивает pyqtgraph в QWidget из .ui
    и возвращает PlotItem для рисования
    """
    if container is None:
        raise RuntimeError("Graph container widget not found")

    layout = container.layout()
    if layout is None:
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

    plot_widget = pg.PlotWidget()
    plot_widget.showGrid(x=True, y=True)

    layout.addWidget(plot_widget)

    return plot_widget.getPlotItem()


def setup_timeseries_tab(app: 'MyApp') -> None:
    app.P_graphic = attach_pg_to_widget(
        app.findChild(QWidget, "p_graphic")
    )
    app.Q_graphic = attach_pg_to_widget(
        app.findChild(QWidget, "q_graphic")
    )
    app.dim_plot = attach_pg_to_widget(
        app.findChild(QWidget, "dim_plot")
    )


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
