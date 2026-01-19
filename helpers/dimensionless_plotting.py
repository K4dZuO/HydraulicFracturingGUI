"""
Модуль для построения графиков безразмерных кривых МГРП
Поддерживает разные плоскости отображения для разных групп графиков
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Optional, Union
import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen, mkBrush
from helpers.dimensionless_analysis import (
    DimensionlessParameters, 
    convert_to_dimensionless_curves,
)


def plot_dimensionless_grouped(plot_widget: PlotWidget,
                               dim_data: DimensionlessParameters,
                               time: pd.Series,
                               pressure: pd.Series,
                               flow_rate: pd.Series,
                               checked_groups: Dict[str, bool],
                               validation_data: Optional[Dict] = None,
                               X_data: Optional[pd.Series] = None,
                               Y_data: Optional[pd.Series] = None,
                               show_calculated_XY: bool = False,
                               interpolated_mask_XY: Optional[np.ndarray] = None,
                               extrapolated_XY: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
    """
    Отображает графики, сгруппированные по плоскостям отображения.
    
    Args:
        plot_widget: Виджет графика PyQtGraph
        dim_data: Безразмерные данные
        time: Временной ряд
        pressure: Давление
        flow_rate: Дебит
        checked_groups: Словарь с флагами выбранных групп:
            - 'real_params': Реальные параметры (P(t), Q(t))
            - 'dimensionless': Безразмерные (pD, dpD/dlogY, tD, CD)
            - 'type_curves': Типовые кривые
            - 'special': Специальные пространства (G-функция, MBT)
        validation_data: Данные для валидации (опционально)
    """
    plot_widget.clear()
    
    # Определяем, какие группы выбраны
    has_real = checked_groups.get('real_params', False)
    # has_dim включает безразмерные кривые, X-Y график из данных и расчётные X-Y
    has_dim = (checked_groups.get('dimensionless', False) or 
               checked_groups.get('cb_XY_plot', False) or 
               show_calculated_XY)
    has_type = checked_groups.get('type_curves', False)
    has_special = checked_groups.get('special', False)
    
    # Безразмерные величины
    pD = dim_data.pressure / (dim_data.delta_p_i if dim_data.delta_p_i != 0 else 1.0)
    qD = dim_data.flow_rate / (dim_data.Q if dim_data.Q != 0 else 1.0)
    
    # ГРУППА 1: Реальные параметры - плоскость (t, P) или (t, Q)
    # Обычные оси (не логарифмические)
    if has_real:
        plot_widget.setLogMode(x=False, y=False)
        plot_widget.setLabel('bottom', 'Время, ч')
        
        if checked_groups.get('cb_real_p', False):
            plot_widget.setLabel('left', 'Давление, атм')
            # Используем connect='finite' для правильного отображения пропусков
            plot_widget.plot(time.values, pressure.values,
                            pen=pg.mkPen(color=(200, 50, 50), width=2),
                            name="P(t)", 
                            connect='finite')
        
        if checked_groups.get('cb_real_q', False):
            if checked_groups.get('cb_real_p', False):
                # Если уже есть давление, используем правую ось или переключаем
                plot_widget.setLabel('left', 'Давление / Дебит')
            else:
                plot_widget.setLabel('left', 'Дебит, м³/сут')
            # Используем connect='finite' для правильного отображения пропусков
            plot_widget.plot(time.values, flow_rate.values,
                            pen=pg.mkPen(color=(50, 150, 50), width=2),
                            name="Q(t)", 
                            connect='finite')
        
        plot_widget.setTitle("Реальные параметры скважины")
        plot_widget.addLegend()
        return  # Реальные параметры в своем пространстве
    
    # ГРУППА 2: Безразмерные кривые - плоскость (X, pD/qD/tD/CD)
    # Все безразмерные кривые отображаются в плоскости X-Y
    if has_dim:
        # Всегда вычисляем расчётные X и Y из dim_data для независимого отображения
        X_calc = dim_data.X.astype(float)
        Y_calc = dim_data.Y.astype(float)
        
        # Используем X и Y из данных, если они есть, иначе используем рассчитанные из dim_data
        # (как в example.py - напрямую из convert_to_dimensionless_curves, без препроцессора)
        if X_data is not None and Y_data is not None:
            # Преобразуем в numpy массивы (без ограничения снизу, так как не log-log)
            X = X_data.values.astype(float)
            Y = Y_data.values.astype(float)
        else:
            # Используем напрямую рассчитанные X и Y из dim_data (без препроцессора, как в example.py)
            X = dim_data.X.astype(float)
            Y = dim_data.Y.astype(float)
        
        # --- Проверка и нормализация диапазонов ---
        # Убираем NaN
        pD = pD.astype(float)
        qD = qD.astype(float)

        # Сохраняем оригинальные X и Y для графика X-Y (без нормализации, как в example.py)
        X_original = X.copy()
        Y_original = Y.copy()
        
        # Нормализация к 1 при необходимости (чтобы избежать вылетов) - только для других графиков
        def normalize_if_flat(arr):
            rng = np.nanmax(arr) - np.nanmin(arr)
            if not np.isfinite(rng) or rng < 1e-6:
                arr = arr / (np.nanmax(arr) if np.nanmax(arr) != 0 else 1.0)
            return arr

        # Нормализуем только для других графиков (pD, dpD, tD, CD), но не для X-Y
        #X = normalize_if_flat(X)
        #Y = normalize_if_flat(Y)
        #pD = normalize_if_flat(pD)
        #qD = normalize_if_flat(qD)

        # Для производных – фильтруем шумы и NaN
        if np.any(np.isnan(pD)) or np.any(np.isnan(Y)) or np.any(np.isnan(qD)):
            mask_valid = (~np.isnan(pD)) & (~np.isnan(Y)) & (~np.isnan(qD))
            pD = pD[mask_valid]
            qD = qD[mask_valid]
            Y = Y[mask_valid]
            X = X[mask_valid]

        # Гарантируем, что диапазон данных корректен
        if np.allclose(np.nanmin(X), np.nanmax(X)) or np.allclose(np.nanmin(pD), np.nanmax(pD)):
            print("⚠️ Предупреждение: диапазон X или pD слишком узкий для отображения")

        plot_widget.setLogMode(x=False, y=False)
        plot_widget.setLabel('bottom', 'X (безразмерный фильтрационный параметр)')
        plot_widget.setLabel('left', 'Безразмерный параметр')
        plot_widget.setTitle("Безразмерные кривые МГРП")
        
        # Подготавливаем маску для интерполированных точек (если есть)
        interp_mask_for_plot = None
        if interpolated_mask_XY is not None:
            # Применяем маску к нормализованным X и Y (которые используются для графиков)
            if len(interpolated_mask_XY) >= len(X):
                interp_mask_for_plot = interpolated_mask_XY[:len(X)]
            else:
                # Если маска короче, расширяем её
                interp_mask_for_plot = np.zeros(len(X), dtype=bool)
                interp_mask_for_plot[:len(interpolated_mask_XY)] = interpolated_mask_XY
        
        # Строим только выбранные графики
        if checked_groups.get('cb_dim_pD', False):
            # Используем отфильтрованные массивы X и pD
            mask = np.isfinite(X) & np.isfinite(pD)
            if np.any(mask):
                X_plot = X[mask]
                pD_plot = pD[mask]
                
                # Если есть маска интерполированных точек, выделяем их
                if interp_mask_for_plot is not None and len(interp_mask_for_plot) >= len(X):
                    interp_mask_plot = interp_mask_for_plot[mask]
                    # Исходные точки
                    if np.any(~interp_mask_plot):
                        plot_widget.plot(X_plot[~interp_mask_plot], pD_plot[~interp_mask_plot],
                                        pen=pg.mkPen(color=(200, 50, 50), width=2),
                                        name="pD(X)",
                                        connect='finite')
                    # Интерполированные точки
                    if np.any(interp_mask_plot):
                        plot_widget.plot(X_plot[interp_mask_plot], pD_plot[interp_mask_plot],
                                        pen=pg.mkPen(color=(255, 100, 100), width=2),
                                        symbol='o', symbolSize=7,
                                        symbolBrush=pg.mkBrush(255, 100, 100, 220),
                                        name="pD(X) (восстановлено)",
                                        connect='finite')
                else:
                    plot_widget.plot(X_plot, pD_plot,
                                    pen=pg.mkPen(color=(200, 50, 50), width=2),
                                    name="pD(X)",
                                    connect='finite')
        
        if checked_groups.get('cb_dim_dpD', False):
            # Вычисляем производную на отфильтрованных данных
            #Yc = np.clip(Y, 1e-30, None)
            Yc = Y
            # Для обычного графика используем производную по Y, а не по log(Y)
            dpdY = np.gradient(pD, Yc)
            mask = np.isfinite(X) & np.isfinite(dpdY)
            if np.any(mask):
                X_plot = X[mask]
                dpdY_plot = dpdY[mask]
                
                # Если есть маска интерполированных точек, выделяем их
                if interp_mask_for_plot is not None and len(interp_mask_for_plot) >= len(X):
                    interp_mask_plot = interp_mask_for_plot[mask]
                    # Исходные точки
                    if np.any(~interp_mask_plot):
                        plot_widget.plot(X_plot[~interp_mask_plot], dpdY_plot[~interp_mask_plot],
                                        pen=pg.mkPen(color=(150, 0, 150), width=2),
                                        name="dpD/dY(X)",
                                        connect='finite')
                    # Интерполированные точки
                    if np.any(interp_mask_plot):
                        plot_widget.plot(X_plot[interp_mask_plot], dpdY_plot[interp_mask_plot],
                                        pen=pg.mkPen(color=(255, 100, 100), width=2),
                                        symbol='o', symbolSize=7,
                                        symbolBrush=pg.mkBrush(255, 100, 100, 220),
                                        name="dpD/dY(X) (восстановлено)",
                                        connect='finite')
                else:
                    plot_widget.plot(X_plot, dpdY_plot,
                                    pen=pg.mkPen(color=(150, 0, 150), width=2),
                                    name="dpD/dY(X)",
                                    connect='finite')
        
        if checked_groups.get('cb_dim_tD', False):
            # Используем отфильтрованные массивы X и Y
            mask = np.isfinite(X) & np.isfinite(Y)
            if np.any(mask):
                X_plot = X[mask]
                Y_plot = Y[mask]
                
                # Если есть маска интерполированных точек, выделяем их
                if interp_mask_for_plot is not None and len(interp_mask_for_plot) >= len(X):
                    interp_mask_plot = interp_mask_for_plot[mask]
                    # Исходные точки
                    if np.any(~interp_mask_plot):
                        plot_widget.plot(X_plot[~interp_mask_plot], Y_plot[~interp_mask_plot],
                                        pen=pg.mkPen(color=(0, 120, 200), width=2),
                                        name="tD (Y)",
                                        connect='finite')
                    # Интерполированные точки
                    if np.any(interp_mask_plot):
                        plot_widget.plot(X_plot[interp_mask_plot], Y_plot[interp_mask_plot],
                                        pen=pg.mkPen(color=(255, 100, 100), width=2),
                                        symbol='o', symbolSize=7,
                                        symbolBrush=pg.mkBrush(255, 100, 100, 220),
                                        name="tD (Y) (восстановлено)",
                                        connect='finite')
                else:
                    plot_widget.plot(X_plot, Y_plot,
                                    pen=pg.mkPen(color=(0, 120, 200), width=2),
                                    name="tD (Y)",
                                    connect='finite')
        
        if checked_groups.get('cb_dim_CD', False):
            # Вычисляем CD на отфильтрованных данных
            Yc = np.clip(np.abs(Y), 1e-30, None)
            pD_clip = np.clip(np.abs(pD), 1e-30, None)
            
            # Для обычного графика используем производную по Y
            dpdY = np.gradient(pD_clip, Yc)
            CD = np.abs(Yc * dpdY)
            CD = np.clip(CD, 1e-10, 1e10)  # ограничиваем диапазон
            CD /= np.nanmax(CD) if np.nanmax(CD) != 0 else 1  # нормализация

            mask = np.isfinite(X) & np.isfinite(CD)
            if np.any(mask):
                X_plot = X[mask]
                CD_plot = CD[mask]
                
                # Если есть маска интерполированных точек, выделяем их
                if interp_mask_for_plot is not None and len(interp_mask_for_plot) >= len(X):
                    interp_mask_plot = interp_mask_for_plot[mask]
                    # Исходные точки
                    if np.any(~interp_mask_plot):
                        plot_widget.plot(X_plot[~interp_mask_plot], CD_plot[~interp_mask_plot],
                                        pen=pg.mkPen(color=(0, 180, 80), width=2),
                                        name="CD(X)",
                                        connect='finite')
                    # Интерполированные точки
                    if np.any(interp_mask_plot):
                        plot_widget.plot(X_plot[interp_mask_plot], CD_plot[interp_mask_plot],
                                        pen=pg.mkPen(color=(255, 100, 100), width=2),
                                        symbol='o', symbolSize=7,
                                        symbolBrush=pg.mkBrush(255, 100, 100, 220),
                                        name="CD(X) (восстановлено)",
                                        connect='finite')
                else:
                    plot_widget.plot(X_plot, CD_plot,
                                    pen=pg.mkPen(color=(0, 180, 80), width=2),
                                    name="CD(X)",
                                    connect='finite')
        
        # Отображаем X-Y график (пары точек X-Y из данных), если установлен флаг
        if checked_groups.get('cb_XY_plot', False):
            if X_data is not None and Y_data is not None:
                # Используем исходные данные напрямую, без нормализации
                # Убеждаемся, что индексы совпадают
                X_raw = X_data.values.astype(float)
                Y_raw = Y_data.values.astype(float)
                
                # Проверяем, что данные имеют одинаковую длину
                min_len = min(len(X_raw), len(Y_raw))
                if min_len > 0:
                    X_raw = X_raw[:min_len]
                    Y_raw = Y_raw[:min_len]
                    
                    mask_xy = np.isfinite(X_raw) & np.isfinite(Y_raw)
                    if np.any(mask_xy):
                        # Убеждаемся, что у нас есть несколько точек
                        X_plot = X_raw[mask_xy]
                        Y_plot = Y_raw[mask_xy]
                        if len(X_plot) > 0:
                            # Если есть маска восстановленных точек, подсветим их
                            if interpolated_mask_XY is not None:
                                # Убеждаемся, что маска имеет правильную длину
                                if len(interpolated_mask_XY) >= len(X_raw):
                                    # Применяем маску к исходным данным (до фильтрации по isfinite)
                                    # Маска указывает на позиции, где были пропуски ДО интерполяции
                                    interp_mask_full = interpolated_mask_XY[:len(X_raw)]
                                    # Применяем маску к отфильтрованным данным
                                    interp_mask = interp_mask_full[mask_xy]
                                    
                                    # Исходные точки (не были пропущены)
                                    if np.any(~interp_mask):
                                        plot_widget.plot(X_plot[~interp_mask], Y_plot[~interp_mask],
                                                         pen=pg.mkPen(color=(100, 150, 255), width=2),
                                                         symbol='o', symbolSize=6,
                                                         symbolBrush=pg.mkBrush(100, 150, 255, 200),
                                                         name="X-Y (исходные)",
                                                         connect='finite')
                                    # Восстановленные точки (были пропущены и восстановлены)
                                    if np.any(interp_mask):
                                        plot_widget.plot(X_plot[interp_mask], Y_plot[interp_mask],
                                                         pen=pg.mkPen(color=(255, 100, 100), width=2),
                                                         symbol='o', symbolSize=7,
                                                         symbolBrush=pg.mkBrush(255, 100, 100, 220),
                                                         name="X-Y (восстановлено)",
                                                         connect='finite')
                                else:
                                    # Маска короче данных - используем как есть
                                    plot_widget.plot(X_plot, Y_plot,
                                                     pen=pg.mkPen(color=(100, 150, 255), width=2),
                                                     symbol='o', symbolSize=5,
                                                     name="X-Y (из данных)",
                                                     connect='finite')
                            else:
                                # Нет маски - отображаем все точки одним цветом
                                plot_widget.plot(X_plot, Y_plot,
                                                 pen=pg.mkPen(color=(100, 150, 255), width=2),
                                                 symbol='o', symbolSize=5,
                                                 name="X-Y (из данных)",
                                                 connect='finite')
            # Если X_data и Y_data отсутствуют, график "X-Y (из данных)" не отображается
            # Расчётные X и Y можно отобразить независимо через чекбокс "Отобразить расчётные X и Y"

            # Экстраполированные X-Y, если переданы
            
        if checked_groups.get('cb_calc_XY', False):
            if extrapolated_XY is not None:
                try:
                    X_ext, Y_ext = extrapolated_XY
                    if X_ext is not None and Y_ext is not None and len(X_ext) > 0:
                        plot_widget.plot(np.asarray(X_ext, dtype=float), np.asarray(Y_ext, dtype=float),
                                         pen=pg.mkPen(color=(255, 150, 50), width=2, style=pg.QtCore.Qt.DotLine),
                                         symbol=None,
                                         name="X-Y (экстраполяция)",
                                         connect='finite')
                except Exception:
                    pass
        
        # Отображаем расчётные X и Y независимо от других графиков, если установлен флаг
        # X_calc и Y_calc всегда содержат ненормализованные значения из dim_data
        if show_calculated_XY and X_calc is not None and Y_calc is not None:
            # Отображаем расчётные значения пунктирной линией
            mask_calc = np.isfinite(X_calc) & np.isfinite(Y_calc)
            if np.any(mask_calc):
                X_calc_plot = X_calc[mask_calc]
                Y_calc_plot = Y_calc[mask_calc]
                
                # Если есть маска интерполированных точек, выделяем их
                if interpolated_mask_XY is not None and len(interpolated_mask_XY) >= len(X_calc):
                    interp_mask_calc = interpolated_mask_XY[:len(X_calc)][mask_calc]
                    
                    # Исходные точки (не были пропущены)
                    if np.any(~interp_mask_calc):
                        plot_widget.plot(X_calc_plot[~interp_mask_calc], Y_calc_plot[~interp_mask_calc],
                                        pen=pg.mkPen(color=(200, 200, 0), width=1, 
                                                    style=pg.QtCore.Qt.DashLine),
                                        symbol='+', symbolSize=8,
                                        name="X-Y (расчётные)")
                    
                    # Интерполированные точки (были пропущены и восстановлены)
                    if np.any(interp_mask_calc):
                        plot_widget.plot(X_calc_plot[interp_mask_calc], Y_calc_plot[interp_mask_calc],
                                        pen=pg.mkPen(color=(255, 100, 100), width=2),
                                        symbol='o', symbolSize=7,
                                        symbolBrush=pg.mkBrush(255, 100, 100, 220),
                                        name="X-Y (расчётные, восстановлено)")
                else:
                    # Нет маски - отображаем все точки жёлтым цветом
                    plot_widget.plot(X_calc_plot, Y_calc_plot,
                                    pen=pg.mkPen(color=(200, 200, 0), width=1, 
                                                style=pg.QtCore.Qt.DashLine),
                                    symbol='+', symbolSize=8,
                                    name="X-Y (расчётные)")
        
        # Добавляем эталонные кривые, если есть
        # Важно: validation_data должна содержать действительно эталонные данные,
        # а не просто текущие данные с пропусками
        # Эталон pD(X) отображается ТОЛЬКО если явно выбран график pD(Y) в GUI
        # И validation_data не пустой словарь
        # И явно загружен файл для валидации
        
        # Эталон отображается только если явно выбран график pD(Y)
        pD_plot_selected = checked_groups.get('cb_dim_pD', False)
        
        if validation_data is not None and validation_data != {} and pD_plot_selected:
            try:
                ref_dim = validation_data.get('ref_dim')
                # Дополнительная проверка: убеждаемся, что ref_dim не пустой и содержит валидные данные
                # И что это действительно эталонные данные (не текущие)
                if ref_dim and hasattr(ref_dim, 'X') and hasattr(ref_dim, 'pressure'):
                    # Проверяем, что эталонные данные отличаются от текущих
                    # Сравниваем X и Y с текущими данными
                    current_X = dim_data.X
                    current_Y = dim_data.Y
                    ref_X = ref_dim.X.astype(float)
                    ref_Y = ref_dim.Y.astype(float) if hasattr(ref_dim, 'Y') else None
                    
                    # Проверяем, что эталонные данные действительно отличаются
                    # (разная длина или разные значения)
                    # Сначала проверяем длину
                    if len(ref_X) != len(current_X):
                        is_really_different = True
                    else:
                        # Если длины совпадают, проверяем значения
                        try:
                            # Сравниваем X
                            x_close = np.allclose(ref_X, current_X, rtol=1e-3, equal_nan=True)
                            # Сравниваем Y, если есть
                            y_close = True
                            if ref_Y is not None and hasattr(ref_dim, 'Y') and len(ref_Y) == len(current_Y):
                                y_close = np.allclose(ref_Y, current_Y, rtol=1e-3, equal_nan=True)
                            
                            # Данные отличаются, если хотя бы один параметр не совпадает
                            is_really_different = not (x_close and y_close)
                        except (ValueError, TypeError):
                            # Если не удалось сравнить, считаем что данные разные
                            is_really_different = True
                    
                    if is_really_different:
                        pD_ref = ref_dim.pressure / (ref_dim.delta_p_i if ref_dim.delta_p_i != 0 else 1.0)
                        mask_rp = np.isfinite(ref_X) & np.isfinite(pD_ref)
                        # Проверяем, что есть достаточно точек для отображения
                        if np.any(mask_rp) and np.sum(mask_rp) > 1:
                            plot_widget.plot(ref_X[mask_rp], pD_ref[mask_rp],
                                            pen=pg.mkPen(color=(120, 120, 120), width=2, 
                                                        style=pg.QtCore.Qt.DashLine),
                                            name="Эталон pD(X)")
            except Exception:
                # В случае ошибки просто не отображаем эталон
                pass
        
        # Всегда добавляем легенду в конце, если были построены какие-либо графики
        plot_widget.addLegend()
        return  # Безразмерные кривые в своем пространстве
    
    # ГРУППА 3: Типовые кривые - плоскость (Y, pD)
    # Типовые кривые отображаются в плоскости Y-pD
    if has_type:
        plot_widget.setLogMode(x=False, y=False)
        plot_widget.setLabel('bottom', 'Y (безразмерный ёмкостной параметр)')
        plot_widget.setLabel('left', 'pD (безразмерное давление)')
        plot_widget.setTitle("Типовые кривые")
        
        # Используем Y из данных или расчётных значений для определения диапазона
        # Если типовые кривые выбраны отдельно (без безразмерных), используем dim_data.Y
        if Y_data is not None and len(Y_data) > 0:
            y_min = float(Y_data.min())
            y_max = float(Y_data.max())
        else:
            # Используем Y из dim_data
            y_min = float(np.min(dim_data.Y))
            y_max = float(np.max(dim_data.Y))
        
        # Используем линейную сетку вместо логарифмической
        y_ref = np.linspace(max(y_min, 1e-3), max(y_max, 1e2), 100)
        
        if checked_groups.get('cb_type_gry', False):
            curve = 1.2 / (y_ref ** 0.5)
            plot_widget.plot(y_ref, curve,
                            pen=pg.mkPen(color=(120, 120, 120, 160), width=1, 
                                        style=pg.QtCore.Qt.DashLine),
                            name="Gringarten & Ramey (прибл.)")
        
        if checked_groups.get('cb_type_cinco', False):
            curve = 0.9 / (y_ref ** 0.4)
            plot_widget.plot(y_ref, curve,
                            pen=pg.mkPen(color=(120, 120, 120, 160), width=1, 
                                        style=pg.QtCore.Qt.DotLine),
                            name="Cinco-Ley & Samaniego (прибл.)")
        
        if checked_groups.get('cb_type_valko', False):
            curve = 0.7 / (y_ref ** 0.3)
            plot_widget.plot(y_ref, curve,
                            pen=pg.mkPen(color=(120, 120, 120, 160), width=1),
                            name="Valkó & Economides (прибл.)")
        
        plot_widget.addLegend()
        return  # Типовые кривые в своем пространстве
    
    # ГРУППА 4: Специальные пространства - разные плоскости в зависимости от типа
    if has_special:
        if checked_groups.get('cb_gfunc', False):
            # G-функция Nolte: плоскость (t, G(t))
            plot_widget.setLogMode(x=False, y=False)
            plot_widget.setLabel('bottom', 'Время, ч')
            plot_widget.setLabel('left', 'G(t)')
            plot_widget.setTitle("G-функция Nolte")
            
            # Вычисляем G-функцию
            t = time.values
            G = (2.0 / np.sqrt(np.pi)) * np.sqrt(np.clip(t, 0.0, None))
            mask = ~np.isnan(G)
            if np.any(mask):
                plot_widget.plot(t[mask], G[mask],
                                pen=pg.mkPen(color=(100, 100, 200), width=2),
                                name="G-функция")
            return
        
        if checked_groups.get('cb_mbt', False):
            # Material Balance Time: плоскость (t_mb, Q)
            plot_widget.setLogMode(x=False, y=False)
            plot_widget.setLabel('bottom', 'Material Balance Time')
            plot_widget.setLabel('left', 'Дебит Q, м³/сут')
            plot_widget.setTitle("Material Balance Time")
            
            # Вычисляем MBT
            q0 = flow_rate.iloc[0] if len(flow_rate) > 0 else 1.0
            if q0 != 0:
                dt = np.gradient(time.values)
                cum = np.cumsum(flow_rate.values * dt)
                t_mb = cum / q0
                mask = (t_mb > 0) & (flow_rate.values > 0)
                if np.any(mask):
                    plot_widget.plot(t_mb[mask], flow_rate.values[mask],
                                    pen=pg.mkPen(color=(150, 100, 50), width=2),
                                    name="MBT")
            return
    
    # Если ничего не выбрано, просто очищаем график и показываем пустую область
    if not (has_real or has_dim or has_type or has_special):
        plot_widget.setLogMode(x=False, y=False)
        plot_widget.setLabel('bottom', 'X (безразмерный фильтрационный параметр)')
        plot_widget.setLabel('left', 'Безразмерный параметр')
        plot_widget.setTitle("Безразмерные кривые МГРП")
        plot_widget.showGrid(x=True, y=True)
        plot_widget.addLegend()
        return  # Просто показываем пустой график без данных
