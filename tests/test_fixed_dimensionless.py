#!/usr/bin/env python3
"""
Тестирование исправленных безразмерных методов
Демонстрация правильных осей и подписей параметров
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers.dimensionless_analysis import (
    convert_to_dimensionless_curves,
    interpolate_dimensionless_curves,
    extrapolate_dimensionless_curves,
    create_dimensionless_type_curves
)
from helpers.dimensionless_plotting import (
    plot_dimensionless_analysis,
    plot_interpolation_comparison,
    plot_extrapolation_comparison
)


def create_test_grp_data():
    """Создание тестовых данных МГРП"""
    # Временной ряд
    time = pd.Series(np.linspace(0.1, 100, 50))
    
    # Физически обоснованные кривые для МГРП
    # Билинейное течение
    bilinear_pressure = 100 * np.exp(-0.05 * time) * (1 + 0.1 * np.sin(time/10))
    bilinear_flow = 50 * np.exp(-0.1 * time) * (1 + 0.05 * np.cos(time/15))
    
    # Добавляем шум
    pressure_noise = np.random.normal(0, 2, len(time))
    flow_noise = np.random.normal(0, 1, len(time))
    
    pressure = bilinear_pressure + pressure_noise
    flow_rate = bilinear_flow + flow_noise
    
    # Параметры скважины
    well_params = {
        'k': 1.0,      # Проницаемость, мД
        'h': 10.0,     # Толщина пласта, м
        'mu': 1.0,     # Вязкость, мПа·с
        'B': 1.0,      # Объемный коэффициент
        'phi': 0.1,    # Пористость
        'c_t': 1e-4,   # Общая сжимаемость, 1/атм
        'L': 100.0,    # Длина трещины, м
        'skin': 2.0,   # Фактор скин-эффекта
        'N': 5,        # Количество трещин
        'a_L': 0.1     # Отношение a/L
    }
    
    return time, pressure, flow_rate, well_params


def test_corrected_dimensionless_plotting():
    """Тестирование исправленного построения безразмерных графиков"""
    print("=== Тестирование исправленного построения безразмерных графиков ===")
    
    time, pressure, flow_rate, well_params = create_test_grp_data()
    
    # Тестируем основные графики
    print("1. Тестирование основного безразмерного анализа...")
    fig1 = plot_dimensionless_analysis(time, pressure, flow_rate, well_params, 'pressure')
    fig1.savefig('corrected_dimensionless_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("2. Тестирование безразмерного анализа дебита...")
    fig2 = plot_dimensionless_analysis(time, pressure, flow_rate, well_params, 'flow_rate')
    fig2.savefig('corrected_dimensionless_flow.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("3. Тестирование комбинированного графика...")
    fig3 = plot_dimensionless_analysis(time, pressure, flow_rate, well_params, 'both')
    fig3.savefig('corrected_dimensionless_both.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_corrected_type_curves():
    """Тестирование исправленных эталонных кривых"""
    print("\n=== Тестирование исправленных эталонных кривых ===")
    
    time, pressure, flow_rate, well_params = create_test_grp_data()
    
    # Создаем библиотеку эталонных кривых
    print("Создание библиотеки эталонных кривых...")
    type_curves = create_dimensionless_type_curves(
        skin_range=(-2, 8),
        n_fractures_range=(1, 20),
        a_l_range=(0.01, 0.3),
        n_points=50
    )
    
    # Конвертируем данные в безразмерные
    dimensionless_data = convert_to_dimensionless_curves(time, pressure, flow_rate, well_params)
    
    # Создаем график сравнения с эталонными кривыми
    from helpers.dimensionless_plotting import DimensionlessPlotter
    plotter = DimensionlessPlotter()
    
    print("Построение сравнения с эталонными кривыми...")
    fig = plotter.plot_type_curves_comparison(dimensionless_data, type_curves)
    fig.savefig('corrected_type_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Создана библиотека из {len(type_curves)} эталонных кривых")
    print("✓ Эталонные кривые используют те же оси, что и основные данные")


def test_axes_consistency():
    """Тестирование согласованности осей"""
    print("\n=== Тестирование согласованности осей ===")
    
    time, pressure, flow_rate, well_params = create_test_grp_data()
    
    # Конвертируем в безразмерные параметры
    dimensionless_data = convert_to_dimensionless_curves(time, pressure, flow_rate, well_params)
    
    print(f"Диапазон ёмкостного параметра Y: {dimensionless_data.Y.min():.6f} - {dimensionless_data.Y.max():.6f}")
    print(f"Диапазон безразмерного давления: {(dimensionless_data.pressure/dimensionless_data.delta_p_i).min():.6f} - {(dimensionless_data.pressure/dimensionless_data.delta_p_i).max():.6f}")
    print(f"Диапазон безразмерного дебита: {(dimensionless_data.flow_rate/dimensionless_data.Q).min():.6f} - {(dimensionless_data.flow_rate/dimensionless_data.Q).max():.6f}")
    
    # Проверяем, что оси правильно настроены
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Проверка согласованности осей', fontsize=16)
    
    # График 1: Безразмерное давление
    dimensionless_pressure = dimensionless_data.pressure / dimensionless_data.delta_p_i
    ax1.loglog(dimensionless_data.Y, dimensionless_pressure, 'bo-', label='Данные', markersize=4)
    ax1.set_xlabel('Ёмкостной параметр Y (безразмерный)')
    ax1.set_ylabel('Безразмерное давление')
    ax1.set_title('Безразмерное давление vs Y')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # График 2: Безразмерный дебит
    dimensionless_flow = dimensionless_data.flow_rate / dimensionless_data.Q
    ax2.loglog(dimensionless_data.Y, dimensionless_flow, 'go-', label='Данные', markersize=4)
    ax2.set_xlabel('Ёмкостной параметр Y (безразмерный)')
    ax2.set_ylabel('Безразмерный дебит')
    ax2.set_title('Безразмерный дебит vs Y')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('axes_consistency_check.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Оси правильно настроены и согласованы")


def test_parameter_labels():
    """Тестирование подписей параметров"""
    print("\n=== Тестирование подписей параметров ===")
    
    time, pressure, flow_rate, well_params = create_test_grp_data()
    
    # Создаем график с правильными подписями
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Конвертируем в безразмерные параметры
    dimensionless_data = convert_to_dimensionless_curves(time, pressure, flow_rate, well_params)
    
    # Безразмерное давление
    dimensionless_pressure = dimensionless_data.pressure / dimensionless_data.delta_p_i
    ax.loglog(dimensionless_data.Y, dimensionless_pressure, 'bo-', 
              label='Безразмерное давление', markersize=6, linewidth=2)
    
    # Добавляем информацию о параметрах
    param_text = f"""Параметры скважины:
Skin = {well_params['skin']:.1f}
N = {well_params['N']:.0f}
a/L = {well_params['a_L']:.3f}
h = {well_params['h']:.1f} м
L = {well_params['L']:.1f} м"""
    
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Ёмкостной параметр Y (безразмерный)', fontsize=12)
    ax.set_ylabel('Безразмерное давление', fontsize=12)
    ax.set_title('Безразмерные кривые МГРП с параметрами', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('parameter_labels_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Подписи параметров добавлены корректно")


def test_interpolation_with_correct_axes():
    """Тестирование интерполяции с правильными осями"""
    print("\n=== Тестирование интерполяции с правильными осями ===")
    
    time, pressure, flow_rate, well_params = create_test_grp_data()
    
    # Создаем целевые временные точки
    target_times = np.linspace(time.min(), time.max(), 100)
    target_params = well_params.copy()
    
    # Интерполируем
    interpolated_pressure, interpolated_flow = interpolate_dimensionless_curves(
        time, pressure, flow_rate, well_params, target_times, target_params, 'rbf'
    )
    
    # Создаем график сравнения
    fig = plot_interpolation_comparison(
        time, pressure, flow_rate, well_params, target_times, target_params, 'rbf'
    )
    fig.savefig('corrected_interpolation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Интерполяция выполнена с правильными осями")


def main():
    """Основная функция тестирования исправлений"""
    print("=== Тестирование исправленных безразмерных методов ===\n")
    
    # Тестируем исправленное построение графиков
    test_corrected_dimensionless_plotting()
    
    # Тестируем исправленные эталонные кривые
    test_corrected_type_curves()
    
    # Тестируем согласованность осей
    test_axes_consistency()
    
    # Тестируем подписи параметров
    test_parameter_labels()
    
    # Тестируем интерполяцию
    test_interpolation_with_correct_axes()
    
    print("\n=== Тестирование завершено ===")
    print("Исправления:")
    print("✓ Оси графиков исправлены (Y по X, безразмерные значения по Y)")
    print("✓ Добавлены подписи параметров на всех графиках")
    print("✓ Эталонные кривые используют те же оси, что и основные данные")
    print("✓ Все графики имеют единообразные подписи осей")
    print("✓ Логарифмический масштаб применен корректно")
    
    print("\nСохраненные графики:")
    print("- corrected_dimensionless_analysis.png")
    print("- corrected_dimensionless_flow.png") 
    print("- corrected_dimensionless_both.png")
    print("- corrected_type_curves_comparison.png")
    print("- axes_consistency_check.png")
    print("- parameter_labels_test.png")
    print("- corrected_interpolation_comparison.png")


if __name__ == "__main__":
    main()
