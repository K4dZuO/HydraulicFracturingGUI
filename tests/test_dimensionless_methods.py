#!/usr/bin/env python3
"""
Тестирование безразмерных методов интерполяции и экстраполяции для МГРП
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


def test_dimensionless_conversion():
    """Тестирование конвертации в безразмерные параметры"""
    print("=== Тестирование конвертации в безразмерные параметры ===")
    
    time, pressure, flow_rate, well_params = create_test_grp_data()
    
    # Конвертируем в безразмерные параметры
    dimensionless_data = convert_to_dimensionless_curves(time, pressure, flow_rate, well_params)
    
    print(f"Фильтрационный параметр X: {dimensionless_data.X[0]:.6f}")
    print(f"Ёмкостной параметр Y (диапазон): {dimensionless_data.Y.min():.6f} - {dimensionless_data.Y.max():.6f}")
    print(f"Безразмерное давление (диапазон): {(dimensionless_data.pressure/dimensionless_data.delta_p_i).min():.6f} - {(dimensionless_data.pressure/dimensionless_data.delta_p_i).max():.6f}")
    
    # Создаем график
    fig = plot_dimensionless_analysis(time, pressure, flow_rate, well_params, 'pressure')
    fig.savefig('dimensionless_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dimensionless_data


def test_dimensionless_interpolation():
    """Тестирование интерполяции в безразмерном пространстве"""
    print("\n=== Тестирование безразмерной интерполяции ===")
    
    time, pressure, flow_rate, well_params = create_test_grp_data()
    
    # Создаем целевые временные точки
    target_times = np.linspace(time.min(), time.max(), 100)
    target_params = well_params.copy()
    
    # Тестируем различные методы
    methods = ['rbf', 'linear']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Сравнение методов безразмерной интерполяции', fontsize=16)
    
    for i, method in enumerate(methods):
        try:
            print(f"Тестируем метод: {method}")
            
            # Интерполируем
            interpolated_pressure, interpolated_flow = interpolate_dimensionless_curves(
                time, pressure, flow_rate, well_params, target_times, target_params, method
            )
            
            # График давления
            ax1 = axes[i, 0]
            ax1.semilogy(time, pressure, 'ko-', label='Исходные данные', markersize=4)
            ax1.semilogy(target_times, interpolated_pressure, 'r-', label=f'Интерполяция ({method})', linewidth=2)
            ax1.set_xlabel('Время, ч')
            ax1.set_ylabel('Давление, атм')
            ax1.set_title(f'Давление - {method}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # График дебита
            ax2 = axes[i, 1]
            ax2.semilogy(time, flow_rate, 'ko-', label='Исходные данные', markersize=4)
            ax2.semilogy(target_times, interpolated_flow, 'r-', label=f'Интерполяция ({method})', linewidth=2)
            ax2.set_xlabel('Время, ч')
            ax2.set_ylabel('Дебит, м³/сут')
            ax2.set_title(f'Дебит - {method}')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            print(f"✓ {method} - успешно")
            
        except Exception as e:
            print(f"✗ {method} - ошибка: {e}")
    
    plt.tight_layout()
    plt.savefig('dimensionless_interpolation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_dimensionless_extrapolation():
    """Тестирование экстраполяции безразмерных кривых"""
    print("\n=== Тестирование безразмерной экстраполяции ===")
    
    time, pressure, flow_rate, well_params = create_test_grp_data()
    
    # Создаем будущие временные точки
    time_max = time.max()
    future_times = np.linspace(time_max, time_max * 3, 50)
    extrapolation_params = well_params.copy()
    
    # Тестируем различные методы экстраполяции
    methods = ['physics_constrained', 'rbf']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Сравнение методов безразмерной экстраполяции', fontsize=16)
    
    for i, method in enumerate(methods):
        try:
            print(f"Тестируем экстраполяцию: {method}")
            
            # Экстраполируем
            extrapolated_pressure, extrapolated_flow = extrapolate_dimensionless_curves(
                time, pressure, flow_rate, well_params, future_times, extrapolation_params, method
            )
            
            # График давления
            ax1 = axes[i, 0]
            ax1.semilogy(time, pressure, 'ko-', label='Исходные данные', markersize=4)
            ax1.semilogy(future_times, extrapolated_pressure, 'g-', label=f'Экстраполяция ({method})', linewidth=2)
            ax1.set_xlabel('Время, ч')
            ax1.set_ylabel('Давление, атм')
            ax1.set_title(f'Давление - {method}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # График дебита
            ax2 = axes[i, 1]
            ax2.semilogy(time, flow_rate, 'ko-', label='Исходные данные', markersize=4)
            ax2.semilogy(future_times, extrapolated_flow, 'g-', label=f'Экстраполяция ({method})', linewidth=2)
            ax2.set_xlabel('Время, ч')
            ax2.set_ylabel('Дебит, м³/сут')
            ax2.set_title(f'Дебит - {method}')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            print(f"✓ {method} - успешно")
            
        except Exception as e:
            print(f"✗ {method} - ошибка: {e}")
    
    plt.tight_layout()
    plt.savefig('dimensionless_extrapolation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_type_curves():
    """Тестирование создания библиотеки эталонных кривых"""
    print("\n=== Тестирование создания эталонных кривых ===")
    
    try:
        # Создаем библиотеку эталонных кривых
        type_curves = create_dimensionless_type_curves(
            skin_range=(-5, 10),
            n_fractures_range=(1, 20),
            a_l_range=(0.01, 0.3),
            n_points=50
        )
        
        print(f"Создана библиотека из {len(type_curves)} эталонных кривых")
        
        # Визуализируем несколько кривых
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Библиотека эталонных кривых МГРП', fontsize=16)
        
        # Выбираем несколько кривых для отображения
        selected_curves = list(type_curves.items())[:4]
        
        for i, (curve_id, curve_data) in enumerate(selected_curves):
            ax = axes[i//2, i%2]
            
            # Давление
            ax.semilogy(curve_data['time'], curve_data['bilinear_pressure'], 
                       'r-', label='Билинейное течение', linewidth=2)
            ax.semilogy(curve_data['time'], curve_data['linear_pressure'], 
                       'b-', label='Линейное течение', linewidth=2)
            ax.semilogy(curve_data['time'], curve_data['pseudoradial_pressure'], 
                       'g-', label='Псевдорадиальное течение', linewidth=2)
            
            ax.set_xlabel('Время, ч')
            ax.set_ylabel('Давление, атм')
            ax.set_title(f'Кривая {i+1}: Skin={curve_data["skin"]:.1f}, N={curve_data["n_fractures"]:.0f}, a/L={curve_data["a_l_ratio"]:.3f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('type_curves_library.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Библиотека эталонных кривых создана успешно")
        
    except Exception as e:
        print(f"✗ Ошибка создания библиотеки: {e}")


def test_physics_constraints():
    """Тестирование физических ограничений"""
    print("\n=== Тестирование физических ограничений ===")
    
    time, pressure, flow_rate, well_params = create_test_grp_data()
    
    # Тестируем различные параметры
    test_params = [
        {'skin': -10, 'N': 1, 'a_L': 0.01},      # Минимальные значения
        {'skin': 50, 'N': 100, 'a_L': 1.0},     # Максимальные значения
        {'skin': 0, 'N': 10, 'a_L': 0.1},        # Средние значения
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Тестирование физических ограничений', fontsize=16)
    
    for i, params in enumerate(test_params):
        try:
            # Обновляем параметры
            test_well_params = well_params.copy()
            test_well_params.update(params)
            
            # Создаем будущие временные точки
            future_times = np.linspace(time.max(), time.max() * 2, 30)
            
            # Экстраполируем с физическими ограничениями
            extrapolated_pressure, extrapolated_flow = extrapolate_dimensionless_curves(
                time, pressure, flow_rate, test_well_params, future_times, test_well_params, 'physics_constrained'
            )
            
            ax = axes[i]
            ax.semilogy(time, pressure, 'ko-', label='Исходные данные', markersize=4)
            ax.semilogy(future_times, extrapolated_pressure, 'r-', label='Экстраполяция', linewidth=2)
            ax.set_xlabel('Время, ч')
            ax.set_ylabel('Давление, атм')
            ax.set_title(f'Параметры: Skin={params["skin"]}, N={params["N"]}, a/L={params["a_L"]}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            print(f"✓ Параметры {params} - успешно")
            
        except Exception as e:
            print(f"✗ Параметры {params} - ошибка: {e}")
    
    plt.tight_layout()
    plt.savefig('physics_constraints_test.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Основная функция тестирования"""
    print("=== Тестирование безразмерных методов для МГРП ===\n")
    
    # Тестируем конвертацию
    dimensionless_data = test_dimensionless_conversion()
    
    # Тестируем интерполяцию
    test_dimensionless_interpolation()
    
    # Тестируем экстраполяцию
    test_dimensionless_extrapolation()
    
    # Тестируем создание эталонных кривых
    test_type_curves()
    
    # Тестируем физические ограничения
    test_physics_constraints()
    
    print("\n=== Тестирование завершено ===")
    print("Графики сохранены:")
    print("- dimensionless_analysis.png")
    print("- dimensionless_interpolation_comparison.png")
    print("- dimensionless_extrapolation_comparison.png")
    print("- type_curves_library.png")
    print("- physics_constraints_test.png")


if __name__ == "__main__":
    main()
