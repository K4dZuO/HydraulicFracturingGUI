#!/usr/bin/env python3
"""
Тестирование новых методов интерполяции для ГРП данных
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers.ml_methods import (
    apply_kriging_interpolation,
    apply_rbf_interpolation, 
    apply_gp_interpolation,
    apply_physics_constrained_interpolation,
    apply_adaptive_interpolation,
    apply_ml_interpolation,
)

def create_test_data():
    """Создание тестовых данных ГРП"""
    # Создаем временной ряд с пропусками
    time = pd.date_range('2023-01-01', periods=100, freq='H')
    
    # Симулируем данные давления с физическими ограничениями
    t = np.linspace(0, 10, 100)
    pressure = 50 + 20 * np.exp(-t/5) + 5 * np.sin(t) + np.random.normal(0, 2, 100)
    
    # Симулируем данные дебита
    flow_rate = 100 * np.exp(-t/3) + 10 * np.cos(t) + np.random.normal(0, 5, 100)
    
    # Добавляем пропуски
    missing_indices = np.random.choice(100, size=20, replace=False)
    pressure_with_gaps = pressure.copy()
    flow_rate_with_gaps = flow_rate.copy()
    pressure_with_gaps[missing_indices] = np.nan
    flow_rate_with_gaps[missing_indices] = np.nan
    
    return time, pressure_with_gaps, flow_rate_with_gaps, pressure, flow_rate

def test_interpolation_methods():
    """Тестирование различных методов интерполяции"""
    time, pressure_gaps, flow_rate_gaps, pressure_true, flow_rate_true = create_test_data()
    
    # Создаем pandas Series
    time_series = pd.Series(time)
    pressure_series = pd.Series(pressure_gaps, index=time)
    flow_rate_series = pd.Series(flow_rate_gaps, index=time)
    
    print("Тестирование методов интерполяции...")
    
    # Тестируем различные методы
    methods = {
        'ML (Random Forest)': lambda: apply_ml_interpolation(time_series, pressure_series, 'random_forest'),
        'ML (Polynomial)': lambda: apply_ml_interpolation(time_series, pressure_series, 'polynomial'),
        'Кригинг': lambda: apply_kriging_interpolation(time_series, pressure_series),
        'RBF (Multiquadric)': lambda: apply_rbf_interpolation(time_series, pressure_series, 'multiquadric'),
        'RBF (Gaussian)': lambda: apply_rbf_interpolation(time_series, pressure_series, 'gaussian'),
        'GP (RBF kernel)': lambda: apply_gp_interpolation(time_series, pressure_series, return_std=False),
        'Физически ограниченная': lambda: apply_physics_constrained_interpolation(
            time_series, pressure_series, 'pressure'),
        'Адаптивная': lambda: apply_adaptive_interpolation(time_series, pressure_series)
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        try:
            print(f"Тестируем {method_name}...")
            interpolated = method_func()
            results[method_name] = interpolated
            print(f"✓ {method_name} - успешно")
        except Exception as e:
            print(f"✗ {method_name} - ошибка: {e}")
            results[method_name] = None
    
    return results, pressure_true, flow_rate_true

def plot_comparison(results, pressure_true, flow_rate_true):
    """Визуализация сравнения методов"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Сравнение методов интерполяции для ГРП данных', fontsize=16)
    
    # График 1: Давление
    ax1 = axes[0, 0]
    ax1.plot(pressure_true, 'b-', label='Истинные данные', alpha=0.7)
    
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    for i, (method_name, result) in enumerate(results.items()):
        if result is not None:
            ax1.plot(result.values, '--', color=colors[i % len(colors)], 
                    label=method_name, alpha=0.8)
    
    ax1.set_title('Интерполяция давления')
    ax1.set_xlabel('Время (часы)')
    ax1.set_ylabel('Давление (МПа)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # График 2: Дебит
    ax2 = axes[0, 1]
    ax2.plot(flow_rate_true, 'b-', label='Истинные данные', alpha=0.7)
    
    for i, (method_name, result) in enumerate(results.items()):
        if result is not None:
            ax2.plot(result.values, '--', color=colors[i % len(colors)], 
                    label=method_name, alpha=0.8)
    
    ax2.set_title('Интерполяция дебита')
    ax2.set_xlabel('Время (часы)')
    ax2.set_ylabel('Дебит (м³/сут)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # График 3: Ошибки интерполяции
    ax3 = axes[1, 0]
    errors = {}
    for method_name, result in results.items():
        if result is not None:
            error = np.abs(result.values - pressure_true)
            errors[method_name] = np.mean(error)
    
    if errors:
        methods = list(errors.keys())
        error_values = list(errors.values())
        bars = ax3.bar(methods, error_values, color=colors[:len(methods)])
        ax3.set_title('Средняя абсолютная ошибка')
        ax3.set_ylabel('Ошибка (МПа)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, error_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom')
    
    # График 4: Временная динамика ошибки
    ax4 = axes[1, 1]
    for method_name, result in results.items():
        if result is not None:
            error = np.abs(result.values - pressure_true)
            ax4.plot(error, label=method_name, alpha=0.7)
    
    ax4.set_title('Временная динамика ошибки')
    ax4.set_xlabel('Время (часы)')
    ax4.set_ylabel('Абсолютная ошибка (МПа)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_gp_uncertainty():
    """Тестирование GP с оценкой неопределенности"""
    time, pressure_gaps, _, pressure_true, _ = create_test_data()
    time_series = pd.Series(time)
    pressure_series = pd.Series(pressure_gaps, index=time)
    
    print("\nТестирование GP с оценкой неопределенности...")
    
    try:
        mean, std = apply_gp_interpolation(time_series, pressure_series, return_std=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(pressure_true, 'b-', label='Истинные данные', linewidth=2)
        plt.plot(mean.values, 'r--', label='GP предсказание', linewidth=2)
        plt.fill_between(range(len(mean)), 
                        mean.values - 2*std.values, 
                        mean.values + 2*std.values, 
                        alpha=0.3, color='red', label='95% доверительный интервал')
        plt.title('GP интерполяция с оценкой неопределенности')
        plt.xlabel('Время (часы)')
        plt.ylabel('Давление (МПа)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('gp_uncertainty.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ GP с неопределенностью - успешно")
        
    except Exception as e:
        print(f"✗ GP с неопределенностью - ошибка: {e}")

def main():
    """Основная функция тестирования"""
    print("=== Тестирование новых методов интерполяции для ГРП данных ===\n")
    
    # Тестируем основные методы
    results, pressure_true, flow_rate_true = test_interpolation_methods()
    
    # Визуализируем результаты
    plot_comparison(results, pressure_true, flow_rate_true)
    
    # Тестируем GP с неопределенностью
    test_gp_uncertainty()
    
    print("\n=== Тестирование завершено ===")
    print("Графики сохранены как 'interpolation_comparison.png' и 'gp_uncertainty.png'")

if __name__ == "__main__":
    main()
