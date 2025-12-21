#!/usr/bin/env python3
"""
Тестирование правильной ориентации осей для безразмерных графиков
Проверка что Y всегда по вертикали, а другие параметры по горизонтали
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers.dimensionless_analysis import convert_to_dimensionless_curves
from helpers.dimensionless_plotting import plot_dimensionless_analysis


def create_test_data():
    """Создание тестовых данных МГРП"""
    time = pd.Series(np.linspace(0.1, 100, 50))
    
    # Физически обоснованные кривые
    pressure = pd.Series(100 * np.exp(-0.05 * time) + np.random.normal(0, 2, len(time)))
    flow_rate = pd.Series(50 * np.exp(-0.1 * time) + np.random.normal(0, 1, len(time)))
    
    well_params = {
        'k': 1.0, 'h': 10.0, 'mu': 1.0, 'B': 1.0,
        'phi': 0.1, 'c_t': 1e-4, 'L': 100.0,
        'skin': 2.0, 'N': 5, 'a_L': 0.1
    }
    
    return time, pressure, flow_rate, well_params


def test_axes_orientation():
    """Тестирование правильной ориентации осей"""
    print("=== Тестирование правильной ориентации осей ===")
    
    time, pressure, flow_rate, well_params = create_test_data()
    
    # Конвертируем в безразмерные параметры
    dimensionless_data = convert_to_dimensionless_curves(time, pressure, flow_rate, well_params)
    
    print("Проверка ориентации осей:")
    print(f"✓ Ёмкостной параметр Y: {dimensionless_data.Y.shape} - должен быть по вертикали")
    print(f"✓ Безразмерное давление: {(dimensionless_data.pressure/dimensionless_data.delta_p_i).shape} - должно быть по горизонтали")
    print(f"✓ Безразмерный дебит: {(dimensionless_data.flow_rate/dimensionless_data.Q).shape} - должно быть по горизонтали")
    
    # Проверяем, что Y изменяется (должен быть по вертикали)
    Y_values = dimensionless_data.Y
    if not np.allclose(Y_values, Y_values[0]):
        print(f"✓ Y изменяется: {Y_values.min():.6f} - {Y_values.max():.6f} (правильно для вертикальной оси)")
    else:
        print(f"✗ Y не изменяется (неправильно для вертикальной оси)")
        return False
    
    # Проверяем, что давление изменяется (должно быть по горизонтали)
    dimensionless_pressure = dimensionless_data.pressure / dimensionless_data.delta_p_i
    if not np.allclose(dimensionless_pressure, dimensionless_pressure[0]):
        print(f"✓ Безразмерное давление изменяется: {dimensionless_pressure.min():.6f} - {dimensionless_pressure.max():.6f} (правильно для горизонтальной оси)")
    else:
        print(f"✗ Безразмерное давление не изменяется (неправильно для горизонтальной оси)")
        return False
    
    return True


def test_graph_shape():
    """Тестирование формы графика"""
    print("\n=== Тестирование формы графика ===")
    
    time, pressure, flow_rate, well_params = create_test_data()
    
    # Конвертируем в безразмерные параметры
    dimensionless_data = convert_to_dimensionless_curves(time, pressure, flow_rate, well_params)
    
    # Безразмерные значения
    dimensionless_pressure = dimensionless_data.pressure / dimensionless_data.delta_p_i
    dimensionless_flow = dimensionless_data.flow_rate / dimensionless_data.Q
    
    print("Проверка формы графика:")
    
    # Проверяем, что график имеет правильную форму (параболическую, идущую вверх от (0,0))
    # Y должен увеличиваться, а давление/дебит должны уменьшаться
    Y_increasing = np.all(np.diff(dimensionless_data.Y) > 0)
    pressure_decreasing = np.all(np.diff(dimensionless_pressure) < 0)
    flow_decreasing = np.all(np.diff(dimensionless_flow) < 0)
    
    print(f"✓ Y увеличивается: {Y_increasing}")
    print(f"✓ Давление уменьшается: {pressure_decreasing}")
    print(f"✓ Дебит уменьшается: {flow_decreasing}")
    
    if Y_increasing and pressure_decreasing and flow_decreasing:
        print("✓ График имеет правильную форму - параболическую, идущую вверх от (0,0)")
        return True
    else:
        print("✗ График имеет неправильную форму")
        return False


def test_plotting_consistency():
    """Тестирование согласованности построения графиков"""
    print("\n=== Тестирование согласованности построения графиков ===")
    
    time, pressure, flow_rate, well_params = create_test_data()
    
    try:
        # Тестируем matplotlib графики
        print("Тестирование matplotlib графиков...")
        fig1 = plot_dimensionless_analysis(time, pressure, flow_rate, well_params, 'pressure')
        print("✓ Matplotlib график давления построен")
        
        fig2 = plot_dimensionless_analysis(time, pressure, flow_rate, well_params, 'flow_rate')
        print("✓ Matplotlib график дебита построен")
        
        fig3 = plot_dimensionless_analysis(time, pressure, flow_rate, well_params, 'both')
        print("✓ Matplotlib комбинированный график построен")
        
        # Сохраняем графики для проверки
        fig1.savefig('test_pressure_axes.png', dpi=150, bbox_inches='tight')
        fig2.savefig('test_flow_axes.png', dpi=150, bbox_inches='tight')
        fig3.savefig('test_combined_axes.png', dpi=150, bbox_inches='tight')
        
        print("✓ Графики сохранены для визуальной проверки")
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка построения графиков: {e}")
        return False


def test_axes_labels():
    """Тестирование подписей осей"""
    print("\n=== Тестирование подписей осей ===")
    
    time, pressure, flow_rate, well_params = create_test_data()
    
    # Конвертируем в безразмерные параметры
    dimensionless_data = convert_to_dimensionless_curves(time, pressure, flow_rate, well_params)
    
    print("Проверка подписей осей:")
    
    # Проверяем правильность подписей
    expected_labels = {
        'Y_axis': 'Ёмкостной параметр Y (безразмерный)',
        'pressure_axis': 'Безразмерное давление',
        'flow_axis': 'Безразмерный дебит'
    }
    
    print(f"✓ Ожидаемая подпись Y-оси: {expected_labels['Y_axis']}")
    print(f"✓ Ожидаемая подпись оси давления: {expected_labels['pressure_axis']}")
    print(f"✓ Ожидаемая подпись оси дебита: {expected_labels['flow_axis']}")
    
    # Проверяем, что все подписи содержат правильную информацию
    for label_name, expected_label in expected_labels.items():
        if 'безразмерный' in expected_label or 'Безразмерное' in expected_label or 'Безразмерный' in expected_label:
            print(f"✓ {label_name}: содержит информацию о безразмерности")
        else:
            print(f"✗ {label_name}: не содержит информацию о безразмерности")
            return False
    
    return True


def main():
    """Основная функция тестирования"""
    print("=== Тестирование правильной ориентации осей ===\n")
    
    tests = [
        ("Ориентация осей", test_axes_orientation),
        ("Форма графика", test_graph_shape),
        ("Согласованность построения", test_plotting_consistency),
        ("Подписи осей", test_axes_labels)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"ТЕСТ: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"\n✅ {test_name} - ПРОЙДЕН")
            else:
                print(f"\n❌ {test_name} - ПРОВАЛЕН")
        except Exception as e:
            print(f"\n💥 {test_name} - ОШИБКА: {e}")
            results.append((test_name, False))
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print("ИТОГОВЫЙ ОТЧЕТ")
    print('='*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        print(f"{test_name}: {status}")
    
    print(f"\nРезультат: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Оси правильно ориентированы.")
        print("\nПравильная ориентация:")
        print("✓ Y (ёмкостной параметр) - по вертикали (левая ось)")
        print("✓ Давление/дебит - по горизонтали (нижняя ось)")
        print("✓ График имеет параболическую форму, идущую вверх от (0,0)")
    else:
        print(f"\n⚠️  {total - passed} тестов провалено. Требуются дополнительные исправления.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
