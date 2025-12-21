#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для создания тестовых данных с "побитыми" значениями для проверки ML-функций
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Создаем директорию для тестовых данных
TEST_DATA_DIR = Path("tests/test_data")
TEST_DATA_DIR.mkdir(exist_ok=True, parents=True)


def create_synthetic_well_data(n_points=100, n_wells=3):
    """
    Создает синтетические данные скважин с различными проблемами:
    - Выбросы
    - Пропуски (NaN)
    - Шум
    - Ступенчатые изменения
    - Тренды
    """
    data_list = []
    
    for well_id in range(n_wells):
        # Базовые параметры скважины
        skin = 0.05 + well_id * 0.1  # Разные значения скин-фактора для каждой скважины
        h = 10.0
        N = 5 + well_id
        W = 1000.0 + well_id * 100
        L = 200.0 + well_id * 50
        a_L = 0.5
        
        # Временной ряд
        time = np.linspace(0, 24, n_points)
        
        # Базовое давление с экспоненциальным спадом
        pressure_base = 300 * np.exp(-0.01 * time)
        
        # Базовый дебит
        flow_rate_base = 100 * np.exp(-0.005 * time)
        
        # Добавляем различные проблемы в зависимости от ID скважины
        if well_id == 0:
            # Скважина 1: Выбросы
            outlier_indices = np.random.choice(n_points, size=5, replace=False)
            pressure = pressure_base.copy()
            flow_rate = flow_rate_base.copy()
            
            pressure[outlier_indices] = pressure[outlier_indices] * (1 + np.random.uniform(-0.3, 0.3, size=len(outlier_indices)))
            flow_rate[outlier_indices] = flow_rate[outlier_indices] * (1 + np.random.uniform(-0.3, 0.3, size=len(outlier_indices)))
            
            problem_type = "outliers"
            
        elif well_id == 1:
            # Скважина 2: Пропуски (NaN)
            gap_indices = np.random.choice(n_points, size=10, replace=False)
            pressure = pressure_base.copy()
            flow_rate = flow_rate_base.copy()
            
            pressure[gap_indices] = np.nan
            flow_rate[gap_indices] = np.nan
            
            problem_type = "gaps"
            
        else:
            # Скважина 3: Шум
            pressure = pressure_base + np.random.normal(0, 5, size=n_points)
            flow_rate = flow_rate_base + np.random.normal(0, 3, size=n_points)
            
            problem_type = "noise"
        
        # Создаем DataFrame для скважины
        well_data = pd.DataFrame({
            'Skin': [skin] * n_points,
            'h': [h] * n_points,
            'N': [N] * n_points,
            'W': [W] * n_points,
            'L': [L] * n_points,
            'a/L': [a_L] * n_points,
            'ElemIdx': range(1, n_points + 1),
            'X': [0.0] * n_points,
            'Y': [0.0] * n_points,
            't': time,
            'P': pressure,
            'dP': np.gradient(pressure),
            'Q': flow_rate,
            'problem_type': [problem_type] * n_points
        })
        
        data_list.append(well_data)
    
    # Объединяем все данные
    all_data = pd.concat(data_list, ignore_index=True)
    
    # Сохраняем в CSV
    csv_path = TEST_DATA_DIR / "synthetic_well_data.csv"
    all_data.to_csv(csv_path, index=False)
    print(f"Синтетические данные сохранены в {csv_path}")
    
    # Визуализация данных
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    for well_id in range(n_wells):
        well_data = data_list[well_id]
        plt.plot(well_data['t'], well_data['P'], label=f"Скважина {well_id+1}: {well_data['problem_type'].iloc[0]}")
    plt.title("Давление vs Время")
    plt.xlabel("Время, ч")
    plt.ylabel("Давление, атм")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for well_id in range(n_wells):
        well_data = data_list[well_id]
        plt.plot(well_data['t'], well_data['Q'], label=f"Скважина {well_id+1}: {well_data['problem_type'].iloc[0]}")
    plt.title("Дебит vs Время")
    plt.xlabel("Время, ч")
    plt.ylabel("Дебит, м³/сут")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(TEST_DATA_DIR / "synthetic_data_visualization.png")
    plt.close()
    
    return all_data


def create_test_cases_for_ml():
    """
    Создает специальные тестовые случаи для проверки ML-функций
    """
    # Базовый временной ряд
    x = np.linspace(0, 10, 100)
    
    # Различные типы сигналов
    signals = {
        # Синусоида с шумом
        'sine_noisy': np.sin(x) + np.random.normal(0, 0.1, size=len(x)),
        
        # Линейный тренд с шумом
        'linear_trend': 0.5 * x + np.random.normal(0, 0.1, size=len(x)),
        
        # Ступенчатая функция с шумом
        'step_function': np.concatenate([
            np.zeros(25) + np.random.normal(0, 0.05, size=25),
            np.ones(25) + np.random.normal(0, 0.05, size=25),
            2 * np.ones(25) + np.random.normal(0, 0.05, size=25),
            3 * np.ones(25) + np.random.normal(0, 0.05, size=25)
        ]),
        
        # Экспоненциальный спад с шумом
        'exponential_decay': np.exp(-0.3 * x) + np.random.normal(0, 0.05, size=len(x)),
        
        # Логарифмический рост с шумом
        'logarithmic_growth': np.log(1 + x) + np.random.normal(0, 0.05, size=len(x))
    }
    
    # Создаем DataFrame
    df = pd.DataFrame({'x': x})
    
    for name, signal in signals.items():
        df[name] = signal
        
        # Создаем версию с выбросами
        outliers = signal.copy()
        outlier_indices = np.random.choice(len(x), size=5, replace=False)
        outliers[outlier_indices] = outliers[outlier_indices] + np.random.choice([-1, 1], size=5) * 2
        df[f"{name}_outliers"] = outliers
        
        # Создаем версию с пропусками
        gaps = signal.copy()
        gap_indices = np.random.choice(len(x), size=10, replace=False)
        gaps[gap_indices] = np.nan
        df[f"{name}_gaps"] = gaps
    
    # Сохраняем в CSV
    csv_path = TEST_DATA_DIR / "ml_test_cases.csv"
    df.to_csv(csv_path, index=False)
    print(f"Тестовые случаи для ML сохранены в {csv_path}")
    
    # Визуализация
    plt.figure(figsize=(15, 10))
    
    for i, name in enumerate(signals.keys()):
        plt.subplot(len(signals), 3, i*3 + 1)
        plt.plot(x, df[name])
        plt.title(f"{name}")
        plt.grid(True)
        
        plt.subplot(len(signals), 3, i*3 + 2)
        plt.plot(x, df[f"{name}_outliers"])
        plt.title(f"{name}_outliers")
        plt.grid(True)
        
        plt.subplot(len(signals), 3, i*3 + 3)
        plt.plot(x, df[f"{name}_gaps"])
        plt.title(f"{name}_gaps")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(TEST_DATA_DIR / "ml_test_cases_visualization.png")
    plt.close()
    
    return df


if __name__ == "__main__":
    print("Создание тестовых данных для ML-функций...")
    
    # Создаем синтетические данные скважин
    well_data = create_synthetic_well_data(n_points=100, n_wells=3)
    
    # Создаем тестовые случаи для ML
    ml_test_cases = create_test_cases_for_ml()
    
    print("Готово! Тестовые данные созданы.")
