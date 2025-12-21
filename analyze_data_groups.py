#!/usr/bin/env python3
"""
Скрипт для анализа групп данных в файле с данными скважин.
Помогает понять структуру данных и количество групп с разными параметрами.
"""

import pandas as pd
import sys
import os
from typing import Dict

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.parse_well_data import analyze_well_groups, group_data_by_well_parameters


def analyze_file(file_path: str):
    """Анализирует файл с данными скважин"""
    print(f"Анализ файла: {file_path}")
    print("=" * 50)
    
    try:
        # Читаем файл
        if file_path.lower().endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
        
        print(f"Общее количество строк: {len(df)}")
        print(f"Колонки: {list(df.columns)}")
        print()
        
        # Анализируем группы
        analysis = analyze_well_groups(df)
        
        print(f"Найдено групп данных: {analysis['total_groups']}")
        print(f"Общее количество точек: {analysis['total_points']}")
        print()
        
        # Детальная информация по группам
        for i, group_info in enumerate(analysis['groups_info'], 1):
            print(f"Группа {i} ({group_info['group_id']}):")
            print(f"  Количество точек: {group_info['points_count']}")
            print(f"  Skin: {group_info['skin']:.6f}")
            print(f"  Толщина (h): {group_info['thickness']:.3f}")
            print(f"  Количество трещин (N): {group_info['fractures']}")
            print(f"  Ширина трещины (W): {group_info['fracture_width']:.6f}")
            print(f"  Длина трещины (L): {group_info['fracture_length']:.3f}")
            print(f"  Отношение a/L: {group_info['a_l_ratio']:.6f}")
            print(f"  Временной диапазон: {group_info['time_range'][0]:.2f} - {group_info['time_range'][1]:.2f} ч")
            print(f"  Диапазон давления: {group_info['pressure_range'][0]:.2f} - {group_info['pressure_range'][1]:.2f} атм")
            print(f"  Диапазон дебита: {group_info['flow_rate_range'][0]:.2f} - {group_info['flow_rate_range'][1]:.2f} м³/сут")
            print()
        
        # Статистика по параметрам
        print("Статистика по параметрам:")
        print("-" * 30)
        for param in ['Skin', 'h', 'N', 'W', 'L', 'a/L']:
            if param in df.columns:
                values = df[param].unique()
                print(f"{param}: {len(values)} уникальных значений")
                if len(values) <= 10:  # Показываем значения, если их немного
                    print(f"  Значения: {sorted(values)}")
                else:
                    print(f"  Диапазон: {df[param].min():.6f} - {df[param].max():.6f}")
                print()
        
        # Рекомендации
        print("Рекомендации:")
        print("-" * 15)
        if analysis['total_groups'] == 1:
            print("✅ Все данные имеют одинаковые параметры скважины")
            print("   Можно использовать стандартный парсер")
        else:
            print(f"⚠️  Найдено {analysis['total_groups']} групп с разными параметрами")
            print("   Рекомендуется использовать новый парсер с группировкой")
            
            # Проверяем, есть ли группы с малым количеством точек
            small_groups = [g for g in analysis['groups_info'] if g['points_count'] < 5]
            if small_groups:
                print(f"⚠️  {len(small_groups)} групп содержат менее 5 точек")
                print("   Эти группы могут быть исключены из анализа")
        
        return analysis
        
    except Exception as e:
        print(f"Ошибка анализа файла: {e}")
        return None


def suggest_grouping_strategy(analysis: Dict):
    """Предлагает стратегию группировки данных"""
    if not analysis:
        return
    
    print("\nСтратегии группировки:")
    print("=" * 25)
    
    groups_info = analysis['groups_info']
    
    # Группировка по Skin
    skin_groups = {}
    for group in groups_info:
        skin = group['skin']
        if skin not in skin_groups:
            skin_groups[skin] = []
        skin_groups[skin].append(group)
    
    print(f"1. Группировка по Skin: {len(skin_groups)} групп")
    for skin, groups in skin_groups.items():
        total_points = sum(g['points_count'] for g in groups)
        print(f"   Skin={skin:.6f}: {len(groups)} подгрупп, {total_points} точек")
    
    # Группировка по количеству трещин
    n_groups = {}
    for group in groups_info:
        n = group['fractures']
        if n not in n_groups:
            n_groups[n] = []
        n_groups[n].append(group)
    
    print(f"\n2. Группировка по количеству трещин: {len(n_groups)} групп")
    for n, groups in n_groups.items():
        total_points = sum(g['points_count'] for g in groups)
        print(f"   N={n}: {len(groups)} подгрупп, {total_points} точек")
    
    # Группировка по a/L
    al_groups = {}
    for group in groups_info:
        al = round(group['a_l_ratio'], 4)  # Округляем для группировки
        if al not in al_groups:
            al_groups[al] = []
        al_groups[al].append(group)
    
    print(f"\n3. Группировка по a/L (округлено до 4 знаков): {len(al_groups)} групп")
    for al, groups in al_groups.items():
        total_points = sum(g['points_count'] for g in groups)
        print(f"   a/L={al:.4f}: {len(groups)} подгрупп, {total_points} точек")


def main():
    """Основная функция"""
    if len(sys.argv) != 2:
        print("Использование: python analyze_data_groups.py <путь_к_файлу>")
        print("Пример: python analyze_data_groups.py well_data.csv")
        return
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Файл не найден: {file_path}")
        return
    
    analysis = analyze_file(file_path)
    if analysis:
        suggest_grouping_strategy(analysis)


if __name__ == "__main__":
    main()
