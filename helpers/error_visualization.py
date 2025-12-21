#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация зависимости ошибки метода/подсистемы от объёма данных/качества
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import seaborn as sns

from helpers.math_error_logger import get_logger, MathErrorLogger


class ErrorVisualizer:
    """Класс для визуализации ошибок"""
    
    def __init__(self, output_dir: str = "test_results", log_file: str = "math_errors.jsonl"):
        """
        :param output_dir: директория для сохранения графиков
        :param log_file: имя файла лога для загрузки данных (по умолчанию math_errors.jsonl)
                        Можно указать "math_errors_reverse.jsonl" для обратных данных
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.log_file = log_file
        # Создаём логгер с указанным файлом
        self.logger = MathErrorLogger(log_dir="logs", log_file=log_file)
    
    def plot_error_vs_volume(
        self,
        df: Optional[pd.DataFrame] = None,
        error_type: str = 'rmse',
        save_path: Optional[str] = None
    ) -> None:
        """
        График зависимости ошибки от объёма данных
        
        :param df: DataFrame с ошибками (если None, загружается из логов)
        :param error_type: тип ошибки для анализа
        :param save_path: путь для сохранения графика
        """
        if df is None:
            df = self.logger.load_errors()
        
        if df.empty:
            print("Нет данных для визуализации")
            return
        
        # Фильтруем по типу ошибки и убираем исключения
        df_filtered = df[
            (df['error_type'] == error_type) &
            (df['data_volume'].notna()) &
            (df['error_value'].notna()) &
            (df['error_value'] < np.inf)
        ].copy()
        
        if df_filtered.empty:
            print(f"Нет данных для типа ошибки '{error_type}'")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Зависимость ошибки {error_type.upper()} от объёма данных', fontsize=14)
        
        # 1. Общий график по всем подсистемам
        ax = axes[0, 0]
        ax.scatter(df_filtered['data_volume'], df_filtered['error_value'], alpha=0.6, s=50)
        ax.set_xlabel('Объём данных (количество точек)')
        ax.set_ylabel(f'Ошибка ({error_type})')
        ax.set_title('Все подсистемы')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Линейная регрессия
        if len(df_filtered) > 1:
            x = np.log10(df_filtered['data_volume'].values)
            y = np.log10(df_filtered['error_value'].values)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() > 1:
                coeffs = np.polyfit(x[valid], y[valid], 1)
                x_fit = np.logspace(
                    np.log10(df_filtered['data_volume'].min()),
                    np.log10(df_filtered['data_volume'].max()),
                    100
                )
                y_fit = 10 ** np.polyval(coeffs, np.log10(x_fit))
                ax.plot(x_fit, y_fit, 'r--', alpha=0.7, label=f'Тренд (slope={coeffs[0]:.2f})')
                ax.legend()
        
        # 2. По подсистемам
        ax = axes[0, 1]
        subsystems = df_filtered['subsystem'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(subsystems)))
        
        for subsystem, color in zip(subsystems, colors):
            subset = df_filtered[df_filtered['subsystem'] == subsystem]
            ax.scatter(subset['data_volume'], subset['error_value'],
                      label=subsystem, alpha=0.6, s=50, color=color)
        
        ax.set_xlabel('Объём данных')
        ax.set_ylabel(f'Ошибка ({error_type})')
        ax.set_title('По подсистемам')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. По методам
        ax = axes[1, 0]
        methods = df_filtered['method'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        for method, color in zip(methods, colors):
            subset = df_filtered[df_filtered['method'] == method]
            ax.scatter(subset['data_volume'], subset['error_value'],
                      label=method, alpha=0.6, s=50, color=color)
        
        ax.set_xlabel('Объём данных')
        ax.set_ylabel(f'Ошибка ({error_type})')
        ax.set_title('По методам')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Боксплот по диапазонам объёма
        ax = axes[1, 1]
        df_filtered['volume_bin'] = pd.cut(
            df_filtered['data_volume'],
            bins=np.logspace(
                np.log10(df_filtered['data_volume'].min()),
                np.log10(df_filtered['data_volume'].max()),
                6
            ),
            labels=[f'10^{i:.1f}' for i in np.linspace(
                np.log10(df_filtered['data_volume'].min()),
                np.log10(df_filtered['data_volume'].max()),
                5
            )]
        )
        
        box_data = [df_filtered[df_filtered['volume_bin'] == bin]['error_value'].values
                    for bin in df_filtered['volume_bin'].cat.categories]
        
        ax.boxplot(box_data, labels=df_filtered['volume_bin'].cat.categories)
        ax.set_xlabel('Диапазон объёма данных')
        ax.set_ylabel(f'Ошибка ({error_type})')
        ax.set_title('Распределение ошибок по диапазонам объёма')
        ax.set_yscale('log')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"error_vs_volume_{error_type}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
        plt.close()
    
    def plot_error_vs_quality(
        self,
        df: Optional[pd.DataFrame] = None,
        error_type: str = 'rmse',
        save_path: Optional[str] = None
    ) -> None:
        """
        График зависимости ошибки от качества данных
        
        :param df: DataFrame с ошибками
        :param error_type: тип ошибки
        :param save_path: путь для сохранения
        """
        if df is None:
            df = self.logger.load_errors()
        
        if df.empty:
            print("Нет данных для визуализации")
            return
        
        df_filtered = df[
            (df['error_type'] == error_type) &
            (df['data_quality'].notna()) &
            (df['error_value'].notna()) &
            (df['error_value'] < np.inf)
        ].copy()
        
        if df_filtered.empty:
            print(f"Нет данных с качеством для типа ошибки '{error_type}'")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Зависимость ошибки {error_type.upper()} от качества данных', fontsize=14)
        
        # 1. Общий график
        ax = axes[0, 0]
        ax.scatter(df_filtered['data_quality'], df_filtered['error_value'], alpha=0.6, s=50)
        ax.set_xlabel('Качество данных (0-1, где 1 - идеальное)')
        ax.set_ylabel(f'Ошибка ({error_type})')
        ax.set_title('Все подсистемы')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Полиномиальная регрессия
        if len(df_filtered) > 2:
            x = df_filtered['data_quality'].values
            y = np.log10(df_filtered['error_value'].values)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() > 2:
                coeffs = np.polyfit(x[valid], y[valid], 2)
                x_fit = np.linspace(df_filtered['data_quality'].min(),
                                   df_filtered['data_quality'].max(), 100)
                y_fit = 10 ** np.polyval(coeffs, x_fit)
                ax.plot(x_fit, y_fit, 'r--', alpha=0.7, label='Тренд (полином 2-й степени)')
                ax.legend()
        
        # 2. По подсистемам
        ax = axes[0, 1]
        subsystems = df_filtered['subsystem'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(subsystems)))
        
        for subsystem, color in zip(subsystems, colors):
            subset = df_filtered[df_filtered['subsystem'] == subsystem]
            ax.scatter(subset['data_quality'], subset['error_value'],
                      label=subsystem, alpha=0.6, s=50, color=color)
        
        ax.set_xlabel('Качество данных')
        ax.set_ylabel(f'Ошибка ({error_type})')
        ax.set_title('По подсистемам')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. По методам
        ax = axes[1, 0]
        methods = df_filtered['method'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        for method, color in zip(methods, colors):
            subset = df_filtered[df_filtered['method'] == method]
            ax.scatter(subset['data_quality'], subset['error_value'],
                      label=method, alpha=0.6, s=50, color=color)
        
        ax.set_xlabel('Качество данных')
        ax.set_ylabel(f'Ошибка ({error_type})')
        ax.set_title('По методам')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Боксплот по диапазонам качества
        ax = axes[1, 1]
        df_filtered['quality_bin'] = pd.cut(
            df_filtered['data_quality'],
            bins=5,
            labels=['Очень низкое', 'Низкое', 'Среднее', 'Высокое', 'Очень высокое']
        )
        
        box_data = [df_filtered[df_filtered['quality_bin'] == bin]['error_value'].values
                    for bin in df_filtered['quality_bin'].cat.categories]
        
        ax.boxplot(box_data, labels=df_filtered['quality_bin'].cat.categories)
        ax.set_xlabel('Диапазон качества данных')
        ax.set_ylabel(f'Ошибка ({error_type})')
        ax.set_title('Распределение ошибок по диапазонам качества')
        ax.set_yscale('log')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"error_vs_quality_{error_type}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
        plt.close()
    
    def plot_error_heatmap(
        self,
        df: Optional[pd.DataFrame] = None,
        error_type: str = 'rmse',
        save_path: Optional[str] = None
    ) -> None:
        """
        Тепловая карта ошибок: метод vs подсистема
        
        :param df: DataFrame с ошибками
        :param error_type: тип ошибки
        :param save_path: путь для сохранения
        """
        if df is None:
            df = self.logger.load_errors()
        
        if df.empty:
            print("Нет данных для визуализации")
            return
        
        df_filtered = df[
            (df['error_type'] == error_type) &
            (df['error_value'].notna()) &
            (df['error_value'] < np.inf)
        ].copy()
        
        if df_filtered.empty:
            print(f"Нет данных для типа ошибки '{error_type}'")
            return
        
        # Группируем по подсистеме и методу, берём среднее значение ошибки
        heatmap_data = df_filtered.groupby(['subsystem', 'method'])['error_value'].mean().unstack(fill_value=np.nan)
        
        if heatmap_data.empty:
            print("Недостаточно данных для построения тепловой карты")
            return
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.2e', cmap='YlOrRd', cbar_kws={'label': f'Ошибка ({error_type})'})
        plt.title(f'Тепловая карта ошибок: {error_type.upper()}')
        plt.xlabel('Метод')
        plt.ylabel('Подсистема')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"error_heatmap_{error_type}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
        plt.close()
    
    def generate_all_visualizations(self, suffix: str = "") -> None:
        """
        Генерирует все визуализации для всех типов ошибок
        
        :param suffix: суффикс для имен файлов (например, "_reverse" для обратных данных)
        """
        df = self.logger.load_errors()
        
        if df.empty:
            print(f"Нет данных для визуализации в файле {self.log_file}")
            return
        
        error_types = df[df['error_type'] != 'exception']['error_type'].unique()
        
        for error_type in error_types:
            print(f"\nГенерация визуализаций для {error_type}...")
            
            # Проверяем наличие данных по объёму
            if df[df['error_type'] == error_type]['data_volume'].notna().any():
                save_path = self.output_dir / f"error_vs_volume_{error_type}{suffix}.png"
                self.plot_error_vs_volume(df=df, error_type=error_type, save_path=str(save_path))
            
            # Проверяем наличие данных по качеству
            if df[df['error_type'] == error_type]['data_quality'].notna().any():
                save_path = self.output_dir / f"error_vs_quality_{error_type}{suffix}.png"
                self.plot_error_vs_quality(df=df, error_type=error_type, save_path=str(save_path))
            
            # Тепловая карта
            save_path = self.output_dir / f"error_heatmap_{error_type}{suffix}.png"
            self.plot_error_heatmap(df=df, error_type=error_type, save_path=str(save_path))
        
        print(f"\n✅ Все визуализации сгенерированы (источник: {self.log_file})")


if __name__ == "__main__":
    visualizer = ErrorVisualizer()
    visualizer.generate_all_visualizations()

