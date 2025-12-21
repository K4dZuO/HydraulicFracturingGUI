#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для логирования математических ошибок в вычислениях ГРП
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd


class MathErrorLogger:
    """Логгер математических ошибок с сохранением в файл и метаданными"""
    
    def __init__(self, log_dir: str = "logs", log_file: str = "math_errors.jsonl"):
        """
        :param log_dir: директория для логов
        :param log_file: имя файла лога (JSONL формат)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / log_file
        
        # Настройка Python logging для консоли
        self.logger = logging.getLogger('math_errors')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_dir / "math_errors.log")
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(handler)
    
    def log_error(
        self,
        subsystem: str,
        method: str,
        error_type: str,
        error_value: float,
        error_message: str,
        data_volume: Optional[int] = None,
        data_quality: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Логирует математическую ошибку
        
        :param subsystem: подсистема (например, 'interpolation', 'filtering', 'extrapolation')
        :param method: метод (например, 'linear', 'rbf', 'gp')
        :param error_type: тип ошибки ('rmse', 'mae', 'mape', 'max_error', 'exception')
        :param error_value: значение ошибки
        :param error_message: описание ошибки
        :param data_volume: объём данных (количество точек)
        :param data_quality: качество данных (0-1, где 1 - идеальное)
        :param metadata: дополнительные метаданные
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "subsystem": subsystem,
            "method": method,
            "error_type": error_type,
            "error_value": float(error_value) if np.isfinite(error_value) else None,
            "error_message": error_message,
            "data_volume": data_volume,
            "data_quality": data_quality,
            "metadata": metadata or {}
        }
        
        # Записываем в JSONL файл
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # Логируем в консольный лог
        self.logger.warning(
            f"{subsystem}/{method}: {error_type}={error_value:.6e} - {error_message}"
        )
    
    def log_computation_error(
        self,
        subsystem: str,
        method: str,
        exception: Exception,
        data_volume: Optional[int] = None,
        data_quality: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Логирует исключение при вычислениях"""
        self.log_error(
            subsystem=subsystem,
            method=method,
            error_type="exception",
            error_value=float('inf'),
            error_message=str(exception),
            data_volume=data_volume,
            data_quality=data_quality,
            metadata={"exception_type": type(exception).__name__, "context": context or {}}
        )
    
    def log_metric_error(
        self,
        subsystem: str,
        method: str,
        metrics: Dict[str, float],
        data_volume: Optional[int] = None,
        data_quality: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Логирует метрики ошибок (RMSE, MAE, MAPE и т.д.)"""
        for metric_name, metric_value in metrics.items():
            if np.isfinite(metric_value):
                self.log_error(
                    subsystem=subsystem,
                    method=method,
                    error_type=metric_name.lower(),
                    error_value=metric_value,
                    error_message=f"Metric {metric_name}",
                    data_volume=data_volume,
                    data_quality=data_quality,
                    metadata=metadata
                )
    
    def load_errors(self) -> pd.DataFrame:
        """Загружает все логированные ошибки в DataFrame"""
        if not self.log_file.exists():
            return pd.DataFrame()
        
        entries = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if not entries:
            return pd.DataFrame()
        
        df = pd.DataFrame(entries)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по ошибкам"""
        df = self.load_errors()
        
        if df.empty:
            return {"total_errors": 0}
        
        stats = {
            "total_errors": len(df),
            "by_subsystem": df.groupby('subsystem').size().to_dict(),
            "by_method": df.groupby('method').size().to_dict(),
            "by_error_type": df.groupby('error_type').size().to_dict(),
            "error_values": {}
        }
        
        # Статистика по значениям ошибок (только для числовых)
        numeric_errors = df[df['error_value'].notna() & df['error_type'] != 'exception']
        if not numeric_errors.empty:
            stats["error_values"] = {
                "mean": float(numeric_errors['error_value'].mean()),
                "median": float(numeric_errors['error_value'].median()),
                "std": float(numeric_errors['error_value'].std()),
                "min": float(numeric_errors['error_value'].min()),
                "max": float(numeric_errors['error_value'].max())
            }
        
        return stats


# Глобальный экземпляр логгера
_global_logger: Optional[MathErrorLogger] = None


def get_logger() -> MathErrorLogger:
    """Возвращает глобальный экземпляр логгера"""
    global _global_logger
    if _global_logger is None:
        _global_logger = MathErrorLogger()
    return _global_logger


def log_math_error(*args, **kwargs):
    """Удобная функция для логирования ошибок"""
    get_logger().log_error(*args, **kwargs)


def log_computation_error(*args, **kwargs):
    """Удобная функция для логирования исключений"""
    get_logger().log_computation_error(*args, **kwargs)


def log_metric_error(*args, **kwargs):
    """Удобная функция для логирования метрик"""
    get_logger().log_metric_error(*args, **kwargs)

