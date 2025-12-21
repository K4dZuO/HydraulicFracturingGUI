#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для запуска комплексных тестов с генерацией отчета
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def run_test_suite(markers: str, description: str) -> dict:
    """Запускает набор тестов с заданными маркерами"""
    print(f"\n{'='*80}")
    print(f"🧪 {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_comprehensive_performance.py",
        "-v", "-s",
        "-m", markers
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    elapsed_time = time.time() - start_time
    
    return {
        'description': description,
        'exit_code': result.returncode,
        'elapsed_time': elapsed_time,
        'success': result.returncode == 0
    }


def main():
    """Основная функция запуска тестов"""
    print(f"\n{'#'*80}")
    print(f"# КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ПРИЛОЖЕНИЯ")
    print(f"# Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")
    
    results = []
    
    # 1. Быстрые тесты (unit + performance без long-running)
    results.append(run_test_suite(
        "(unit or performance) and not longrun and not slow",
        "Быстрые тесты производительности и точности"
    ))
    
    # 2. Тесты памяти
    results.append(run_test_suite(
        "memory and not longrun",
        "Тесты потребления памяти"
    ))
    
    # 3. Нагрузочные тесты
    results.append(run_test_suite(
        "load and not longrun",
        "Нагрузочные тесты (failure rate)"
    ))
    
    # 4. Долгие тесты стабильности (опционально)
    print(f"\n{'='*80}")
    print("⚠️  ВНИМАНИЕ: Следующие тесты могут занять 10-20 минут")
    print("='*80}")
    
    response = input("\nЗапустить long-running тесты стабильности? (y/N): ")
    
    if response.lower() == 'y':
        results.append(run_test_suite(
            "longrun",
            "Тесты долгой стабильности (деградация производительности)"
        ))
    else:
        print("⏭️  Long-running тесты пропущены")
    
    # Генерация итогового отчета
    print(f"\n{'#'*80}")
    print(f"# ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'#'*80}\n")
    
    total_time = sum(r['elapsed_time'] for r in results)
    passed_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - passed_count
    
    print(f"Всего наборов тестов: {len(results)}")
    print(f"✅ Успешно: {passed_count}")
    print(f"❌ Провалено: {failed_count}")
    print(f"⏱️  Общее время: {total_time:.2f} секунд\n")
    
    print("Детали по наборам:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{i}. {result['description']}")
        print(f"   Статус: {status}")
        print(f"   Время: {result['elapsed_time']:.2f}с")
        print()
    
    # Сохраняем отчет в файл
    report_file = Path("test_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Комплексное тестирование\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Всего наборов: {len(results)}\n")
        f.write(f"Успешно: {passed_count}\n")
        f.write(f"Провалено: {failed_count}\n")
        f.write(f"Общее время: {total_time:.2f}с\n\n")
        
        for i, result in enumerate(results, 1):
            status = "PASS" if result['success'] else "FAIL"
            f.write(f"{i}. {result['description']}: {status} ({result['elapsed_time']:.2f}с)\n")
    
    print(f"📄 Отчет сохранен в: {report_file.absolute()}")
    
    # Возвращаем код завершения
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

