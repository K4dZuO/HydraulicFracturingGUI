#!/usr/bin/env python3
"""
Скрипт для запуска тестов анализа ГРП
"""

import subprocess
import sys
import os

def run_tests():
    """Запуск всех тестов"""
    print("Запуск тестов анализа ГРП...")
    
    # Команда для запуска pytest
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/",
        "-v",
        "--tb=short",
        "--cov=helpers",
        "--cov=schemas",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode == 0
    except Exception as e:
        print(f"Ошибка запуска тестов: {e}")
        return False

def run_unit_tests():
    """Запуск только unit тестов"""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_well_analysis.py::TestWellDataSchemas",
        "tests/test_well_analysis.py::TestParseWellData", 
        "tests/test_well_analysis.py::TestGRPAnalysis",
        "tests/test_well_analysis.py::TestMLMethods",
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode == 0
    except Exception as e:
        print(f"Ошибка запуска unit тестов: {e}")
        return False

def run_integration_tests():
    """Запуск только интеграционных тестов"""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_well_analysis.py::TestIntegration",
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode == 0
    except Exception as e:
        print(f"Ошибка запуска интеграционных тестов: {e}")
        return False

def run_adequacy_tests():
    """Запуск тестов на физическую адекватность"""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_physical_adequacy.py",
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode == 0
    except Exception as e:
        print(f"Ошибка запуска тестов адекватности: {e}")
        return False

def run_specific_test(test_name):
    """Запуск конкретного теста"""
    cmd = [
        sys.executable, "-m", "pytest", 
        f"tests/test_well_analysis.py::{test_name}",
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode == 0
    except Exception as e:
        print(f"Ошибка запуска теста {test_name}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "unit":
            success = run_unit_tests()
        elif test_type == "integration":
            success = run_integration_tests()
        elif test_type == "adequacy":
            success = run_adequacy_tests()
        else:
            success = run_specific_test(test_type)
    else:
        success = run_tests()
    
    if success:
        print("\n✅ Все тесты прошли успешно!")
    else:
        print("\n❌ Некоторые тесты не прошли")
        sys.exit(1)
