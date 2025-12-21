#!/usr/bin/env python3
"""
Анализ покрытия кода тестами
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple

class TestCoverageAnalyzer:
    """Анализатор покрытия кода тестами"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.source_files = []
        self.test_files = []
        self.coverage_report = {}
        
    def find_source_files(self) -> List[Path]:
        """Находит все Python файлы в проекте"""
        source_files = []
        
        # Основные модули
        for pattern in ["*.py"]:
            source_files.extend(self.project_root.glob(pattern))
        
        # Модули в helpers/
        helpers_dir = self.project_root / "helpers"
        if helpers_dir.exists():
            source_files.extend(helpers_dir.glob("*.py"))
        
        # Модули в schemas/
        schemas_dir = self.project_root / "schemas"
        if schemas_dir.exists():
            source_files.extend(schemas_dir.glob("*.py"))
        
        # Исключаем тестовые файлы и __pycache__
        source_files = [
            f for f in source_files 
            if not f.name.startswith("test_") 
            and not f.name.startswith("__")
            and f.name != "ui.py"  # Автогенерированный файл
        ]
        
        self.source_files = source_files
        return source_files
    
    def find_test_files(self) -> List[Path]:
        """Находит все тестовые файлы"""
        test_files = []
        
        # Тесты в папке tests/
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            test_files.extend(tests_dir.glob("test_*.py"))
        
        # Тесты в корне проекта
        test_files.extend(self.project_root.glob("test_*.py"))
        
        self.test_files = test_files
        return test_files
    
    def extract_functions_and_classes(self, file_path: Path) -> Dict[str, List[str]]:
        """Извлекает функции и классы из Python файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            functions = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
            
            return {
                'functions': functions,
                'classes': classes
            }
        except Exception as e:
            print(f"Ошибка при анализе {file_path}: {e}")
            return {'functions': [], 'classes': []}
    
    def analyze_test_coverage(self) -> Dict:
        """Анализирует покрытие тестами"""
        print("🔍 Анализ покрытия кода тестами...")
        print("=" * 60)
        
        # Находим файлы
        source_files = self.find_source_files()
        test_files = self.find_test_files()
        
        print(f"📁 Найдено исходных файлов: {len(source_files)}")
        print(f"🧪 Найдено тестовых файлов: {len(test_files)}")
        print()
        
        # Анализируем каждый исходный файл
        total_functions = 0
        total_classes = 0
        tested_functions = 0
        tested_classes = 0
        
        coverage_details = {}
        
        for source_file in source_files:
            print(f"📄 Анализ файла: {source_file.name}")
            
            # Извлекаем функции и классы
            source_content = self.extract_functions_and_classes(source_file)
            functions = source_content['functions']
            classes = source_content['classes']
            
            total_functions += len(functions)
            total_classes += len(classes)
            
            # Проверяем, какие функции/классы тестируются
            file_tested_functions = 0
            file_tested_classes = 0
            
            for test_file in test_files:
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        test_content = f.read()
                    
                    # Проверяем импорты и использование
                    for func in functions:
                        if func in test_content or f"test_{func}" in test_content:
                            file_tested_functions += 1
                    
                    for cls in classes:
                        if cls in test_content or f"test_{cls.lower()}" in test_content:
                            file_tested_classes += 1
                            
                except Exception as e:
                    print(f"  ⚠️  Ошибка при анализе теста {test_file.name}: {e}")
            
            tested_functions += file_tested_functions
            tested_classes += file_tested_classes
            
            # Сохраняем детали
            coverage_details[source_file.name] = {
                'functions': functions,
                'classes': classes,
                'tested_functions': file_tested_functions,
                'tested_classes': file_tested_classes,
                'function_coverage': file_tested_functions / len(functions) * 100 if functions else 100,
                'class_coverage': file_tested_classes / len(classes) * 100 if classes else 100
            }
            
            print(f"  📊 Функции: {len(functions)} (тестируется: {file_tested_functions})")
            print(f"  📊 Классы: {len(classes)} (тестируется: {file_tested_classes})")
            print()
        
        # Общая статистика
        overall_function_coverage = (tested_functions / total_functions * 100) if total_functions > 0 else 100
        overall_class_coverage = (tested_classes / total_classes * 100) if total_classes > 0 else 100
        
        self.coverage_report = {
            'total_files': len(source_files),
            'total_test_files': len(test_files),
            'total_functions': total_functions,
            'total_classes': total_classes,
            'tested_functions': tested_functions,
            'tested_classes': tested_classes,
            'function_coverage': overall_function_coverage,
            'class_coverage': overall_class_coverage,
            'details': coverage_details
        }
        
        return self.coverage_report
    
    def print_coverage_report(self):
        """Выводит отчет о покрытии"""
        if not self.coverage_report:
            self.analyze_test_coverage()
        
        report = self.coverage_report
        
        print("📊 ОТЧЕТ О ПОКРЫТИИ КОДА ТЕСТАМИ")
        print("=" * 60)
        print(f"📁 Исходных файлов: {report['total_files']}")
        print(f"🧪 Тестовых файлов: {report['total_test_files']}")
        print()
        print(f"🔧 Всего функций: {report['total_functions']}")
        print(f"✅ Тестируется функций: {report['tested_functions']}")
        print(f"📈 Покрытие функций: {report['function_coverage']:.1f}%")
        print()
        print(f"🏗️  Всего классов: {report['total_classes']}")
        print(f"✅ Тестируется классов: {report['tested_classes']}")
        print(f"📈 Покрытие классов: {report['class_coverage']:.1f}%")
        print()
        
        # Детали по файлам
        print("📋 ДЕТАЛИ ПО ФАЙЛАМ:")
        print("-" * 60)
        
        for filename, details in report['details'].items():
            print(f"📄 {filename}")
            print(f"  Функции: {details['tested_functions']}/{len(details['functions'])} ({details['function_coverage']:.1f}%)")
            print(f"  Классы: {details['tested_classes']}/{len(details['classes'])} ({details['class_coverage']:.1f}%)")
            
            if details['functions']:
                print(f"  📝 Функции: {', '.join(details['functions'])}")
            if details['classes']:
                print(f"  🏗️  Классы: {', '.join(details['classes'])}")
            print()
        
        # Рекомендации
        print("💡 РЕКОМЕНДАЦИИ:")
        print("-" * 60)
        
        if report['function_coverage'] < 80:
            print("⚠️  Покрытие функций ниже 80% - рекомендуется добавить больше unit-тестов")
        
        if report['class_coverage'] < 70:
            print("⚠️  Покрытие классов ниже 70% - рекомендуется добавить интеграционные тесты")
        
        # Анализ типов тестов
        self.analyze_test_types()
    
    def analyze_test_types(self):
        """Анализирует типы тестов"""
        print("\n🧪 АНАЛИЗ ТИПОВ ТЕСТОВ:")
        print("-" * 60)
        
        test_types = {
            'unit_tests': 0,
            'integration_tests': 0,
            'functional_tests': 0,
            'performance_tests': 0
        }
        
        for test_file in self.test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                # Простая эвристика для определения типа тестов
                if 'test_' in test_file.name and 'integration' not in test_file.name:
                    test_types['unit_tests'] += 1
                
                if 'integration' in test_file.name or 'integration' in content:
                    test_types['integration_tests'] += 1
                
                if 'functional' in test_file.name or 'functional' in content:
                    test_types['functional_tests'] += 1
                
                if 'performance' in test_file.name or 'performance' in content:
                    test_types['performance_tests'] += 1
                    
            except Exception as e:
                print(f"  ⚠️  Ошибка при анализе {test_file.name}: {e}")
        
        print(f"🔧 Unit-тесты: {test_types['unit_tests']}")
        print(f"🔗 Интеграционные тесты: {test_types['integration_tests']}")
        print(f"⚙️  Функциональные тесты: {test_types['functional_tests']}")
        print(f"⚡ Тесты производительности: {test_types['performance_tests']}")
        
        # Рекомендации по типам тестов
        print("\n💡 РЕКОМЕНДАЦИИ ПО ТИПАМ ТЕСТОВ:")
        if test_types['integration_tests'] == 0:
            print("⚠️  Отсутствуют интеграционные тесты - рекомендуется добавить")
        if test_types['functional_tests'] == 0:
            print("⚠️  Отсутствуют функциональные тесты - рекомендуется добавить")
        if test_types['performance_tests'] == 0:
            print("⚠️  Отсутствуют тесты производительности - рекомендуется добавить")

def main():
    """Главная функция"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    analyzer = TestCoverageAnalyzer(project_root)
    
    print("🔍 АНАЛИЗАТОР ПОКРЫТИЯ КОДА ТЕСТАМИ")
    print("=" * 60)
    print()
    
    analyzer.print_coverage_report()
    
    print("\n" + "=" * 60)
    print("✅ Анализ завершен!")

if __name__ == "__main__":
    main()
