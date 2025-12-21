import pandas as pd
from typing import Optional, Tuple

from schemas.required_params import RequiredParams


def parse_csv_params(file_path: str) -> Tuple[Optional[RequiredParams], Optional[pd.DataFrame], Optional[str]]:
    """
    Читает CSV, валидирует обязательные поля по схеме RequiredParams и извлекает значения
    первой строки данных (индекс 0) как набор параметров. Возвращает кортеж
    (required_params, dataframe, error_msg). В случае ошибки первые два элемента равны None.
    """
    if not file_path:
        return None, None, "Файл не выбран"

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return None, None, f"Ошибка чтения файла: {exc}"

    if df.shape[0] < 1:
        return None, None, "Недостаточно строк в CSV (нужны заголовки и хотя бы одна строка данных)"

    required_fields = set(RequiredParams.model_fields.keys())
    available_fields = set(df.columns)

    missing_fields = required_fields - available_fields
    extra_fields = available_fields - required_fields

    if missing_fields:
        error_msg = "Отсутствуют обязательные параметры:\n" + ", ".join(sorted(missing_fields))
        return None, None, error_msg
    if extra_fields:
        error_msg = "Присутствуют посторонние параметры:\n" + ", ".join(sorted(extra_fields))
        return None, None, error_msg

    # Извлекаем значения первой строки данных и приводим к float
    try:
        first_row = df.iloc[0]
        required_params_dict = {name: float(first_row[name]) for name in required_fields}
    except Exception as exc:
        return None, None, f"Ошибка извлечения параметров из CSV: {exc}"

    # Валидация по Pydantic-схеме
    try:
        required_params = RequiredParams(**required_params_dict)
    except Exception as exc:
        return None, None, f"Ошибка валидации параметров: {exc}"

    return required_params, df, None

