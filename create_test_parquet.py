import pandas as pd
import numpy as np

# Создаем тестовые данные
data = {
    'Skin': [0.05] * 21,
    'h': [10.0] * 21,
    'N': [10] * 21,
    'W': [1125.0] * 21,
    'L': [280.0] * 21,
    'a/L': [0.446429] * 21,
    'ElemIdx': list(range(1, 22)),
    'X': [0.0] * 21,
    'Y': [0.0] * 21,
    't': list(range(21)),
    'P': [300 - i * 2.5 for i in range(21)],
    'dP': [0] + [-2.5] * 20,
    'Q': [100 - i * 1.0 for i in range(21)]
}

df = pd.DataFrame(data)

# Сохраняем как Parquet
df.to_parquet('well_data_test.parquet', index=False)

print("Тестовый Parquet файл создан: well_data_test.parquet")
print(f"Размер файла: {df.shape}")
print(f"Колонки: {list(df.columns)}")
