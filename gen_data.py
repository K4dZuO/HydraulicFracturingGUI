import numpy as np
import pandas as pd
import random

def generate_samples(n_samples=1, n_points_per_sample=1000, output_file='samples.csv'):
    data = []
    
    for i in range(n_samples):
        print(f"Генерация {i}-го сэмпла")
        # Фиксированные параметры для одного сэмпла (выбираются случайно из возможных диапазонов)
        skin = np.random.choice([0.05, 0.8])
        h = np.random.choice([10, 30])
        N = np.random.choice([10, 55])
        W = np.random.randint(1100, 1600)
        L = np.random.randint(240, 381)
        aL = round(N / L, 20)
        
        # Генерация 1000 точек
        elem_indices = np.random.choice(np.arange(1, n_points_per_sample+1), size=n_points_per_sample, replace=False)
        
        for elem_idx in elem_indices:
            X = random.uniform(0, 0.5)
            Y = random.uniform(0, 21)
            t = random.uniform(0, 50000)
            P = random.uniform(90, 300)
            dP = random.uniform(0, 210)
            Q = random.uniform(100, 300)  # Как в примерах
            
            row = [skin, h, N, W, L, aL, elem_idx, X, Y, t, P, dP, Q]
            data.append(row)
    
    # Сохранение в CSV
    columns = ['Skin','h','N','W','L','a/L','ElemIdx','X','Y','t','P','dP','Q']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"Сгенерировано {len(data)} строк, сохранено в {output_file}")

# Пример использования: 
n_samples=30000
n_points_per_sample=500
generate_samples(n_samples=n_samples, 
                 n_points_per_sample=n_points_per_sample, 
                 output_file=f'generated_data_{n_samples}_samples.csv')
