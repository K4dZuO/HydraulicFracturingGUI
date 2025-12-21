import numpy as np
import pandas as pd

def diag_dimensional(df):
    # основные колонки — проверяем наличие и типы
    for c in ['X','Y','P','Q','t', "dP"]:
        if c not in df.columns:
            print(f"⚠️ Нет колонки {c}")
    X = df['X'].astype(float).values
    Y = df['Y'].astype(float).values
    P = df['P'].astype(float).values
    Q = df['Q'].astype(float).ffill().bfill().values
    t = df['t'].astype(float).values
    # dP = df['dP'].astype(float).values

    def rng(a):
        mn = np.nanmin(a); mx = np.nanmax(a)
        return mn, mx, mx-mn, np.nanmax(a)/ (np.nanmin(a) if np.nanmin(a)>0 else 1)
    print("RANGES (min,max,span,ratio):")
    print(" X:", rng(X))
    print(" Y:", rng(Y))
    print(" P:", rng(P))
    print(" Q:", rng(Q))
    print(" t:", rng(t))
    # print(" dP:", rng(dP))

    # orders of magnitude (log10 span)
    def log_span(a):
        a = np.clip(a, 1e-30, None)
        return np.log10(np.nanmax(a)) - np.log10(np.nanmin(a))
    print("Log10 spans:") # кол-во порядков между min и max
    print(" X span (dex):", log_span(X))
    print(" Y span (dex):", log_span(Y))
    print(" pD span (dex):", log_span(np.clip((P - np.nanmin(P)),1e-30,None)))
    
    # delta p
    delta_p = P[0] - P[-1] if len(P)>1 else P[0]
    print("ΔP (P0 - Pend) = ", delta_p)
    
    # correlations
    for a,b,name in [(X,Y,'corr X-Y'), (np.log10(np.clip(Y,1e-30,None)), np.log10(np.clip(X,1e-30,None)),'corr logY-logX'),
                     (np.log10(np.clip(Y,1e-30,None)), P,'corr logY-P')]:
        valid = np.isfinite(a) & np.isfinite(b)
        if np.sum(valid) > 2:
            print(name, np.corrcoef(a[valid], b[valid])[0,1])
        else:
            print(name, "недостаточно данных")

    # check if X or Y almost constant
    if np.allclose(np.nanmin(X), np.nanmax(X)) or np.allclose(np.nanmin(Y), np.nanmax(Y)):
        print("⚠️ X или Y почти константа — это ключевая причина прямых линий в лог-лог.")
    if log_span(Y) < 0.5:
        print("⚠️ Диапазон Y < 0.5 dex — малый лог-диапазон, визуально будет почти прямая.")
    if log_span(X) < 0.5:
        print("⚠️ Диапазон X < 0.5 dex — малый лог-диапазон.")
        
    # delta_p = P0 - P_end ? или P0 - P_min? Проверить оба
    P0 = P[0]; Pend = P[-1]; Pmin = np.nanmin(P)
    print("P0, Pend, Pmin:", P0, Pend, Pmin)

    # Частая ошибка: delta_p близок к 0 -> pD бесформен
    delta_p1 = P0 - Pend
    delta_p2 = P0 - Pmin
    print("ΔP (P0-Pend) = ", delta_p1)
    print("ΔP (P0-Pmin) = ", delta_p2)
    
    
    corr = np.corrcoef(np.log10(np.clip(X,1e-30,None)), np.log10(np.clip(Y,1e-30,None)))[0,1]
    print("Корреляция logX-logY:", corr)
    
    # Вычисляем pD (безразмерное давление)
    # Используем более безопасный delta_p (берем максимальный перепад)
    delta_p = max(abs(delta_p1), abs(delta_p2))
    if delta_p < 1e-10:
        delta_p = 1.0  # Защита от деления на 0
    pD = (P - Pmin) / delta_p
    
    def compute_dpdlogY(Y, pD, smooth_window=5):
        # сортируем по Y
        idx = np.argsort(Y)
        Ys = Y[idx]; pDs = pD[idx]
        # защита от нулей
        Ys = np.clip(Ys, 1e-30, None)
        pDs = np.clip(pDs, 1e-30, None)
        # лог
        logY = np.log10(Ys)
        # вычисление производной dp/dlogY
        dpdlogY = np.gradient(np.log10(pDs), logY)  # или gradient(pDs, logY) в зависимости от определения
        # опциональное сглаживание
        try:
            from scipy.signal import savgol_filter
            if len(dpdlogY) >= 7:
                dpdlogY = savgol_filter(dpdlogY, 7, 2)
        except Exception:
            pass
        CD = Ys * dpdlogY
        return Ys, pDs, dpdlogY, CD

    Ys, pDs, dpdlogY, CD = compute_dpdlogY(Y, pD)
    print("dpdlogY min/max:", np.nanmin(dpdlogY), np.nanmax(dpdlogY))
    print("CD min/max:", np.nanmin(CD), np.nanmax(CD))
