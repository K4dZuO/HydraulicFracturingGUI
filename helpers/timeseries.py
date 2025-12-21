import numpy as np
import pandas as pd


def compute_derivative(y: pd.Series, x: pd.Series) -> pd.Series:
    """
    Численная производная dy/dx по центральным разностям.
    Корректно работает как с числовым x, так и с datetime64 (используется разность в секундах).
    """
    if len(y) != len(x):
        raise ValueError("Длины x и y должны совпадать")

    if np.issubdtype(x.dtype, np.datetime64):
        x_numeric = x.view("int64") / 1e9  # наносекунды -> секунды
    else:
        x_numeric = x.astype(float).to_numpy()

    y_numeric = y.astype(float).to_numpy()

    dy = np.gradient(y_numeric)
    dx = np.gradient(x_numeric)
    derivative = dy / dx
    return pd.Series(derivative, index=y.index, name=f"d{y.name}/d{x.name}")


def interpolate_series(y: pd.Series, method: str = "time") -> pd.Series:
    """
    Интерполяция пропусков в Series. Метод по умолчанию: 'time' (если индекс — datetime).
    Падение обратно на 'linear', если 'time' не поддерживается.
    """
    try:
        if method == "time" and not isinstance(y.index, pd.DatetimeIndex):
            # Если индекс не время — используем линейную
            return y.interpolate(method="linear")
        return y.interpolate(method=method)
    except Exception:
        return y.interpolate(method="linear")


def smooth_series(y: pd.Series, window: int = 5) -> pd.Series:
    """
    Простое сглаживание скользящим средним.
    """
    if window < 1:
        return y
    return y.rolling(window=window, min_periods=1, center=True).mean()


