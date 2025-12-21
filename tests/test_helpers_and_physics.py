import io
import os
import math
import tempfile
import numpy as np
import pandas as pd
import pytest

from helpers.physics import (
    compute_transmissivity,
    compute_pore_volume,
    compute_darcy_flux,
    compute_diffusivity,
)
from helpers.timeseries import (
    compute_derivative,
    interpolate_series,
    smooth_series,
)
from helpers.parse_csv_params import parse_csv_params


def write_csv(tmpdir, header, rows):
    path = os.path.join(tmpdir, "test.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    return path


def test_compute_transmissivity_scalar_vector():
    k = np.array([10.0, 20.0, 30.0])
    h = np.array([1.0, 2.0, 3.0])
    t = compute_transmissivity(k, h)
    assert np.allclose(t, np.array([10.0, 40.0, 90.0]))


def test_compute_pore_volume_basic():
    phi = np.array([0.1, 0.2, 0.3])
    h = np.array([10.0, 10.0, 10.0])
    vp = compute_pore_volume(phi, h)
    assert np.allclose(vp, np.array([1.0, 2.0, 3.0]))


def test_compute_darcy_flux_sign_and_scale():
    k = np.array([1e-14, 2e-14])  # м^2 (~10 мД)
    mu = np.array([1e-3, 1e-3])   # Па·с (вода)
    grad = np.array([100.0, -50.0])  # Па/м
    q = compute_darcy_flux(k, mu, grad)
    assert q[0] < 0  # положительный градиент -> отрицательный расход
    assert q[1] > 0


def test_compute_diffusivity_monotonic():
    k = np.array([1e-14, 2e-14, 4e-14])
    c_t = np.array([1e-9, 1e-9, 1e-9])
    mu = np.array([1e-3, 1e-3, 1e-3])
    phi = np.array([0.2, 0.2, 0.2])
    alpha = compute_diffusivity(k, c_t, mu, phi)
    assert alpha[2] > alpha[1] > alpha[0]


def test_derivative_linear_should_be_constant():
    x = pd.Series(np.linspace(0, 10, 101), name="x")
    y = pd.Series(3.0 * x + 2.0, name="y")
    dy_dx = compute_derivative(y, x)
    assert np.allclose(dy_dx[1:-1], 3.0, atol=1e-12)


def test_derivative_datetime_index():
    t0 = pd.Timestamp("2024-01-01 00:00:00")
    times = pd.Series(pd.date_range(t0, periods=10, freq="S"), name="time")
    y = pd.Series(np.arange(10), name="y")
    dy_dt = compute_derivative(y, times)
    assert np.allclose(dy_dt[1:-1], 1.0, atol=1e-12)


def test_interpolate_linear_gaps():
    y = pd.Series([0.0, np.nan, np.nan, 3.0, 4.0], name="y")
    y_int = interpolate_series(y, method="linear")
    assert not y_int.isna().any()
    assert math.isclose(y_int.iloc[1], 1.0, rel_tol=1e-6)
    assert math.isclose(y_int.iloc[2], 2.0, rel_tol=1e-6)


def test_interpolate_time_index():
    t = pd.date_range("2025-01-01", periods=5, freq="T")
    y = pd.Series([0.0, np.nan, np.nan, 3.0, 4.0], index=t, name="y")
    y_int = interpolate_series(y, method="time")
    assert not y_int.isna().any()


def test_smooth_series_window_1_no_change():
    y = pd.Series([1, 2, 3, 4, 5], name="y")
    ys = smooth_series(y, window=1)
    assert np.allclose(ys, y)


def test_smooth_series_reduces_noise():
    rng = np.random.default_rng(42)
    y = pd.Series(np.sin(np.linspace(0, 2*np.pi, 200)) + 0.5 * rng.standard_normal(200), name="y")
    ys = smooth_series(y, window=9)
    # Дисперсия должна уменьшиться
    assert ys.var() < y.var()


def test_parse_csv_params_ok(tmp_path):
    header = ["k", "h", "phi"]
    rows = [["k", "h", "phi"], [10.0, 5.0, 0.2]]  # Доп. строка заголовков в файле не требуется, но вторую строку используем для значений
    # Корректный CSV: одна строка заголовка + одна строка значений
    path = os.path.join(tmp_path, "ok.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        f.write("10.0,5.0,0.2\n")

    rp, df, err = parse_csv_params(path)
    assert err is None
    assert rp is not None
    assert math.isclose(rp.k, 10.0)
    assert math.isclose(rp.h, 5.0)
    assert math.isclose(rp.phi, 0.2)


def test_parse_csv_params_missing_fields(tmp_path):
    header = ["k", "h"]
    path = write_csv(str(tmp_path), header, [[10.0, 5.0]])
    rp, df, err = parse_csv_params(path)
    assert rp is None and df is None
    assert "Отсутствуют обязательные" in err


def test_parse_csv_params_extra_fields(tmp_path):
    header = ["k", "h", "phi", "junk"]
    path = write_csv(str(tmp_path), header, [[10.0, 5.0, 0.2, 1]])
    rp, df, err = parse_csv_params(path)
    assert rp is None and df is None
    assert "Присутствуют посторонние" in err


def test_parse_csv_params_not_enough_rows(tmp_path):
    path = write_csv(str(tmp_path), ["k", "h", "phi"], [])
    rp, df, err = parse_csv_params(path)
    assert rp is None and df is None
    assert "Недостаточно строк" in err


def test_parse_csv_params_type_error(tmp_path):
    # phi выходит за пределы [0,1]
    path = write_csv(str(tmp_path), ["k", "h", "phi"], [[10.0, 5.0, 1.5]])
    rp, df, err = parse_csv_params(path)
    assert rp is None and df is None
    assert "Ошибка валидации" in err


def test_derivative_constant_zero():
    x = pd.Series(np.linspace(0, 10, 50), name="x")
    y = pd.Series(np.full(50, 7.0), name="y")
    dy_dx = compute_derivative(y, x)
    assert np.allclose(dy_dx, 0.0, atol=1e-12)


def test_derivative_non_uniform_grid():
    x = pd.Series(np.sort(np.random.default_rng(0).random(100) * 10), name="x")
    y = pd.Series(np.sin(x), name="y")
    dy_dx = compute_derivative(y, x)
    # Проверим, что нет NaN и конечные значения
    assert np.isfinite(dy_dx).all()


def test_interpolate_linear_with_edge_nans():
    y = pd.Series([np.nan, 1.0, np.nan, 3.0, np.nan], name="y")
    y_int = interpolate_series(y, method="linear")
    # Краевые NaN останутся (поведение pandas), центральные — интерполируются
    assert math.isclose(y_int.iloc[2], 2.0, rel_tol=1e-6)


def test_smooth_series_window_large_equals_mean():
    y = pd.Series([1.0, 3.0, 5.0], name="y")
    ys = smooth_series(y, window=99)
    assert np.allclose(ys, np.full_like(y, y.mean()))


def test_compute_transmissivity_with_zeros():
    k = np.array([0.0, 10.0])
    h = np.array([5.0, 0.0])
    t = compute_transmissivity(k, h)
    assert np.allclose(t, np.array([0.0, 0.0]))


def test_compute_diffusivity_sensitivity_phi():
    k = np.array([1e-14])
    c_t = np.array([1e-9])
    mu = np.array([1e-3])
    phi_small = np.array([0.1])
    phi_large = np.array([0.3])
    a_small = compute_diffusivity(k, c_t, mu, phi_small)
    a_large = compute_diffusivity(k, c_t, mu, phi_large)
    assert a_small > a_large


