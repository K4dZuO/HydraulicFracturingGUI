import numpy as np

from helpers.dimensionless_analysis import fit_xy_curve_coefficients


def test_fit_xy_curve_coefficients_recovers_x_scale_factor():
    """Коэффициент a должен восстанавливать масштаб X_data относительно X_calc."""
    x_calc = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    y_calc = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)

    x_scale = 3.0
    x_data = x_calc * x_scale
    y_data = y_calc.copy()

    result = fit_xy_curve_coefficients(
        X_data=x_data,
        Y_data=y_data,
        X_calc=x_calc,
        Y_calc=y_calc,
    )

    assert np.isfinite(result["a"])
    assert abs(result["a"] - x_scale) < 1e-2
