import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.interpolate import RBFInterpolator
from sklearn.metrics import mean_squared_error

from helpers.math_error_logger import log_math_error, log_metric_error, log_computation_error


class DimensionlessCurveInterpolator:
    """
    Интерполятор безразмерных кривых ГРП с адаптивным выбором метода.
    Интерполирует зависимость P_D(Y) по параметрам Skin, N, a/L.
    """

    def __init__(self, methods=("linear", "rbf", "gp"), constraints=None):
        """
        :param methods: список методов, которые будут тестироваться
        :param constraints: словарь физических ограничений
        """
        self.methods = methods
        self.best_method = None
        self.models = []
        self.Y_grid = None
        self.param_grid = None
        self.is_fitted = False
        self.rmse_scores = {}

        self.constraints = constraints or {
            "monotonic": True,
            "positive": True,
            "smooth": True,
            "clip_range": (0, 5.0),
            "enforce_direction": True,
            "local_suppression": True,
            "max_local_ratio": 1.5,
        }

    def fit(self, param_grid: np.ndarray, Y_grid: np.ndarray, P_curves: np.ndarray):
        """Обучение интерполяции и автоматический выбор метода."""
        self.param_grid = np.asarray(param_grid)
        self.Y_grid = np.asarray(Y_grid)
        P_curves = np.asarray(P_curves)
        
        # Сохраняем исходные данные для точного предсказания на обучающей выборке
        self.training_params = None
        self.training_curves = None
        
        # Нормализуем param_grid к 2D форме
        if self.param_grid.ndim == 1:
            self.param_grid = self.param_grid.reshape(-1, 1)
        
        n_samples = self.param_grid.shape[0]
        
        # Определяем правильную форму P_curves: ожидаем (n_samples, n_points)
        if P_curves.ndim == 1:
            # Одномерный массив - это одна кривая, нужно развернуть
            P_curves = P_curves.reshape(1, -1)
            n_samples_check, n_points = P_curves.shape
        elif P_curves.shape[0] == n_samples:
            # Форма (n_samples, n_points) - правильная
            n_samples_check, n_points = P_curves.shape
        elif P_curves.shape[1] == n_samples:
            # Форма (n_points, n_samples) - нужно транспонировать
            P_curves = P_curves.T
            n_samples_check, n_points = P_curves.shape
        else:
            # Пытаемся определить по Y_grid
            if P_curves.shape[1] == len(self.Y_grid):
                n_samples_check, n_points = P_curves.shape
            else:
                raise ValueError(f"Не могу определить размерность P_curves. "
                               f"Форма: {P_curves.shape}, param_grid.shape[0]: {n_samples}, "
                               f"Y_grid.shape[0]: {len(self.Y_grid)}")

        # Санити-проверки входа
        self._validate_input_ranges(self.param_grid, P_curves)

        # ОБРАБОТКА ПРОПУСКОВ (NaN): для каждой точки обучаемся на доступных данных
        # Сохраняем маску пропусков для каждой точки (маска по оси n_samples)
        # P_curves теперь точно имеет форму (n_samples, n_points)
        self.nan_masks = [~np.isnan(P_curves[:, i]) for i in range(n_points)]
        self.n_points_original = n_points

        # Проверяем, что есть хотя бы одна точка с данными
        if not any(any(mask) for mask in self.nan_masks):
            raise ValueError("Во всех точках отсутствуют данные - невозможно обучить модель")

        # Обработка кривых с учетом пропусков
        # P_curves уже имеет форму (n_samples, n_points), обрабатываем каждую точку
        P_curves_proc = np.copy(P_curves).astype(float)
        
        # Применяем ограничения к каждой кривой (по строкам, т.е. для каждой скважины)
        # Для каждой скважины применяем ограничения по ее кривой вдоль Y_grid
        for sample_idx in range(n_samples):
            curve = P_curves_proc[sample_idx, :]  # Кривая одной скважины вдоль Y_grid
            # Оставляем NaN как есть
            valid_mask = ~np.isnan(curve)
            if not np.any(valid_mask):
                continue
            
            curve_valid = curve[valid_mask]
            Y_valid = self.Y_grid[valid_mask]
            
            # Применяем ограничения только к валидным значениям
            if self.constraints.get("enforce_direction", True):
                curve_valid = self._enforce_monotonic_direction(Y_valid, curve_valid)
            
            if self.constraints.get("local_suppression", True):
                curve_valid = self._enforce_local_smoothness(curve_valid, max_ratio=self.constraints.get("max_local_ratio", 1.5))
            
            # Записываем обратно
            curve[valid_mask] = curve_valid
            P_curves_proc[sample_idx, :] = np.maximum(curve, 0.0, where=valid_mask)

        best_rmse = np.inf
        best_method = None
        best_models = None
        
        # Также сохраняем лучший результат без учета стабильности (fallback)
        fallback_rmse = np.inf
        fallback_method = None
        fallback_models = None

        # Тестируем все методы с использованием скользящего окна для контроля адекватности
        window_size = max(3, min(10, len(self.Y_grid) // 4))  # Размер окна для скользящей проверки
        stability_failures = {}  # Для отладки: сохраняем причины неудач

        for method in self.methods:
            models = []
            preds_all = []

            for i in range(n_points):
                y_values = P_curves_proc[:, i]
                valid_mask = self.nan_masks[i]  # Доступные значения для этой точки

                if not np.any(valid_mask):
                    # Все значения пропущены - пропускаем эту точку
                    models.append(None)
                    preds_all.append(np.full(len(self.param_grid), np.nan))
                    continue

                # Используем только доступные значения для обучения
                param_grid_valid = self.param_grid[valid_mask]
                y_values_valid = y_values[valid_mask]

                # Обучаем модель на доступных данных
                if method == "linear":
                    if len(param_grid_valid) == 1:
                        # Специальный случай: один пример - просто копируем значение
                        model = None  # Не обучаем модель
                        preds = np.full(len(self.param_grid), y_values_valid[0])
                    else:
                        model = LinearRegression().fit(param_grid_valid, y_values_valid)
                        preds = model.predict(self.param_grid)
                elif method == "rbf":
                    # RBFInterpolator требует минимум (degree + 1) * dimensions точек для thin_plate_spline
                    # degree по умолчанию = 1, dimensions = param_grid_valid.shape[1]
                    # Минимум: 2 * dimensions точек для надежности
                    if param_grid_valid.ndim == 1:
                        param_grid_2d = param_grid_valid.reshape(-1, 1)
                        param_grid_full_2d = self.param_grid.reshape(-1, 1) if self.param_grid.ndim == 1 else self.param_grid
                    else:
                        param_grid_2d = param_grid_valid
                        param_grid_full_2d = self.param_grid
                    
                    n_dims = param_grid_2d.shape[1]
                    min_points_required = max(4, 2 * n_dims)  # Минимум 4 или 2*dims
                    
                    if len(param_grid_valid) < min_points_required:
                        # Недостаточно данных - используем линейную регрессию
                        model = LinearRegression().fit(param_grid_valid if param_grid_valid.ndim > 1 else param_grid_valid.reshape(-1, 1), y_values_valid)
                        preds = model.predict(self.param_grid if self.param_grid.ndim > 1 else self.param_grid.reshape(-1, 1))
                    else:
                        try:
                            model = RBFInterpolator(param_grid_2d, y_values_valid, kernel='thin_plate_spline')
                            preds = model(param_grid_full_2d)
                        except (np.linalg.LinAlgError, ValueError) as e:
                            # Singular matrix или другая ошибка - fallback на линейную регрессию
                            log_computation_error(
                                subsystem="interpolation",
                                method="rbf",
                                exception=e,
                                data_volume=len(param_grid_valid),
                                data_quality=1.0 - (np.sum(np.isnan(y_values_valid)) / len(y_values_valid)) if len(y_values_valid) > 0 else 0.0,
                                context={"point_idx": i, "n_samples": len(param_grid_valid)}
                            )
                            model = LinearRegression().fit(param_grid_valid if param_grid_valid.ndim > 1 else param_grid_valid.reshape(-1, 1), y_values_valid)
                            preds = model.predict(self.param_grid if self.param_grid.ndim > 1 else self.param_grid.reshape(-1, 1))
                elif method == "gp":
                    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
                    model = GaussianProcessRegressor(kernel=kernel).fit(param_grid_valid, y_values_valid)
                    preds = model.predict(self.param_grid)
                else:
                    raise ValueError(f"Неизвестный метод: {method}")

                models.append(model)
                # Убеждаемся, что preds - одномерный массив размера n_samples
                preds = np.asarray(preds).flatten()
                if len(preds) != len(self.param_grid):
                    raise ValueError(f"Размер предсказаний {len(preds)} не совпадает с размером param_grid {len(self.param_grid)}")
                preds_all.append(preds)

            preds_all = np.array(preds_all).T  # Форма: (n_samples, n_points)
            
            # Убеждаемся, что формы совпадают
            if preds_all.shape != P_curves_proc.shape:
                raise ValueError(f"Формы не совпадают: preds_all {preds_all.shape} vs P_curves_proc {P_curves_proc.shape}")

            # Простой RMSE - только по доступным точкам
            rmse = self._simple_rmse_with_nans(P_curves_proc, preds_all, self.nan_masks)
            self.rmse_scores[method] = rmse
            
            # Сохраняем fallback вариант (лучший по RMSE без проверки стабильности)
            if rmse < fallback_rmse:
                fallback_rmse = rmse
                fallback_method = method
                fallback_models = models

            # Контроль нестабильности по стандартному отклонению
            valid_preds = preds_all[~np.isnan(preds_all)]
            valid_true = P_curves_proc[~np.isnan(P_curves_proc)]
            if len(valid_preds) > 0 and np.std(valid_preds) > 10.0 * np.std(valid_true):
                # Исключаем метод как нестабильный, но fallback уже сохранен
                continue

            # Проверка стабильности с использованием скользящего окна для каждой кривой
            # Делаем проверку более мягкой: проверяем стабильность на большей части окон, а не на всех
            is_stable = True
            stability_checks = 0
            stability_passes = 0
            
            for curve_idx in range(preds_all.shape[0]):  # Для каждой кривой в обучающем наборе
                curve_pred = preds_all[curve_idx, :]
                valid_points = ~np.isnan(curve_pred)

                if not np.any(valid_points):
                    continue

                # Проверяем стабильность с помощью скользящего окна
                # Используем шаг для уменьшения количества проверок
                step = max(1, window_size // 2)  # Проверяем каждое второе окно
                for start_idx in range(0, len(curve_pred) - window_size + 1, step):
                    end_idx = start_idx + window_size
                    window_pred = curve_pred[start_idx:end_idx]
                    window_Y = self.Y_grid[start_idx:end_idx]
                    window_valid = valid_points[start_idx:end_idx]

                    if not np.any(window_valid) or np.sum(window_valid) < 3:
                        continue

                    stability = self.check_stability(window_Y[window_valid], window_pred[window_valid])
                    stability_checks += 1
                    
                    if stability["stable"]:
                        stability_passes += 1
                    else:
                        # Сохраняем информацию о неудаче для отладки
                        if method not in stability_failures:
                            stability_failures[method] = []
                        stability_failures[method].append({
                            "max_jump": stability["max_jump"],
                            "max_second_derivative": stability["max_second_derivative"]
                        })

            # Метод считается стабильным, если прошло >= 70% проверок
            stability_ratio = stability_passes / stability_checks if stability_checks > 0 else 0.0
            is_stable = stability_ratio >= 0.7
            
            if not is_stable and stability_checks > 0:
                # Логируем информацию для отладки
                avg_max_jump = np.mean([f["max_jump"] for failures in stability_failures.get(method, []) for f in failures]) if method in stability_failures else 0
                avg_max_dd = np.mean([f["max_second_derivative"] for failures in stability_failures.get(method, []) for f in failures]) if method in stability_failures else 0
                print(f"Метод '{method}': стабильность {stability_passes}/{stability_checks} ({stability_ratio:.1%}), "
                      f"средний max_jump={avg_max_jump:.2f}, средний max_dd={avg_max_dd:.2f}")

            # Если метод нестабилен, исключаем его
            if not is_stable:
                continue

            if rmse < best_rmse:
                best_rmse = rmse
                best_method = method
                best_models = models

        # Сохраняем лучший результат
            # Если ни один метод не прошел проверку стабильности, используем fallback
        if best_models is None:
            if fallback_models is not None:
                self.best_method = fallback_method
                self.models = fallback_models
                print(f"Предупреждение: ни один метод не прошел проверку стабильности. "
                      f"Используется fallback метод '{fallback_method}' с RMSE={fallback_rmse:.4f}")
                
                # Логируем предупреждение о fallback
                log_math_error(
                    subsystem="interpolation",
                    method=fallback_method,
                    error_type="stability_warning",
                    error_value=fallback_rmse,
                    error_message="Использован fallback метод из-за нестабильности",
                    data_volume=n_samples,
                    data_quality=1.0 - (np.sum(np.isnan(P_curves_proc)) / P_curves_proc.size) if P_curves_proc.size > 0 else 0.0,
                    metadata={"fallback_rmse": fallback_rmse, "n_points": n_points}
                )
            else:
                error_msg = "Не удалось обучить ни один метод интерполяции"
                log_computation_error(
                    subsystem="interpolation",
                    method="all",
                    exception=RuntimeError(error_msg),
                    data_volume=n_samples,
                    data_quality=1.0 - (np.sum(np.isnan(P_curves_proc)) / P_curves_proc.size) if P_curves_proc.size > 0 else 0.0,
                    context={"n_points": n_points, "n_samples": n_samples}
                )
                raise RuntimeError(error_msg)
        else:
            self.best_method = best_method
            self.models = best_models
        
        # Сохраняем обучающие данные для точного предсказания
        self.training_params = self.param_grid.copy()
        self.training_curves = P_curves_proc.copy()
        
        self.is_fitted = True
        return self
    
    def get_interpolation_info(self) -> dict:
        """
        Возвращает информацию о результатах интерполяции.
        
        Returns:
            dict: Словарь с информацией о выбранном методе и RMSE всех методов
        """
        if not self.is_fitted:
            return {
                "fitted": False,
                "best_method": None,
                "rmse_scores": {},
                "message": "Модель не обучена"
            }
        
        return {
            "fitted": True,
            "best_method": self.best_method,
            "rmse_scores": self.rmse_scores,
            "n_samples": self.param_grid.shape[0] if hasattr(self, 'param_grid') else 0,
            "n_points": len(self.Y_grid) if hasattr(self, 'Y_grid') else 0,
            "message": f"Выбран метод '{self.best_method}' с RMSE={self.rmse_scores.get(self.best_method, 0):.4f}"
        }

    def fit_xy(self, param_grid: np.ndarray, Y_grid: np.ndarray, X_curves: np.ndarray, Y_curves: np.ndarray):
        """Обучение интерполяции на X-Y кривых."""
        # Обучаем отдельные модели для X и Y
        self.param_grid = np.asarray(param_grid)
        self.Y_grid = np.asarray(Y_grid)
        X_curves = np.asarray(X_curves)
        Y_curves = np.asarray(Y_curves)
        
        # Обучаем модель для X
        self.X_interpolator = DimensionlessCurveInterpolator(methods=self.methods, constraints=self.constraints)
        self.X_interpolator.fit(param_grid, Y_grid, X_curves)
        
        # Обучаем модель для Y (Y обычно совпадает с Y_grid, но для консистентности обучаем отдельно)
        self.Y_interpolator = DimensionlessCurveInterpolator(methods=self.methods, constraints=self.constraints)
        self.Y_interpolator.fit(param_grid, Y_grid, Y_curves)
        
        self.is_fitted = True
        self.best_method = self.X_interpolator.best_method  # Используем метод из X интерполятора
        self.rmse_scores = self.X_interpolator.rmse_scores  # Используем метрики из X интерполятора
        
    def predict(self, skin: float, N: float, a_L: float) -> pd.DataFrame:
        """Получение интерполированной X-Y кривой для заданных параметров."""
        if not self.is_fitted:
            raise RuntimeError("Сначала вызови fit() или fit_xy()")
        
        # Если обучены X-Y модели, используем их
        if hasattr(self, 'X_interpolator') and hasattr(self, 'Y_interpolator'):
            X_pred_series = self.X_interpolator.predict(skin, N, a_L)
            Y_pred_series = self.Y_interpolator.predict(skin, N, a_L)
            # X_pred и Y_pred - это Series, извлекаем значения
            if isinstance(X_pred_series, pd.Series):
                X_pred_values = X_pred_series.values
                Y_pred_values = Y_pred_series.values
                index = X_pred_series.index
            else:
                # Если это не Series, пытаемся преобразовать
                X_pred_values = np.asarray(X_pred_series)
                Y_pred_values = np.asarray(Y_pred_series)
                index = self.Y_grid
            return pd.DataFrame({'X': X_pred_values, 'Y': Y_pred_values}, index=index)
        
        # Иначе используем старый метод (для обратной совместимости)
        if not hasattr(self, 'models') or len(self.models) == 0:
            raise RuntimeError("Модель не обучена. Вызовите fit() или fit_xy()")

        # Старый метод для обратной совместимости (pD интерполяция)
        X_pred = np.array([[skin, N, a_L]])
        
        # Проверяем, есть ли точное совпадение в обучающей выборке
        if self.training_params is not None and self.training_curves is not None:
            for i, params in enumerate(self.training_params):
                if np.allclose(params, X_pred[0], rtol=1e-9, atol=1e-9):
                    # Точное совпадение - возвращаем обучающие данные без дополнительных ограничений
                    # (ограничения уже были применены при fit)
                    P_curve = self.training_curves[i, :]
                    return pd.Series(P_curve, index=self.Y_grid, name=f"P_D(s={skin}, N={N}, a/L={a_L})")
        
        # Нет точного совпадения - используем модели
        preds = []

        # Сначала получаем все доступные предсказания
        for model in self.models:
            if model is None:
                preds.append(np.nan)
            elif isinstance(model, RBFInterpolator):
                val = model(X_pred)
                preds.append(val[0])
            else:
                val = model.predict(X_pred)
                preds.append(val[0])

        # Теперь заполняем пропуски fallback интерполяцией
        for i in range(len(preds)):
            if np.isnan(preds[i]):
                preds[i] = self._fallback_interpolation(i, preds)

        P_curve = np.array(preds)
        P_curve = self._apply_physical_constraints_to_series(P_curve)

        return pd.Series(P_curve, index=self.Y_grid, name=f"P_D(s={skin}, N={N}, a/L={a_L})")

    def _fallback_interpolation(self, point_idx: int, current_preds: list) -> float:
        """
        Fallback интерполяция для точек, где нет обученной модели.
        Используется, когда для данной точки не было доступных данных при обучении.
        """
        # Находим ближайшие доступные точки
        available_indices = []
        available_values = []

        for i, pred in enumerate(current_preds):
            if not np.isnan(pred):
                available_indices.append(i)
                available_values.append(pred)

        if len(available_values) < 2:
            # Недостаточно точек для интерполяции - используем среднее или константу
            if len(available_values) == 1:
                return available_values[0]
            else:
                # Полностью отсутствуют данные - возвращаем разумное значение
                return 1.0  # Типичное значение для безразмерного давления

        # Линейная интерполяция по логарифму Y
        y_values = np.log10(self.Y_grid[available_indices])
        target_y = np.log10(self.Y_grid[point_idx])

        # Интерполируем
        from scipy.interpolate import interp1d
        try:
            interp_func = interp1d(y_values, available_values,
                                 kind='linear', bounds_error=False,
                                 fill_value=(available_values[0], available_values[-1]))
            return float(interp_func(target_y))
        except:
            # В случае ошибки интерполяции возвращаем среднее
            return float(np.mean(available_values))

    # ---------------------------
    # ФИЗИЧЕСКИЕ ОГРАНИЧЕНИЯ
    # ---------------------------

    def _apply_physical_constraints(self, P_curve: np.ndarray) -> np.ndarray:
        """Применяет базовые физические ограничения к кривой."""
        if self.constraints.get("positive", True):
            P_curve = np.maximum(P_curve, 0.0)

        if self.constraints.get("clip_range", None):
            low, high = self.constraints["clip_range"]
            P_curve = np.clip(P_curve, low, high)

        if self.constraints.get("monotonic", True):
            P_curve = np.maximum.accumulate(P_curve[::-1])[::-1]

        if self.constraints.get("smooth", True):
            P_curve = np.convolve(P_curve, np.ones(3)/3, mode='same')

        return P_curve

    def _apply_physical_constraints_to_series(self, P_curve: np.ndarray) -> np.ndarray:
        """Применяет базовые физические ограничения к одномерной кривой."""
        # Работаем только с не-NaN значениями
        valid_mask = ~np.isnan(P_curve)

        if not np.any(valid_mask):
            return P_curve

        if self.constraints.get("positive", True):
            P_curve = np.where(valid_mask, np.maximum(P_curve, 0.0), P_curve)

        if self.constraints.get("clip_range", None):
            low, high = self.constraints["clip_range"]
            P_curve = np.where(valid_mask, np.clip(P_curve, low, high), P_curve)

        if self.constraints.get("monotonic", True):
            # Применяем монотонность только к валидным значениям
            valid_values = P_curve[valid_mask]
            valid_values = np.maximum.accumulate(valid_values[::-1])[::-1]
            P_curve = np.where(valid_mask, valid_values, P_curve)

        if self.constraints.get("smooth", True):
            # Сглаживание только валидных значений
            valid_values = P_curve[valid_mask]
            if len(valid_values) >= 3:
                smoothed = np.convolve(valid_values, np.ones(3)/3, mode='same')
                P_curve = np.where(valid_mask, smoothed, P_curve)

        return P_curve

    # ---------------------------
    # 🔒 Дополнительные ограничения/проверки
    # ---------------------------

    def _validate_input_ranges(self, param_grid: np.ndarray, P_curves: np.ndarray):
        if param_grid.shape[1] >= 1 and (np.any(param_grid[:, 0] < -10) or np.any(param_grid[:, 0] > 50)):
            raise ValueError("Skin вне реалистичного диапазона (-10..50)")
        if param_grid.shape[1] >= 2 and (np.any(param_grid[:, 1] < 1) or np.any(param_grid[:, 1] > 100)):
            raise ValueError("N вне диапазона 1..100")
        if param_grid.shape[1] >= 3:
            # Для a/L: обрезаем значения до допустимого диапазона вместо выброса ошибки
            invalid_a_l = (param_grid[:, 2] < 0) | (param_grid[:, 2] > 1)
            if np.any(invalid_a_l):
                n_invalid = np.sum(invalid_a_l)
                print(f"Предупреждение: {n_invalid} кривых с a/L вне диапазона [0, 1]. "
                      f"Значения будут обрезаны до допустимого диапазона.")
                # Обрезаем значения до [0, 1]
                param_grid[:, 2] = np.clip(param_grid[:, 2], 0.0, 1.0)
        if np.any(P_curves < 0):
            # Обрезаем отрицательные значения
            P_curves[P_curves < 0] = 0.0

    def _enforce_monotonic_direction(self, Y: np.ndarray, values: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y).astype(float)
        v = np.asarray(values).astype(float)
        # Если Y убывает, переворачиваем и затем восстановим порядок
        flipped = False
        if len(Y) >= 2 and Y[1] < Y[0]:
            Y = Y[::-1]
            v = v[::-1]
            flipped = True
        # Давление/дебит в безразмерном виде типично не возрастает
        v = np.maximum.accumulate(v[::-1])[::-1]
        if flipped:
            v = v[::-1]
        return v

    def _enforce_local_smoothness(self, arr: np.ndarray, max_ratio: float = 1.5) -> np.ndarray:
        arr = np.asarray(arr).astype(float)
        for i in range(1, len(arr)):
            if arr[i-1] != 0:
                ratio = arr[i] / arr[i-1]
                if ratio > max_ratio:
                    arr[i] = arr[i-1] * max_ratio
                elif ratio < 1.0 / max_ratio:
                    arr[i] = arr[i-1] / max_ratio
        return arr

    def _simple_rmse(self, truth: np.ndarray, pred: np.ndarray) -> float:
        """Простой RMSE без взвешивания."""
        dif = (pred - truth)
        mse = np.nanmean(dif ** 2)
        return float(np.sqrt(mse))

    def _simple_rmse_with_nans(self, truth: np.ndarray, pred: np.ndarray, nan_masks: list) -> float:
        """RMSE только по доступным (не-NaN) точкам."""
        # Убеждаемся, что массивы имеют правильную форму
        truth = np.asarray(truth)
        pred = np.asarray(pred)
        
        if truth.shape != pred.shape:
            raise ValueError(f"Формы truth {truth.shape} и pred {pred.shape} не совпадают")
        
        if len(nan_masks) != truth.shape[1]:
            raise ValueError(f"Количество масок {len(nan_masks)} не совпадает с количеством точек {truth.shape[1]}")
        
        total_mse = 0.0
        total_points = 0

        for i in range(truth.shape[1]):  # Для каждой точки на кривой
            valid_mask = np.asarray(nan_masks[i])
            if not np.any(valid_mask):
                continue

            # Убеждаемся, что маска имеет правильный размер
            if len(valid_mask) != truth.shape[0]:
                raise ValueError(f"Размер маски {len(valid_mask)} для точки {i} не совпадает с размером truth.shape[0] {truth.shape[0]}")

            truth_valid = truth[valid_mask, i]
            pred_valid = pred[valid_mask, i]

            if len(truth_valid) == 0:
                continue

            dif = (pred_valid - truth_valid)
            mse = np.mean(dif ** 2)
            total_mse += mse * len(truth_valid)
            total_points += len(truth_valid)

        if total_points == 0:
            return np.inf

        return float(np.sqrt(total_mse / total_points))
    
    def check_stability(self, Y: np.ndarray, P: np.ndarray, 
                        max_ratio: float = 5.0, max_second_deriv: float = 10.0) -> dict:
        """
        Проверяет устойчивость интерполяции:
        - нет резких скачков;
        - нет осцилляций;
        - вторая производная не взрывается.
        """
        Y, P = np.asarray(Y), np.asarray(P)
        
        # Фильтруем NaN значения
        valid_mask = ~(np.isnan(Y) | np.isnan(P))
        Y = Y[valid_mask]
        P = P[valid_mask]
        
        if len(Y) < 3:
            # Недостаточно точек для проверки стабильности
            return {"stable": True, "max_jump": 0.0, "max_second_derivative": 0.0}
        
        dY = np.diff(Y)
        dP = np.diff(P)

        # Относительное изменение с более robust проверкой
        with np.errstate(divide='ignore', invalid='ignore'):
            # Игнорируем очень малые изменения (плато)
            abs_dP = np.abs(dP)
            threshold = 1e-6 * np.max(np.abs(P))  # Относительный порог
            significant_mask = abs_dP[:-1] > threshold
            
            if np.any(significant_mask):
                # Используем более устойчивую формулу для отношения изменений
                dP_prev = dP[:-1][significant_mask]
                dP_next = dP[1:][significant_mask]
                # Избегаем деления на очень маленькие числа
                denominator = np.abs(dP_prev) + 1e-6 * np.max(np.abs(P))
                ratio = np.abs(dP_next / denominator)
                max_jump = np.nanmax(ratio) if len(ratio) > 0 else 0.0
            else:
                max_jump = 0.0  # Все изменения незначительны
        
        # Вторая производная (в лог-пространстве)
        P_safe = np.clip(P, 1e-12, None)
        log_P = np.log10(P_safe)
        second_deriv = np.gradient(np.gradient(log_P))
        max_dd = np.nanmax(np.abs(second_deriv))
        
        stable = (max_jump < max_ratio) and (max_dd < max_second_deriv)
        return {
            "stable": stable,
            "max_jump": float(max_jump),
            "max_second_derivative": float(max_dd),
        }


    def compare_with_reference(self, Y_pred: np.ndarray, P_pred: np.ndarray,
                               Y_ref: np.ndarray, P_ref: np.ndarray, 
                               n_random_points: int = 100) -> dict:
        """Сравнение предсказанной кривой с эталоном по нескольким метрикам.

        Использует случайные точки эталона для более точной оценки качества.
        
        Args:
            Y_pred: Y координаты предсказания
            P_pred: pD значения предсказания
            Y_ref: Y координаты эталона
            P_ref: pD значения эталона
            n_random_points: Количество случайных точек для оценки (по умолчанию 100)
        
        Returns:
            Словарь с метриками: {'rmse', 'mae', 'mape', 'r2', 'max_error', 'median_error', 'mean_error'}
        """
        Yp = np.clip(np.asarray(Y_pred), 1e-30, None)
        Yr = np.clip(np.asarray(Y_ref), 1e-30, None)
        Pp = np.asarray(P_pred)
        Pr = np.asarray(P_ref)

        # Определяем диапазон для случайных точек
        Y_min = max(np.nanmin(Yp), np.nanmin(Yr))
        Y_max = min(np.nanmax(Yp), np.nanmax(Yr))
        
        if Y_min >= Y_max:
            # Если диапазоны не пересекаются, используем стандартный подход
            Pr_interp = np.interp(np.log10(Yp), np.log10(Yr), Pr)
            Y_eval = Yp
            P_ref_eval = Pr_interp
            P_pred_eval = Pp
        else:
            # Генерируем случайные точки в логарифмической шкале для равномерного распределения
            np.random.seed(42)  # Для воспроизводимости
            log_Y_random = np.random.uniform(np.log10(Y_min), np.log10(Y_max), n_random_points)
            Y_random = 10 ** log_Y_random
            
            # Интерполируем эталон на случайные точки
            from scipy.interpolate import interp1d
            # Сортируем эталон для интерполяции
            sort_idx_ref = np.argsort(np.log10(Yr))
            Yr_sorted = Yr[sort_idx_ref]
            Pr_sorted = Pr[sort_idx_ref]
            
            # Удаляем дубликаты для интерполяции
            unique_mask = np.concatenate(([True], np.diff(np.log10(Yr_sorted)) > 1e-10))
            Yr_unique = Yr_sorted[unique_mask]
            Pr_unique = Pr_sorted[unique_mask]
            
            if len(Yr_unique) > 1:
                interp_ref = interp1d(np.log10(Yr_unique), Pr_unique, 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                P_ref_random = interp_ref(log_Y_random)
            else:
                P_ref_random = np.full(n_random_points, Pr_unique[0] if len(Pr_unique) > 0 else 0)
            
            # Интерполируем предсказание на те же случайные точки
            sort_idx_pred = np.argsort(np.log10(Yp))
            Yp_sorted = Yp[sort_idx_pred]
            Pp_sorted = Pp[sort_idx_pred]
            
            unique_mask_pred = np.concatenate(([True], np.diff(np.log10(Yp_sorted)) > 1e-10))
            Yp_unique = Yp_sorted[unique_mask_pred]
            Pp_unique = Pp_sorted[unique_mask_pred]
            
            if len(Yp_unique) > 1:
                interp_pred = interp1d(np.log10(Yp_unique), Pp_unique,
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                P_pred_random = interp_pred(log_Y_random)
            else:
                P_pred_random = np.full(n_random_points, Pp_unique[0] if len(Pp_unique) > 0 else 0)
            
            # Фильтруем валидные точки
            valid_mask = np.isfinite(P_ref_random) & np.isfinite(P_pred_random)
            Y_eval = Y_random[valid_mask]
            P_ref_eval = P_ref_random[valid_mask]
            P_pred_eval = P_pred_random[valid_mask]
        
        # Вычисляем метрики
        diff = P_pred_eval - P_ref_eval
        
        # RMSE (Root Mean Square Error)
        rmse = float(np.sqrt(np.nanmean(diff ** 2)))
        
        # MAE (Mean Absolute Error)
        mae = float(np.nanmean(np.abs(diff)))
        
        # MAPE (Mean Absolute Percentage Error) в процентах
        denom = np.where(np.abs(P_ref_eval) < 1e-12, 1e-12, np.abs(P_ref_eval))
        mape = float(np.nanmean(np.abs(diff) / denom) * 100.0)
        
        # R² (Coefficient of Determination)
        ss_res = np.nansum((P_ref_eval - P_pred_eval) ** 2)
        ss_tot = np.nansum((P_ref_eval - np.nanmean(P_ref_eval)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        # Max Error
        max_error = float(np.nanmax(np.abs(diff)))
        
        # Median Error
        median_error = float(np.nanmedian(np.abs(diff)))
        
        # Mean Error (может быть отрицательным)
        mean_error = float(np.nanmean(diff))
        
        # Mean Squared Error (MSE)
        mse = float(np.nanmean(diff ** 2))

        metrics = {
            "rmse": rmse, 
            "mae": mae, 
            "mape": mape,
            "r2": r2,
            "max_error": max_error,
            "median_error": median_error,
            "mean_error": mean_error,
            "mse": mse
        }
        
        # Логируем метрики ошибок
        try:
            data_volume = len(Y_eval) if len(Y_eval) > 0 else len(Y_pred)
            # Оцениваем качество данных (чем меньше NaN/Inf, тем выше качество)
            quality = np.sum(np.isfinite(P_pred_eval)) / len(P_pred_eval) if len(P_pred_eval) > 0 else 0.0
            
            log_metric_error(
                subsystem="interpolation",
                method=self.best_method or "unknown",
                metrics=metrics,
                data_volume=data_volume,
                data_quality=quality,
                metadata={"n_points": len(Y_eval), "method": self.best_method}
            )
        except Exception as e:
            # Не прерываем выполнение при ошибке логирования
            pass
        
        return metrics
