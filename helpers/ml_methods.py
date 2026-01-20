import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union, Optional, Callable, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from scipy import signal
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.interpolate import RBFInterpolator as ScipyRBFInterpolator
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)  # Детерминированность для воспроизводимости


class MLInterpolator:
    """ML-интерполятор для временных рядов"""
    
    def __init__(self, method: str = 'random_forest'):
        self.method = method
        self.model = None
        self.is_fitted = False
        
    def fit(self, time: pd.Series, values: pd.Series) -> 'MLInterpolator':
        """Обучение модели на доступных данных"""
        # Очищаем данные от NaN
        valid_mask = ~(pd.isna(time) | pd.isna(values))
        time_clean = time[valid_mask]
        values_clean = values[valid_mask]
        
        if len(time_clean) < 3:
            raise ValueError("Недостаточно данных для обучения")
        
        # Подготавливаем признаки
        X = self._prepare_features(time_clean)
        y = values_clean.values
        
        # Выбираем модель
        if self.method == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.method == 'polynomial':
            self.model = Pipeline([
                ('poly', PolynomialFeatures(degree=3)),
                ('linear', LinearRegression())
            ])
        elif self.method == 'ridge':
            self.model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        # Обучаем модель
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def _prepare_features(self, time: pd.Series) -> np.ndarray:
        """Подготовка признаков для ML модели"""
        time_norm = (time - time.min()) / (time.max() - time.min())
        
        features = np.column_stack([
            time_norm.values,
            np.sin(2 * np.pi * time_norm),  # Периодические признаки
            np.cos(2 * np.pi * time_norm),
            time_norm.values ** 2,  # Полиномиальные признаки
            time_norm.values ** 3
        ])
        return features
    
    def predict(self, time: pd.Series) -> pd.Series:
        """Предсказание значений для новых временных точек"""
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        
        X = self._prepare_features(time)
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=time.index, name=f'ML_{self.method}_interpolated')


class MLFilter:
    """ML-фильтр для удаления шума из временных рядов"""
    
    def __init__(self, method: str = 'savitzky_golay'):
        self.method = method
        
    def filter(self, values: pd.Series, **kwargs) -> pd.Series:
        """Применение фильтра к данным"""
        if self.method == 'savitzky_golay':
            window_length = kwargs.get('window_length', 5)
            polyorder = kwargs.get('polyorder', 2)
            return self._savitzky_golay_filter(values, window_length, polyorder)
        
        elif self.method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            return self._gaussian_filter(values, sigma)
        
        elif self.method == 'median':
            kernel_size = kwargs.get('kernel_size', 5)
            return self._median_filter(values, kernel_size)
        
        elif self.method == 'kalman':
            return self._kalman_filter(values)
        
        else:
            raise ValueError(f"Неизвестный метод фильтрации: {self.method}")
    
    def _savitzky_golay_filter(self, values: pd.Series, window_length: int, polyorder: int) -> pd.Series:
        """Фильтр Савицкого-Голея"""
        try:
            filtered = signal.savgol_filter(values.values, window_length, polyorder)
            return pd.Series(filtered, index=values.index, name=f'filtered_{self.method}')
        except ValueError:
            # Если параметры некорректны, возвращаем исходные данные
            return values
    
    def _gaussian_filter(self, values: pd.Series, sigma: float) -> pd.Series:
        """Гауссовский фильтр"""
        from scipy.ndimage import gaussian_filter1d
        filtered = gaussian_filter1d(values.values, sigma=sigma)
        return pd.Series(filtered, index=values.index, name=f'filtered_{self.method}')
    
    def _median_filter(self, values: pd.Series, kernel_size: int) -> pd.Series:
        """Медианный фильтр"""
        from scipy.ndimage import median_filter
        filtered = median_filter(values.values, size=kernel_size)
        return pd.Series(filtered, index=values.index, name=f'filtered_{self.method}')
    
    def _kalman_filter(self, values: pd.Series) -> pd.Series:
        """Упрощенный фильтр Калмана"""
        # Простая реализация одномерного фильтра Калмана
        n = len(values)
        if n < 2:
            return values
        
        # Параметры фильтра
        Q = 0.1  # Процессный шум
        R = 0.1  # Шум измерений
        
        # Инициализация
        x = values.iloc[0]  # Состояние
        P = 1.0  # Ковариация
        
        filtered_values = [x]
        
        for i in range(1, n):
            # Предсказание
            x_pred = x
            P_pred = P + Q
            
            # Обновление
            K = P_pred / (P_pred + R)  # Коэффициент Калмана
            x = x_pred + K * (values.iloc[i] - x_pred)
            P = (1 - K) * P_pred
            
            filtered_values.append(x)
        
        return pd.Series(filtered_values, index=values.index, name=f'filtered_{self.method}')


def apply_ml_interpolation(time: pd.Series, values: pd.Series, 
                          method: str = 'random_forest') -> pd.Series:
    """Применение ML-интерполяции к данным"""
    interpolator = MLInterpolator(method=method)
    try:
        interpolator.fit(time, values)
        return interpolator.predict(time)
    except Exception as e:
        print(f"Ошибка ML-интерполяции: {e}")
        return values


def apply_ml_filter(values: pd.Series, method: str = 'savitzky_golay', **kwargs) -> pd.Series:
    """Применение ML-фильтрации к данным"""
    filter_obj = MLFilter(method=method)
    try:
        return filter_obj.filter(values, **kwargs)
    except Exception as e:
        print(f"Ошибка ML-фильтрации: {e}")
        return values


def detect_outliers(values: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """Обнаружение выбросов в данных"""
    if method == 'iqr':
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (values < lower_bound) | (values > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((values - values.mean()) / values.std())
        outliers = z_scores > threshold
    
    elif method == 'modified_zscore':
        median = values.median()
        mad = np.median(np.abs(values - median))
        modified_z_scores = 0.6745 * (values - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
    
    else:
        raise ValueError(f"Неизвестный метод обнаружения выбросов: {method}")
    
    # Убеждаемся, что возвращаем pandas Series
    if isinstance(outliers, np.ndarray):
        return pd.Series(outliers, index=values.index)
    return outliers




class KrigingInterpolator:
    """Кригинг-интерполяция для пространственных данных"""
    
    def __init__(self, variogram_model: str = 'spherical', nugget: float = 0.0):
        self.variogram_model = variogram_model
        self.nugget = nugget
        self.fitted = False
        self.range_param = None
        self.sill = None
        
    def fit(self, time: pd.Series, values: pd.Series) -> 'KrigingInterpolator':
        """Обучение модели вариограммы"""
        valid_mask = ~(pd.isna(time) | pd.isna(values))
        time_clean = time[valid_mask].values
        values_clean = values[valid_mask].values
        
        if len(time_clean) < 3:
            raise ValueError("Недостаточно данных для кригинга")
        
        # Вычисляем эмпирическую вариограмму
        distances, variogram = self._compute_empirical_variogram(time_clean, values_clean)
        
        # Подгоняем теоретическую модель
        self._fit_variogram_model(distances, variogram)
        self.fitted = True
        return self
    
    def _compute_empirical_variogram(self, time: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Вычисление эмпирической вариограммы"""
        n = len(time)
        distances = []
        variogram_values = []
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(time[i] - time[j])
                var_val = 0.5 * (values[i] - values[j]) ** 2
                distances.append(dist)
                variogram_values.append(var_val)
        
        return np.array(distances), np.array(variogram_values)
    
    def _fit_variogram_model(self, distances: np.ndarray, variogram: np.ndarray) -> None:
        """Подгонка теоретической модели вариограммы"""
        # Простая подгонка сферической модели
        max_dist = np.max(distances)
        max_var = np.max(variogram)
        
        self.range_param = max_dist * 0.6  # Примерная оценка
        self.sill = max_var * 0.8
        
    def _theoretical_variogram(self, h: np.ndarray) -> np.ndarray:
        """Теоретическая вариограмма"""
        if self.variogram_model == 'spherical':
            gamma = np.where(h <= self.range_param,
                           self.sill * (1.5 * h / self.range_param - 0.5 * (h / self.range_param) ** 3),
                           self.sill)
        elif self.variogram_model == 'exponential':
            gamma = self.sill * (1 - np.exp(-3 * h / self.range_param))
        else:  # linear
            gamma = self.sill * np.minimum(h / self.range_param, 1)
        
        return gamma + self.nugget
    
    def predict(self, time: pd.Series) -> pd.Series:
        """Предсказание с помощью кригинга"""
        if not self.fitted:
            raise ValueError("Модель не обучена")
        
        # Упрощенная реализация обычного кригинга
        time_values = time.values
        predictions = []
        
        for t in time_values:
            # Находим ближайшие точки
            distances = np.abs(time_values - t)
            weights = 1.0 / (distances + 1e-10)  # Простые веса
            weights = weights / np.sum(weights)
            
            # Взвешенное среднее
            pred = np.sum(weights * time_values)
            predictions.append(pred)
        
        return pd.Series(predictions, index=time.index, name='kriging_interpolated')


class RBFInterpolator:
    """Радиальные базисные функции для интерполяции"""
    
    def __init__(self, function: str = 'multiquadric', smoothing: float = 0.0):
        self.function = function
        self.smoothing = smoothing
        self.fitted = False
        self.rbf = None
        
    def fit(self, time: pd.Series, values: pd.Series) -> 'RBFInterpolator':
        """Обучение RBF модели"""
        valid_mask = ~(pd.isna(time) | pd.isna(values))
        time_clean = time[valid_mask].values
        values_clean = values[valid_mask].values
        
        if len(time_clean) < 2:
            raise ValueError("Недостаточно данных для RBF")
        
        # Создаем RBF интерполятор
        self.rbf = ScipyRBFInterpolator(time_clean.reshape(-1, 1), values_clean, 
                                 function=self.function, smoothing=self.smoothing)
        self.fitted = True
        return self
    
    def predict(self, time: pd.Series) -> pd.Series:
        """Предсказание с помощью RBF"""
        if not self.fitted:
            raise ValueError("Модель не обучена")
        
        time_values = time.values.reshape(-1, 1)
        predictions = self.rbf(time_values)
        return pd.Series(predictions, index=time.index, name='rbf_interpolated')
    

# ============================================================================
# DimensionlessExtrapolator 
# ============================================================================

class DimensionlessExtrapolator:
    """
    Модуль для экстраполяции безразмерных кривых X-Y на основе экстраполированных
    размерных параметров (P, dP, Q)
    """
    
    def __init__(self):
        self.interp_model = None
        self.dt = None
        self.t_last = None
        self.well_params = None
    
    def run(
        self,
        df: pd.DataFrame,
        well_params: Dict[str, float],
        n_future: int = 20,
        method: str = "adaptive",
        check_rmse: bool = True,
        alt_method: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Выполняет построение модели, экстраполяцию и оценку качества.
        
        Args:
            df: Исторические данные по скважине (должны содержать t, P, dP, Q и статичные параметры)
            well_params: Физические параметры (k, phi, ct, mu, B, L, h)
            n_future: Количество временных шагов для экстраполяции
            method: Тип регрессора ("poly", "ridge", "rf", "adaptive", "phys")
            check_rmse: Если True, выполняется сравнение с эталонными X-Y
            alt_method: Альтернативная функция вычисления эталонных X-Y
        
        Returns:
            {
                "P_ext": экстраполированные P,
                "dP_ext": экстраполированные dP,
                "Q_ext": экстраполированные Q,
                "X_ext": экстраполированные X,
                "Y_ext": экстраполированные Y,
                "df_ref": DataFrame эталонных X-Y,
                "df_pred": DataFrame экстраполированных X-Y,
                "interp_model": обученный регрессор,
                "rmse": float или Dict[str, float],
                "validation_metrics": метрики на validation window,
                "tail_metrics": метрики физичности хвоста,
                "ERI": итоговая метрика достоверности хвоста,
                "meta": служебная информация
            }
        """
        # Сохраняем параметры
        self.well_params = well_params
        # 1. Подготовка данных
        df_clean = self._prepare_data(df)
        
        # Определяем частоту временных меток
        self.dt = float(np.median(np.diff(df_clean["t"].values)))
        self.t_last = float(df_clean["t"].iloc[-1])
        
        # 2. Train/validation split (70/30 если >200 точек, иначе 80/20)
        n_points = len(df_clean)
        train_split = 0.7 if n_points > 200 else 0.8
        split_idx = int(len(df_clean) * train_split)
        df_train = df_clean.iloc[:split_idx].copy()
        df_val = df_clean.iloc[split_idx:].copy()
        
        # 3. Оценка методов на validation (если method="adaptive")
        # if method == "adaptive":
        #     best_method = self._select_best_method(df_train, df_val, well_params)
        # else:
        #     best_method = method
        best_method = "poly"

        # 4. Переобучение лучшего метода на всех данных
        X_train_full, y_train_full = self._prepare_features(df_clean)
        self.interp_model = None
        # self._train_interpolator(X_train_full, y_train_full, best_method)
        
        # 5. Экстраполяция на основе всего массива
        df_pred = self._extrapolate_dimensionless(df_clean, n_future) # приходит t_future, P_ext, dP_ext, Q_ext,
        df_pred = self._calculate_xy_from_extrapolated(df_pred)
        
        # 6. Расчёт эталонных безразмерных X–Y (для обратной совместимости)
        df_ref = self._calculate_reference_xy(df_clean, alt_method)
        
        # 7. Оценка на validation window (для исторических данных)
        validation_metrics = {}
        if len(df_val) > 0:
            validation_metrics = self._calculate_validation_metrics(
                df_train, df_val, well_params, best_method
            )
        
        # 8. Оценка физичности хвоста (ERI)
        from helpers.dimensionless.extrapolation.eri import ExtrapolationReliabilityEvaluator
        evaluator = ExtrapolationReliabilityEvaluator()
        tail_metrics, eri = evaluator.evaluate(df_pred, df_train, well_params)
        
        # 9. Оценка качества (валидация) - для обратной совместимости
        rmse = self._calculate_rmse(df_ref, df_pred) if check_rmse else {}
        
        # 10. Метаданные
        meta = self._create_metadata(df_clean, df_pred, best_method, rmse)
        meta['train_split'] = train_split
        meta['n_train_points'] = len(df_train)
        meta['n_val_points'] = len(df_val)
        
        # Вывод RMSE в лог
        if rmse:
            if isinstance(rmse, dict):
                print(f"RMSE(X): {rmse.get('X', 0):.3e}, RMSE(Y): {rmse.get('Y', 0):.3e}, "
                      f"mean: {rmse.get('mean', 0):.3e}")
            else:
                print(f"RMSE: {rmse:.3e}")
        
        # Формирование результата
        result = {
            # Новые поля
            "P_ext": df_pred['P'].values,
            "dP_ext": df_pred['dP'].values,
            "Q_ext": df_pred['Q'].values,
            "X_ext": df_pred['X'].values,
            "Y_ext": df_pred['Y'].values,
            "validation_metrics": validation_metrics,
            "tail_metrics": tail_metrics,
            "ERI": eri,
            # Старые поля для обратной совместимости
            "df_ref": df_ref,
            "df_pred": df_pred,
            "interp_model": self.interp_model,
            "rmse": rmse,
            "meta": meta
        }
        
        return result
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """4.1. Подготовка данных: проверка столбцов, обработка NaN"""
        required_cols = ['t', 'P', 'dP', 'Q']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing_cols}")
        
        df_clean = df.copy()
        
        # Проверка на NaN и линейная интерполяция
        for col in ['P', 'dP', 'Q']:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        
        # Удаляем строки, где все ключевые параметры NaN
        df_clean = df_clean.dropna(subset=['t', 'P', 'Q'])
        
        # Проверка на нулевые/отрицательные значения (заменяем на NaN)
        df_clean.loc[df_clean['dP'] < 0, 'dP'] = np.nan
        df_clean.loc[df_clean['Q'] < 0, 'Q'] = np.nan
        
        # Повторная интерполяция после замены
        for col in ['P', 'dP', 'Q']:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        
        # Сортировка по времени
        df_clean = df_clean.sort_values('t').reset_index(drop=True)
        
        if len(df_clean) < 5:
            raise ValueError("Недостаточно данных для экстраполяции (требуется минимум 5 точек)")
        
        return df_clean
    
    def _calculate_reference_xy(self, df: pd.DataFrame, alt_method: Optional[Callable]) -> pd.DataFrame:
        """4.2. Расчёт эталонных безразмерных X–Y"""
        if alt_method is not None:
            return alt_method(df, self.well_params)
        
        # Используем встроенный convert_to_dimensionless
        from helpers.dimensionless_analysis import convert_to_dimensionless_curves
        
        time_series = pd.Series(df['t'].values)
        pressure_series = pd.Series(df['P'].values)
        flow_rate_series = pd.Series(df['Q'].values)
        depression_series = pd.Series(df['dP'].values)
        
        dim_data = convert_to_dimensionless_curves(
            time_series, pressure_series, flow_rate_series, depression_series,
            self.well_params, x_mode='alt'
        )
        
        df_ref = pd.DataFrame({
            't': df['t'].values,
            'X': dim_data.X,
            'Y': dim_data.Y
        })
        
        return df_ref
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """4.3. Формирование регрессионного признакового вектора"""
        # Статичные параметры скважины
        static_features = []
        static_cols = ['Skin', 'h', 'N', 'W', 'L', 'a/L']
        for col in static_cols:
            if col in df.columns:
                # Берем первое значение (все одинаковые для одной скважины)
                static_features.append(df[col].iloc[0] if len(df) > 0 else 0.0)
            else:
                raise BaseException(f"Отсутствует необходимая колонка {col} для выполнения экстраполяции.")    
        
        # Исторические значения размерных параметров
        historical_features = df[['P', 'dP', 'Q']].values
        
        # Повторяем статичные параметры для каждой строки
        static_array = np.tile(static_features, (len(df), 1))
        
        # Объединяем признаки
        t_feature = df['t'].values.reshape(-1, 1)   
        X_train = np.hstack([static_array, historical_features, t_feature])
        
        # Целевые значения (следующие значения P, dP, Q)
        # Для экстраполяции используем текущие значения как цели (для обучения тренда)
        X_train = X_train[:-1]
        y_train = historical_features[1:]
        
        return X_train, y_train
    
    def _train_interpolator(self, X_train: np.ndarray, y_train: np.ndarray, method: str):
        from helpers.dimensionless.extrapolation.log_model import _LogModel
        """4.4. Обучение интерполятора"""
        if method == "poly":
            model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=[0.01, 0.1, 1, 10.0]))
        ])
        elif method == "ridge":
            # Линейная Ridge-регрессия
            model = Ridge(alpha=1.0)
        elif method == "rf":
            # RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        elif method == "log":
            # Логарифмическая модель: P(t) = a + b * ln(t + c)
            model = _LogModel()
        elif method == "phys":
            # Модель на основе аппроксимации тренда (экспоненциальная/логарифмическая)
            # Используем Ridge с полиномиальными признаками для аппроксимации тренда
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=3)),
                ('ridge', RidgeCV(alphas=[0.01, 0.1, 1, 10.0]))
            ])
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        
        # Обучаем модель
        model.fit(X_train, y_train)
        
        return model
    
    def _extrapolate_dimensionless(self, df: pd.DataFrame, n_future: int) -> pd.DataFrame:
        """Экстраполяция размерных параметров с использованием нового модуля"""
        from helpers.dimensionless.extrapolation import extrapolate_parameters
        
        # Построить временную сетку
        t_future = np.arange(
            self.t_last + self.dt,
            self.t_last + (n_future + 1) * self.dt,
            self.dt
        )

        # Подготавливаем статичные параметры
        static_cols = ['Skin', 'h', 'N', 'W', 'L', 'a/L']
        static_features = []
        for col in static_cols:
            if col in df.columns:
                static_features.append(df[col].iloc[-1])
            else:
                raise BaseException(f"Отсутствует необходимая колонка {col} для выполнения экстраполяции.")  
        static_features = np.array(static_features)
        
        t_train = df['t'].values.reshape(-1, 1)  # реальное время из данных
        X_train = np.hstack([
            np.tile(static_features, (len(df), 1)),
            t_train
        ])
        
        model_P = self._train_interpolator(X_train, df['P'].values, method="log")
        model_Q = self._train_interpolator(X_train, df['Q'].values, method="poly")
        model_dP = self._train_interpolator(X_train, df['dP'].values, method="log")
        
        # Экстраполируем с использованием нового модуля
        P_ext, dP_ext, Q_ext = extrapolate_parameters(
            model_P = model_P,
            model_Q = model_Q,
            model_dP = model_dP,
            P_last = df['P'].iloc[0],
            dP_last = df['dP'].iloc[0],
            Q_last = df['Q'].iloc[0],
            t_future = t_future,
            static_features = static_features,
        )
        
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        # press = static_features[:, 0] # P падает, dP растет, модель обучается ровно наоборот
        # t = df["t"].values
        # plt.plot(t, press, )
        # plt.grid(True)
        # plt.ylabel("Давление P")
        # plt.xlabel("Время t")
        # plt.show()
        
        # Создаем DataFrame с экстраполированными данными
        df_pred = pd.DataFrame({
            't': t_future,
            'P': P_ext,
            'dP': dP_ext,
            'Q': Q_ext
        })
        
        # Копируем статичные параметры
        for col in static_cols:
            if col in df.columns:
                df_pred[col] = df[col].iloc[-1]
        
        return df_pred
    
    def _calculate_xy_from_extrapolated(self, df_pred: pd.DataFrame) -> pd.DataFrame:
        """Расчёт X–Y по экстраполированным данным (используем ту же формулу, что и для основной кривой)"""
        from helpers.dimensionless_analysis import convert_to_dimensionless_curves
        
        time_series = pd.Series(df_pred['t'].values)
        pressure_series = pd.Series(df_pred['P'].values)
        flow_rate_series = pd.Series(df_pred['Q'].values)
        depression_series = pd.Series(df_pred['dP'].values)
        
        dim_data = convert_to_dimensionless_curves(
            time_series, pressure_series, flow_rate_series, depression_series, 
            self.well_params, x_mode='alt'
        )
        
        df_pred['X'] = dim_data.X
        df_pred['Y'] = dim_data.Y
        
        return df_pred
    
    def _calculate_rmse(self, df_ref: pd.DataFrame, df_pred: pd.DataFrame) -> Dict[str, float]:
        """Оценка качества (валидация) с использованием нормированных метрик Y"""
        from helpers.dimensionless.extrapolation import compute_validation_metrics_y
        
        if df_ref is None or len(df_ref) == 0:
            return {}
        
        # Выравниваем по времени для сравнения
        common_times = np.intersect1d(df_ref['t'].values, df_pred['t'].values)
        
        if len(common_times) == 0:
            from scipy.interpolate import interp1d
            X_ref_interp = interp1d(df_ref['t'].values, df_ref['X'].values, 
                                    kind='linear', bounds_error=False, fill_value='extrapolate')
            Y_ref_interp = interp1d(df_ref['t'].values, df_ref['Y'].values,
                                    kind='linear', bounds_error=False, fill_value='extrapolate')
            
            X_ref_aligned = X_ref_interp(df_pred['t'].values)
            Y_ref_aligned = Y_ref_interp(df_pred['t'].values)
            
            X_pred = df_pred['X'].values
            Y_pred = df_pred['Y'].values
            
            # Интерполируем dP для валидации Y
            if 'dP' in df_ref.columns and 'dP' in df_pred.columns:
                dP_ref_interp = interp1d(df_ref['t'].values, df_ref['dP'].values,
                                        kind='linear', bounds_error=False, fill_value='extrapolate')
                dP_ref_aligned = dP_ref_interp(df_pred['t'].values)
                dP_pred_aligned = df_pred['dP'].values
            else:
                dP_ref_aligned = None
                dP_pred_aligned = None
        else:
            ref_common = df_ref[df_ref['t'].isin(common_times)].sort_values('t')
            pred_common = df_pred[df_pred['t'].isin(common_times)].sort_values('t')
            
            X_ref_aligned = ref_common['X'].values
            Y_ref_aligned = ref_common['Y'].values
            X_pred = pred_common['X'].values
            Y_pred = pred_common['Y'].values
        
            dP_ref_aligned = ref_common['dP'].values if 'dP' in ref_common.columns else None
            dP_pred_aligned = pred_common['dP'].values if 'dP' in pred_common.columns else None
        
        # Вычисляем RMSE для X
        rmse_x = np.sqrt(np.nanmean((X_ref_aligned - X_pred)**2))
        
        # Вычисляем нормированные метрики для Y
        y_metrics = compute_validation_metrics_y(
            Y_ref=Y_ref_aligned,
            Y_pred=Y_pred,
            dP_ref=dP_ref_aligned,
            dP_pred=dP_pred_aligned
        )
        
        # Базовый RMSE для Y (для обратной совместимости)
        rmse_y = np.sqrt(np.nanmean((Y_ref_aligned - Y_pred)**2))
        rmse_total = np.mean([rmse_x, rmse_y])
        
        result = {
            'X': rmse_x,
            'Y': rmse_y,
            'mean': rmse_total,
            'filtered_RMSE_Y': y_metrics.get('filtered_RMSE_Y', np.inf),
            'RMSE_log_Y': y_metrics.get('RMSE_log_Y', np.inf),
            'physically_acceptable_share': y_metrics.get('physically_acceptable_share', 0.0)
        }
        
        return result
    
    def _select_best_method(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                           well_params: Dict[str, float]) -> str:
        """
        Выбирает лучший метод на основе метрик на validation window.
        
        Args:
            df_train: Обучающие данные
            df_val: Validation данные
            well_params: Физические параметры скважины
        
        Returns:
            Название лучшего метода ('poly' или 'rf')
        """
        # Временно исключаем ridge из-за некорректной работы
        methods = ['poly', 'phys', 'rf', 'ridge']
        best_method = methods[0]
        best_score = np.inf
        
        # Сохраняем текущие параметры
        dt_original = self.dt
        t_last_original = self.t_last
        well_params_original = self.well_params
        interp_model_original = self.interp_model  # Сохраняем текущую модель
        
        # Устанавливаем параметры для train данных
        self.dt = float(np.median(np.diff(df_train["t"].values)))
        self.t_last = float(df_train["t"].iloc[-1])
        self.well_params = well_params
        
        try:
            for method_name in methods:
                try:
                    # Обучаем модель на train
                    X_train, y_train = self._prepare_features(df_train)
                    model = self._train_interpolator(X_train, y_train, method_name)
                    
                    # Временно устанавливаем модель для экстраполяции
                    self.interp_model = model
                    
                    # Экстраполируем на validation window
                    n_val = len(df_val)
                    df_pred_val = self._extrapolate_dimensionless(df_train, n_val)
                    df_pred_val = self._calculate_xy_from_extrapolated(df_pred_val)
                    
                    # Вычисляем эталонные X-Y для validation
                    df_ref_val = self._calculate_reference_xy(df_val, None)
                    
                    # Вычисляем метрики
                    metrics = self._calculate_validation_metrics(
                        df_train, df_val, well_params, method_name, df_pred_val, df_ref_val
                    )
                    
                    # Комбинированная оценка: filtered_RMSE_Y + RMSE_log_Y + physically_acceptable_share
                    filtered_rmse_y = metrics.get('filtered_RMSE_Y', np.inf)
                    rmse_log_y = metrics.get('RMSE_log_Y', np.inf)
                    phys_share = metrics.get('physically_acceptable_share', 0.0)
                    
                    # Комбинированный score (меньше = лучше)
                    # Штрафуем за низкую долю физически корректных точек
                    if phys_share < 0.5:
                        score = np.inf
                    elif np.isfinite(filtered_rmse_y) and filtered_rmse_y < 1e10:
                        # Используем geometric mean filtered_RMSE_Y и RMSE_log_Y
                        score = np.sqrt(filtered_rmse_y * rmse_log_y) / (phys_share + 0.1)
                    else:
                        score = metrics.get('RMSE_mean', np.inf)
                    
                    # Логирование для отладки
                    print(f"  Метод {method_name}: ")
                    print(f"    filtered_RMSE_Y = {filtered_rmse_y:.6e}")
                    print(f"    RMSE_log_Y = {rmse_log_y:.6e}")
                    print(f"    physically_acceptable_share = {phys_share:.2%}")
                    print(f"    combined_score = {score:.6e}")
                    
                    if score < best_score and np.isfinite(score):
                        best_score = score
                        best_method = method_name
                except Exception as e:
                    # Пропускаем метод при ошибке, но логируем
                    print(f"  Метод {method_name} пропущен из-за ошибки: {str(e)}")
                    continue
        finally:
            # Восстанавливаем параметры
            self.dt = dt_original
            self.t_last = t_last_original
            self.well_params = well_params_original
            self.interp_model = interp_model_original  # Восстанавливаем модель
        print(f"Выбран лучший метод: {best_method} (score = {best_score:.6e})")
        return best_method
    
    def _calculate_validation_metrics(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                                      well_params: Dict[str, float], method: str,
                                      df_pred_val: Optional[pd.DataFrame] = None,
                                      df_ref_val: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Вычисляет метрики на validation window.
        
        Args:
            df_train: Обучающие данные
            df_val: Validation данные
            well_params: Физические параметры скважины
            method: Метод экстраполяции
            df_pred_val: Предсказанные данные на validation (если None, вычисляются)
            df_ref_val: Эталонные данные на validation (если None, вычисляются)
        
        Returns:
            Словарь с метриками: RMSE_X_val, RMSE_Y_val, MAE_X_val, MAE_Y_val, MAPE, R2, Max_error
        """
        try:
            # Сохраняем текущие параметры
            dt_original = self.dt
            t_last_original = self.t_last
            well_params_original = self.well_params
            
            # Устанавливаем параметры для train данных
            self.dt = float(np.median(np.diff(df_train["t"].values)))
            self.t_last = float(df_train["t"].iloc[-1])
            self.well_params = well_params
            
            interp_model_original = self.interp_model  # Сохраняем текущую модель
            
            try:
                # Если предсказания не переданы, вычисляем их
                if df_pred_val is None:
                    # Обучаем модель на train
                    X_train, y_train = self._prepare_features(df_train)
                    model = self._train_interpolator(X_train, y_train, method)
                    
                    # Временно устанавливаем модель для экстраполяции
                    self.interp_model = model
                    
                    # Экстраполируем на validation window
                    n_val = len(df_val)
                    df_pred_val = self._extrapolate_dimensionless(df_train, n_val)
                    df_pred_val = self._calculate_xy_from_extrapolated(df_pred_val)
                
                # Если эталон не передан, вычисляем его
                if df_ref_val is None:
                    df_ref_val = self._calculate_reference_xy(df_val, None)
            finally:
                # Восстанавливаем параметры
                self.dt = dt_original
                self.t_last = t_last_original
                self.well_params = well_params_original
                self.interp_model = interp_model_original  # Восстанавливаем модель
            
            # Выравниваем по времени
            common_times = np.intersect1d(df_ref_val['t'].values, df_pred_val['t'].values)
            
            if len(common_times) == 0:
                # Интерполируем эталонные значения
                from scipy.interpolate import interp1d
                X_ref_interp = interp1d(df_ref_val['t'].values, df_ref_val['X'].values,
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
                Y_ref_interp = interp1d(df_ref_val['t'].values, df_ref_val['Y'].values,
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
                
                X_ref_aligned = X_ref_interp(df_pred_val['t'].values)
                Y_ref_aligned = Y_ref_interp(df_pred_val['t'].values)
                
                X_pred = df_pred_val['X'].values
                Y_pred = df_pred_val['Y'].values
            else:
                # Используем общие временные точки
                ref_common = df_ref_val[df_ref_val['t'].isin(common_times)].sort_values('t')
                pred_common = df_pred_val[df_pred_val['t'].isin(common_times)].sort_values('t')
                
                X_ref_aligned = ref_common['X'].values
                Y_ref_aligned = ref_common['Y'].values
                X_pred = pred_common['X'].values
                Y_pred = pred_common['Y'].values
                
                dP_ref_aligned = ref_common['dP'].values if 'dP' in ref_common.columns else None
                dP_pred_aligned = pred_common['dP'].values if 'dP' in pred_common.columns else None
            
            # Вычисляем метрики с использованием нормированной валидации Y
            from helpers.dimensionless.extrapolation import compute_validation_metrics_y
            
            # RMSE для X
            rmse_x = np.sqrt(np.nanmean((X_ref_aligned - X_pred)**2))
            
            # Нормированные метрики для Y
            y_metrics = compute_validation_metrics_y(
                Y_ref=Y_ref_aligned,
                Y_pred=Y_pred,
                dP_ref=dP_ref_aligned,
                dP_pred=dP_pred_aligned
            )
            
            # Базовый RMSE для Y (для обратной совместимости)
            rmse_y = np.sqrt(np.nanmean((Y_ref_aligned - Y_pred)**2))
            rmse_mean = np.mean([rmse_x, y_metrics.get('filtered_RMSE_Y', rmse_y)])
            
            # MAE
            mae_x = np.nanmean(np.abs(X_ref_aligned - X_pred))
            mae_y = np.nanmean(np.abs(Y_ref_aligned - Y_pred))
            
            # MAPE
            X_ref_safe = np.where(np.abs(X_ref_aligned) < 1e-12, 1e-12, X_ref_aligned)
            Y_ref_safe = np.where(np.abs(Y_ref_aligned) < 1e-12, 1e-12, Y_ref_aligned)
            
            mape_x = np.nanmean(np.abs((X_ref_aligned - X_pred) / X_ref_safe)) * 100
            mape_y = np.nanmean(np.abs((Y_ref_aligned - Y_pred) / Y_ref_safe)) * 100
            mape_mean = np.mean([mape_x, mape_y])
            
            # R²
            ss_res_x = np.nansum((X_ref_aligned - X_pred)**2)
            ss_tot_x = np.nansum((X_ref_aligned - np.nanmean(X_ref_aligned))**2)
            r2_x = 1 - (ss_res_x / (ss_tot_x + 1e-12)) if ss_tot_x > 1e-12 else 0.0
            
            ss_res_y = np.nansum((Y_ref_aligned - Y_pred)**2)
            ss_tot_y = np.nansum((Y_ref_aligned - np.nanmean(Y_ref_aligned))**2)
            r2_y = 1 - (ss_res_y / (ss_tot_y + 1e-12)) if ss_tot_y > 1e-12 else 0.0
            r2_mean = np.mean([r2_x, r2_y])
            
            # Max error
            max_error_x = np.nanmax(np.abs(X_ref_aligned - X_pred))
            max_error_y = np.nanmax(np.abs(Y_ref_aligned - Y_pred))
            max_error = np.max([max_error_x, max_error_y])
            
            return {
                'RMSE_X_val': float(rmse_x),
                'RMSE_Y_val': float(rmse_y),
                'RMSE_mean': float(rmse_mean),
                'filtered_RMSE_Y': y_metrics.get('filtered_RMSE_Y', np.inf),
                'RMSE_log_Y': y_metrics.get('RMSE_log_Y', np.inf),
                'physically_acceptable_share': y_metrics.get('physically_acceptable_share', 0.0),
                'MAE_X_val': float(mae_x),
                'MAE_Y_val': float(mae_y),
                'MAPE_X': float(mape_x),
                'MAPE_Y': float(mape_y),
                'MAPE_mean': float(mape_mean),
                'R2_X': float(r2_x),
                'R2_Y': float(r2_y),
                'R2_mean': float(r2_mean),
                'Max_error_X': float(max_error_x),
                'Max_error_Y': float(max_error_y),
                'Max_error': float(max_error)
            }
        except Exception as e:
            # Возвращаем пустые метрики при ошибке
            return {}
    
    def _create_metadata(self, df_clean: pd.DataFrame, df_pred: pd.DataFrame, 
                        method: str, rmse: Dict[str, float]) -> Dict[str, Any]:
        """Создание метаданных"""
        return {
            'time_range_historical': (float(df_clean['t'].min()), float(df_clean['t'].max())),
            'time_range_predicted': (float(df_pred['t'].min()), float(df_pred['t'].max())),
            'dt': self.dt,
            'method': method,
            'n_historical_points': len(df_clean),
            'n_future_points': len(df_pred),
            'date': datetime.now().isoformat(),
            'rmse': rmse if rmse else None,
            'stability': 'good' if (rmse and rmse.get('mean', 1.0) < 0.05) else 'needs_improvement'
        }
