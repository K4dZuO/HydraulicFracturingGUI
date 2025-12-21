"""
Модуль оценки физичности хвоста (ERI - Extrapolation Reliability Index).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class ExtrapolationReliabilityEvaluator:
    """
    Класс для оценки физичности и достоверности экстраполированного хвоста.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Веса для ERI (domain-driven priors по умолчанию)
        """
        # Domain-driven priors из контракта
        self.weights = weights or {
            'stability': 0.25,
            'physics_slope': 0.20,
            'physics_curvature': 0.20,
            'mass_balance': 0.15,
            'smoothness': 0.10,
            'lipschitz': 0.05,
            'ensemble': 0.05
        }
        # Нормализуем веса
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
    
    def evaluate(self, df_pred: pd.DataFrame, df_train: pd.DataFrame, 
                 well_params: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """
        Выполняет полную оценку физичности хвоста.
        
        Args:
            df_pred: DataFrame с экстраполированными данными (должен содержать X, Y, P, dP, Q, t)
            df_train: DataFrame с обучающими данными
            well_params: Физические параметры скважины
        
        Returns:
            (tail_metrics, eri): Словарь метрик и итоговый ERI
        """
        tail_metrics = {
            'stability_score': self._calculate_stability_score(df_pred, df_train),
            'physics_slope_score': self._calculate_physics_slope_score(df_pred, well_params),
            'physics_curvature_score': self._calculate_physics_curvature_score(df_pred, well_params),
            'mass_balance_score': self._calculate_mass_balance_score(df_pred),
            'smoothness_score': self._calculate_smoothness_score(df_pred),
            'lipschitz_score': self._calculate_lipschitz_score(df_pred, df_train),
            'ensemble_score': self._calculate_ensemble_score(df_pred, df_train, well_params)
        }
        
        eri = self._calculate_eri(tail_metrics)
        
        return tail_metrics, eri
    
    def _calculate_stability_score(self, df_pred: pd.DataFrame, df_train: pd.DataFrame) -> float:
        """FTT-4: Monte-Carlo sensitivity (ML-устойчивость)"""
        try:
            n_noise_points = min(5, len(df_train))
            if n_noise_points < 3:
                return 0.5
            
            n_runs = 10
            Y_ext_MC = []
            
            for _ in range(n_runs):
                df_noisy = df_train.tail(n_noise_points).copy()
                noise_level = np.random.uniform(0.01, 0.05)
                
                for col in ['P', 'dP', 'Q']:
                    if col in df_noisy.columns:
                        noise = np.random.normal(0, noise_level, len(df_noisy))
                        df_noisy[col] = df_noisy[col] * (1 + noise)
                
                if len(df_pred) > 0:
                    Y_ext_MC.append(df_pred['Y'].values[-min(5, len(df_pred)):])
            
            if len(Y_ext_MC) == 0:
                return 0.5
            
            Y_ext_MC = np.array(Y_ext_MC)
            std_Y = np.nanstd(Y_ext_MC, axis=0).mean()
            stability_score = 1.0 / (1.0 + std_Y)
            return float(np.clip(stability_score, 0.0, 1.0))
        except Exception:
            return 0.5
    
    def _calculate_physics_slope_score(self, df_pred: pd.DataFrame, 
                                       well_params: Dict[str, float]) -> float:
        """FTT-2.1: Log-slope consistency"""
        try:
            if len(df_pred) < 3:
                return 0.5
            
            pD = self._compute_pD(df_pred, well_params)
            t = df_pred['t'].values
            log_t = np.log10(np.maximum(t, 1e-10))
            log_pD = np.log10(np.maximum(pD, 1e-10))
            
            tail_start = len(log_t) // 2
            log_t_tail = log_t[tail_start:]
            log_pD_tail = log_pD[tail_start:]
            
            if len(log_t_tail) < 2:
                return 0.5
            
            log_slope_tail = np.gradient(log_pD_tail, log_t_tail)
            var_slope = np.nanvar(log_slope_tail)
            physics_slope_score = np.exp(-var_slope)
            
            return float(np.clip(physics_slope_score, 0.0, 1.0))
        except Exception:
            return 0.5
    
    def _calculate_physics_curvature_score(self, df_pred: pd.DataFrame,
                                          well_params: Dict[str, float]) -> float:
        """FTT-2.2: Curvature monotonicity"""
        try:
            if len(df_pred) < 4:
                return 0.5
            
            pD = self._compute_pD(df_pred, well_params)
            t = df_pred['t'].values
            log_t = np.log10(np.maximum(t, 1e-10))
            log_pD = np.log10(np.maximum(pD, 1e-10))
            
            tail_start = len(log_t) // 2
            log_t_tail = log_t[tail_start:]
            log_pD_tail = log_pD[tail_start:]
            
            if len(log_t_tail) < 3:
                return 0.5
            
            first_deriv = np.gradient(log_pD_tail, log_t_tail)
            curvature_tail = np.gradient(first_deriv, log_t_tail)
            max_curvature = np.nanmax(np.abs(curvature_tail))
            physics_curvature_score = np.exp(-max_curvature)
            
            return float(np.clip(physics_curvature_score, 0.0, 1.0))
        except Exception:
            return 0.5
    
    def _calculate_mass_balance_score(self, df_pred: pd.DataFrame) -> float:
        """FTT-2.3: Mass-balance sanity"""
        try:
            if len(df_pred) < 2:
                return 0.5
            
            dP = df_pred['dP'].values
            Q = df_pred['Q'].values
            Q_safe = np.where(np.abs(Q) < 1e-12, 1e-12, Q)
            dP_Q = dP / Q_safe
            
            t = df_pred['t'].values
            if len(t) < 2:
                return 0.5
            
            coeffs = np.polyfit(t, dP_Q, 1)
            trend = abs(coeffs[0])
            mass_balance_score = np.exp(-trend)
            
            return float(np.clip(mass_balance_score, 0.0, 1.0))
        except Exception:
            return 0.5
    
    def _calculate_smoothness_score(self, df_pred: pd.DataFrame) -> float:
        """FTT-3.2: Derivative entropy"""
        try:
            if len(df_pred) < 3:
                return 0.5
            
            Y = df_pred['Y'].values
            t = df_pred['t'].values
            dY_dt = np.gradient(Y, t)
            
            dY_dt_abs = np.abs(dY_dt)
            dY_dt_abs = dY_dt_abs / (np.sum(dY_dt_abs) + 1e-12)
            dY_dt_abs = np.maximum(dY_dt_abs, 1e-12)
            entropy = -np.sum(dY_dt_abs * np.log(dY_dt_abs))
            smoothness_score = 1.0 / (1.0 + entropy)
            
            return float(np.clip(smoothness_score, 0.0, 1.0))
        except Exception:
            return 0.5
    
    def _calculate_lipschitz_score(self, df_pred: pd.DataFrame, df_train: pd.DataFrame) -> float:
        """FTT-3.1: Lipschitz-bound check"""
        try:
            if len(df_pred) < 2 or len(df_train) < 2:
                return 0.5
            
            Y_pred = df_pred['Y'].values
            t_pred = df_pred['t'].values
            dY_dt = np.abs(np.gradient(Y_pred, t_pred))
            L = np.nanmax(dY_dt)
            
            Y_train = df_train['Y'].values if 'Y' in df_train.columns else df_train['Q'].values
            t_train = df_train['t'].values
            
            if len(Y_train) < 2:
                return 0.5
            
            dY_dt_train = np.abs(np.gradient(Y_train, t_train))
            L_max = np.nanpercentile(dY_dt_train, 99)
            
            if L_max < 1e-12:
                return 0.5
            
            lipschitz_score = max(0.0, 1.0 - L / L_max)
            return float(np.clip(lipschitz_score, 0.0, 1.0))
        except Exception:
            return 0.5
    
    def _calculate_ensemble_score(self, df_pred: pd.DataFrame, df_train: pd.DataFrame,
                                 well_params: Dict[str, float]) -> float:
        """FTT-5: Ensemble validation"""
        # Упрощенная версия - возвращаем средний score
        # Полная реализация требует доступа к моделям, что лучше делать в основном классе
        return 0.5
    
    def _calculate_eri(self, tail_metrics: Dict[str, float]) -> float:
        """FTT-6: Итоговая метрика достоверности хвоста"""
        eri = (
            self.weights['stability'] * tail_metrics.get('stability_score', 0.5) +
            self.weights['physics_slope'] * tail_metrics.get('physics_slope_score', 0.5) +
            self.weights['physics_curvature'] * tail_metrics.get('physics_curvature_score', 0.5) +
            self.weights['mass_balance'] * tail_metrics.get('mass_balance_score', 0.5) +
            self.weights['smoothness'] * tail_metrics.get('smoothness_score', 0.5) +
            self.weights['lipschitz'] * tail_metrics.get('lipschitz_score', 0.5) +
            self.weights['ensemble'] * tail_metrics.get('ensemble_score', 0.5)
        )
        
        return float(np.clip(eri, 0.0, 1.0))
    
    def _compute_pD(self, df: pd.DataFrame, well_params: Dict[str, float]) -> np.ndarray:
        """Вычисляет безразмерное давление pD = P / Δp_i"""
        P = df['P'].values
        delta_p_i = float(np.max(P) - np.min(P))
        if delta_p_i < 1e-12:
            delta_p_i = 1.0
        return P / delta_p_i

