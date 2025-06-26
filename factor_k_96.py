#!/usr/bin/env python3
"""
Factor K Elite 9.6 Enhanced - Temporal Analysis Revolution
Sistema Avanzado de Evaluación de Estrategias de Trading
Autor: Jose Livan Maseda Pereira
Versión: 9.6 Enhanced
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings("ignore")


class FactorKElite96:
    """
    Sistema de evaluación Factor K Elite 9.6 Enhanced
    Incluye análisis temporal IS/OOS, machine learning y Calmar Ratio
    """

    def __init__(self):
        # Categorías de ranking
        self.rank_categories = {
            "Elite": {"min": 9.2, "max": 10.0, "color": "gold"},
            "Excellent": {"min": 8.2, "max": 9.1, "color": "silver"},
            "Very Good": {"min": 7.2, "max": 8.1, "color": "bronze"},
            "Good": {"min": 6.2, "max": 7.1, "color": "green"},
            "Average": {"min": 5.2, "max": 6.1, "color": "yellow"},
            "Below Average": {"min": 4.2, "max": 5.1, "color": "orange"},
            "Poor": {"min": 3.2, "max": 4.1, "color": "red"},
            "Very Poor": {"min": 0.0, "max": 3.1, "color": "darkred"},
        }

        # Pesos por régimen de mercado
        self.regime_weights = {
            "bull": {
                "stability": 0.32,
                "growth": 0.33,
                "efficiency": 0.20,
                "consistency": 0.15,
            },
            "bear": {
                "stability": 0.42,
                "growth": 0.13,
                "efficiency": 0.25,
                "consistency": 0.20,
            },
            "sideways": {
                "stability": 0.28,
                "growth": 0.27,
                "efficiency": 0.25,
                "consistency": 0.20,
            },
            "crisis": {
                "stability": 0.48,
                "growth": 0.08,
                "efficiency": 0.24,
                "consistency": 0.20,
            },
        }

        # Parámetros sigmoideos por régimen
        self.sigmoid_params = {
            "bull": {"steepness": 1.1, "midpoint": 1.1, "scale": 10.0},
            "bear": {"steepness": 1.4, "midpoint": 0.9, "scale": 8.5},
            "sideways": {"steepness": 1.0, "midpoint": 1.0, "scale": 10.0},
            "crisis": {"steepness": 1.8, "midpoint": 0.7, "scale": 7.0},
        }

    def calculate_derived_metrics(self, df):
        """
        Calcula las 16 métricas derivadas automáticamente
        Lógica: IS + OOS = Total
        """
        print("Calculando métricas derivadas...")

        # 1. Net Profit Total
        if "Net profit (IS)" in df.columns and "Net profit (OOS)" in df.columns:
            df["Net_Profit_Total"] = df["Net profit (IS)"] + df["Net profit (OOS)"]
            df["Net_Profit_IS_Proportion"] = df["Net profit (IS)"] / df[
                "Net_Profit_Total"
            ].replace(0, 1)
            df["Net_Profit_OOS_Proportion"] = df["Net profit (OOS)"] / df[
                "Net_Profit_Total"
            ].replace(0, 1)

        # 2. Calmar Ratio Weighted (promedio ponderado por beneficios)
        if "Calmar Ratio (IS)" in df.columns and "Calmar Ratio (OOS)" in df.columns:
            weight_is = df["Net_Profit_IS_Proportion"]
            weight_oos = df["Net_Profit_OOS_Proportion"]
            df["Calmar_Ratio_Weighted"] = (
                df["Calmar Ratio (IS)"] * weight_is
                + df["Calmar Ratio (OOS)"] * weight_oos
            )

        # 3. Sharpe Average
        if "Sharpe (IS)" in df.columns and "Sharpe (OOS)" in df.columns:
            df["Sharpe_Average"] = (df["Sharpe (IS)"] + df["Sharpe (OOS)"]) / 2
            df["Sharpe_Consistency"] = np.minimum(
                df["Sharpe (OOS)"] / df["Sharpe (IS)"].replace(0, 1), 2.0
            )

        # 4. CAGR Average
        if "CAGR % (IS)" in df.columns and "CAGR % (OOS)" in df.columns:
            df["CAGR_Average"] = (df["CAGR % (IS)"] + df["CAGR % (OOS)"]) / 2
            df["CAGR_Consistency"] = np.minimum(
                df["CAGR % (OOS)"] / df["CAGR % (IS)"].replace(0, 1), 2.0
            )

        # 5. Win Rate Average
        if "Win rate (IS)" in df.columns and "Win rate (OOS)" in df.columns:
            df["WinRate_Average"] = (df["Win rate (IS)"] + df["Win rate (OOS)"]) / 2
            df["WinRate_Consistency"] = (
                1 - np.abs(df["Win rate (IS)"] - df["Win rate (OOS)"]) / 100
            )

        # 6. Profit Factor Average
        if "Profit factor (IS)" in df.columns and "Profit factor (OOS)" in df.columns:
            df["PF_Average"] = (
                df["Profit factor (IS)"] + df["Profit factor (OOS)"]
            ) / 2
            df["PF_Consistency"] = np.minimum(
                df["Profit factor (OOS)"] / df["Profit factor (IS)"].replace(0, 1), 2.0
            )

        # 7. Max Drawdown Total (el máximo de ambos períodos)
        if "Max DD % (IS)" in df.columns and "Max DD % (OOS)" in df.columns:
            df["Max_DD_Total"] = np.maximum(df["Max DD % (IS)"], df["Max DD % (OOS)"])
            df["DD_Degradation"] = (df["Max DD % (OOS)"] - df["Max DD % (IS)"]) / df[
                "Max DD % (IS)"
            ].replace(0, 1)

        # 8. Drawdown Total
        if "Drawdown % (IS)" in df.columns and "Drawdown % (OOS)" in df.columns:
            df["Drawdown_Total"] = np.maximum(
                df["Drawdown % (IS)"], df["Drawdown % (OOS)"]
            )
            df["DD_Consistency"] = 1 - np.abs(
                df["Drawdown % (IS)"] - df["Drawdown % (OOS)"]
            ) / df["Drawdown_Total"].replace(0, 1)

        # 9. Otras métricas útiles
        if "Stability (IS)" in df.columns:
            df["Stability_IS"] = df["Stability (IS)"]

        if (
            "Return/DD ratio (IS)" in df.columns
            and "Return/DD ratio (OOS)" in df.columns
        ):
            df["Ret_DD_Ratio_Average"] = (
                df["Return/DD ratio (IS)"] + df["Return/DD ratio (OOS)"]
            ) / 2

        if "Avg profit per month (%)" in df.columns:
            df["Avg_Profit_Month"] = df["Avg profit per month (%)"]

        if "Avg trade" in df.columns:
            df["Avg_Trade"] = df["Avg trade"]

        if "Win/loss ratio" in df.columns:
            df["Win_Loss_Ratio"] = df["Win/loss ratio"]

        return df

    def detect_market_regime(self, df):
        """
        Detecta el régimen de mercado usando K-Means clustering
        """
        print("Detectando regímenes de mercado...")

        # Features para detección de régimen
        features = []
        feature_cols = [
            "Max_DD_Total",
            "CAGR_Average",
            "Sharpe_Average",
            "Stability_IS",
            "Calmar_Ratio_Weighted",
        ]

        for col in feature_cols:
            if col in df.columns:
                features.append(df[col].fillna(0))

        if len(features) == 0:
            df["Market_Regime"] = "sideways"
            return df

        X = np.column_stack(features)

        # Normalización
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Mapeo de clusters a regímenes basado en características
        df["Market_Regime"] = "sideways"  # Default

        for idx, row in df.iterrows():
            # Reglas de asignación basadas en métricas
            if (
                row.get("Max_DD_Total", 0) > 1.5
                or row.get("Calmar_Ratio_Weighted", 999) < 1.5
            ):
                df.at[idx, "Market_Regime"] = "crisis"
            elif (
                row.get("CAGR_Average", 0) > 2.0
                and row.get("Sharpe_Average", 0) > 1.5
                and row.get("Calmar_Ratio_Weighted", 0) > 2.5
            ):
                df.at[idx, "Market_Regime"] = "bull"
            elif (
                row.get("Stability_IS", 0) > 0.8
                and row.get("Calmar_Ratio_Weighted", 0) > 2.0
            ):
                df.at[idx, "Market_Regime"] = "sideways"
            else:
                df.at[idx, "Market_Regime"] = "bear"

        return df

    def adaptive_normalize(self, values, metric_name, percentiles):
        """
        Normalización adaptativa basada en percentiles del dataset
        """
        normalized = np.zeros_like(values, dtype=float)

        for i, value in enumerate(values):
            if pd.isna(value):
                normalized[i] = 0
                continue

            if value >= percentiles[95]:
                normalized[i] = 2.0
            elif value >= percentiles[85]:
                ratio = (value - percentiles[85]) / (percentiles[95] - percentiles[85])
                normalized[i] = 1.7 + 0.3 * ratio
            elif value >= percentiles[75]:
                ratio = (value - percentiles[75]) / (percentiles[85] - percentiles[75])
                normalized[i] = 1.4 + 0.3 * ratio
            elif value >= percentiles[65]:
                ratio = (value - percentiles[65]) / (percentiles[75] - percentiles[65])
                normalized[i] = 1.1 + 0.3 * ratio
            elif value >= percentiles[50]:
                ratio = (value - percentiles[50]) / (percentiles[65] - percentiles[50])
                normalized[i] = 0.8 + 0.3 * ratio
            elif value >= percentiles[35]:
                ratio = (value - percentiles[35]) / (percentiles[50] - percentiles[35])
                normalized[i] = 0.5 + 0.3 * ratio
            elif value >= percentiles[20]:
                ratio = (value - percentiles[20]) / (percentiles[35] - percentiles[20])
                normalized[i] = 0.2 + 0.3 * ratio
            else:
                if percentiles[20] > 0:
                    normalized[i] = max(0.1, 0.2 * value / percentiles[20])
                else:
                    normalized[i] = 0.1

        return normalized

    def calculate_stability_enhanced(self, row, percentiles):
        """
        Calcula el componente de Estabilidad Ultra Avanzada Enhanced
        """
        # A. Retorno Ajustado por Riesgo (30%)
        sharpe_norm = self.adaptive_normalize(
            [row.get("Sharpe_Average", 0)], "Sharpe", percentiles["Sharpe_Average"]
        )[0]
        ret_dd_norm = self.adaptive_normalize(
            [row.get("Ret_DD_Ratio_Average", 0)],
            "RetDD",
            percentiles.get("Ret_DD_Ratio_Average", {50: 1}),
        )[0]
        annual_dd_ratio = row.get("CAGR_Average", 0) / max(
            row.get("Max_DD_Total", 1), 0.01
        )
        annual_dd_norm = self.adaptive_normalize(
            [annual_dd_ratio],
            "AnnualDD",
            {20: 0.5, 35: 0.8, 50: 1, 65: 2, 75: 3, 85: 4, 95: 5},
        )[0]

        risk_adj_score = 0.4 * sharpe_norm + 0.4 * ret_dd_norm + 0.2 * annual_dd_norm

        # B. Resistencia al Drawdown (25%)
        dd_protection = max(0, 1 - row.get("Max_DD_Total", 0) / 2)  # Penaliza DD > 2%
        stagnation_resistance = max(0, 1 - row.get("Drawdown_Total", 0) / 3)
        dd_consistency = row.get("DD_Consistency", 0.5)

        dd_resistance = (
            0.5 * dd_protection + 0.3 * stagnation_resistance + 0.2 * dd_consistency
        )

        # C. Robustez Estadística (20%)
        # SQN Score normalization (0.4 weight)
        sqn_score_norm = self.adaptive_normalize(
            [row.get("SQN", 0)],
            "SQN",
            {20: 0.5, 35: 1.0, 50: 1.5, 65: 2.0, 75: 2.5, 85: 3.0, 95: 4.0},
        )[0]

        # R-Expectancy normalization (0.3 weight)
        r_expectancy_norm = self.adaptive_normalize(
            [row.get("R_Expectancy", 0)],
            "R_Expectancy",
            {20: 0.1, 35: 0.2, 50: 0.3, 65: 0.4, 75: 0.5, 85: 0.7, 95: 1.0},
        )[0]

        # Trade Volume Factor (0.3 weight) - keep existing calculation
        trade_volume_factor = (
            min(row.get("# of trades", 0) / 500, 2) if "# of trades" in row else 1
        )

        # Calculate Statistical Robustness with new formula
        statistical_robustness = (
            0.4 * sqn_score_norm + 0.3 * r_expectancy_norm + 0.3 * trade_volume_factor
        )

        # D. Calmar Ratio Analysis (15%)
        calmar_weighted_norm = self.adaptive_normalize(
            [row.get("Calmar_Ratio_Weighted", 0)],
            "Calmar",
            percentiles["Calmar_Ratio_Weighted"],
        )[0]
        calmar_is_norm = (
            self.adaptive_normalize(
                [row.get("Calmar Ratio (IS)", 0)],
                "CalmarIS",
                {20: 1, 35: 1.2, 50: 1.5, 65: 2, 75: 2.5, 85: 3, 95: 4},
            )[0]
            if "Calmar Ratio (IS)" in row
            else calmar_weighted_norm
        )
        calmar_oos_norm = (
            self.adaptive_normalize(
                [row.get("Calmar Ratio (OOS)", 0)],
                "CalmarOOS",
                {20: 1, 35: 1.2, 50: 1.5, 65: 2, 75: 2.5, 85: 3, 95: 4},
            )[0]
            if "Calmar Ratio (OOS)" in row
            else calmar_weighted_norm
        )

        calmar_analysis = (
            0.60 * calmar_weighted_norm + 0.25 * calmar_is_norm + 0.15 * calmar_oos_norm
        )

        # E. Estabilidad Intrínseca (10%)
        stability_score = self.adaptive_normalize(
            [row.get("Stability_IS", 0)],
            "Stability",
            {20: 0.3, 35: 0.4, 50: 0.5, 65: 0.6, 75: 0.7, 85: 0.8, 95: 0.9},
        )[0]

        # Componente final de estabilidad
        stability_enhanced = (
            0.30 * risk_adj_score
            + 0.25 * dd_resistance
            + 0.20 * statistical_robustness
            + 0.15 * calmar_analysis
            + 0.10 * stability_score
        )

        return stability_enhanced

    def calculate_growth_enhanced(self, row, percentiles):
        """
        Calcula el componente de Crecimiento Inteligente Enhanced
        """
        # A. CAGR Normalizado (50%)
        cagr_norm = self.adaptive_normalize(
            [row.get("CAGR_Average", 0)], "CAGR", percentiles["CAGR_Average"]
        )[0]

        # B. Consistencia del Crecimiento (25%)
        growth_consistency = min(row.get("CAGR_Consistency", 1), 1.5) / 1.5

        # C. Escalabilidad Mensual (15%)
        monthly_scalability = (
            min(row.get("Avg_Profit_Month", 0) / 100, 2)
            if "Avg_Profit_Month" in row
            else 1
        )

        # D. Beneficio Total (10%)
        profit_component = min(row.get("Net_Profit_Total", 0) / 15000, 2)

        # Componente final de crecimiento
        growth_enhanced = (
            0.50 * cagr_norm
            + 0.25 * growth_consistency
            + 0.15 * monthly_scalability
            + 0.10 * profit_component
        )

        return growth_enhanced

    def calculate_efficiency_enhanced(self, row, percentiles):
        """
        Calcula el componente de Eficiencia Avanzada Enhanced
        """
        # A. Eficiencia del Profit Factor (40%)
        pf_avg = row.get("PF_Average", 1)
        pf_efficiency = min((pf_avg - 1) * 2, 2) if pf_avg > 1 else 0

        # B. Eficiencia por Operación (25%)
        trade_efficiency = (
            min(row.get("Avg_Trade", 0) / 50, 2) if "Avg_Trade" in row else 1
        )

        # C. Eficiencia Win/Loss (20%)
        wl_efficiency = min(row.get("Win_Loss_Ratio", 1), 2)

        # D. Eficiencia de Win Rate (15%)
        winrate_avg = row.get("WinRate_Average", 50)
        winrate_efficiency = max(0, 1 - abs(winrate_avg - 55) / 100) * 2

        # Componente final de eficiencia
        efficiency_enhanced = (
            0.40 * pf_efficiency
            + 0.25 * trade_efficiency
            + 0.20 * wl_efficiency
            + 0.15 * winrate_efficiency
        )

        return efficiency_enhanced

    def calculate_consistency_enhanced(self, row, percentiles):
        """
        Calcula el componente de Consistencia Multi-Temporal Enhanced
        """
        # A. Ratios IS/OOS Principales (40%)
        sharpe_consistency = row.get("Sharpe_Consistency", 1)
        cagr_consistency = row.get("CAGR_Consistency", 1)
        pf_consistency = row.get("PF_Consistency", 1)
        main_ratios = (sharpe_consistency + cagr_consistency + pf_consistency) / 3

        # B. Degradación de Beneficios (25%)
        profit_degradation = (
            min(
                row.get("Net_Profit_OOS_Proportion", 0.5)
                / row.get("Net_Profit_IS_Proportion", 0.5),
                1.5,
            )
            if row.get("Net_Profit_IS_Proportion", 0) > 0
            else 1
        )

        # C. Consistencia Win Rate (20%)
        winrate_consistency = row.get("WinRate_Consistency", 0.8)

        # D. Estabilidad Temporal (15%)
        stability_norm = self.adaptive_normalize(
            [row.get("Stability_IS", 0)],
            "Stability",
            {20: 0.3, 35: 0.4, 50: 0.5, 65: 0.6, 75: 0.7, 85: 0.8, 95: 0.9},
        )[0]
        dd_consistency = row.get("DD_Consistency", 0.7)
        temporal_stability = 0.60 * stability_norm + 0.40 * dd_consistency

        # Componente final de consistencia
        consistency_enhanced = (
            0.40 * main_ratios
            + 0.25 * profit_degradation
            + 0.20 * winrate_consistency
            + 0.15 * temporal_stability
        )

        return consistency_enhanced

    def calculate_ml_score(self, df):
        """
        Calcula el ML Score usando detección de outliers y análisis estadístico
        """
        print("Calculando ML Score...")

        # Features para análisis
        feature_cols = [
            "Max_DD_Total",
            "CAGR_Average",
            "Sharpe_Average",
            "Stability_IS",
            "Calmar_Ratio_Weighted",
        ]
        features = []

        for col in feature_cols:
            if col in df.columns:
                features.append(df[col].fillna(0))

        if len(features) == 0:
            df["ML_Score"] = 1.0
            return df

        X = np.column_stack(features)

        # Detección de outliers con Isolation Forest
        iso_forest = IsolationForest(contamination=0.15, random_state=42)
        outlier_scores = iso_forest.fit_predict(X)
        df["Outlier_Factor"] = np.where(outlier_scores == 1, 1.0, 0.8)

        # Análisis estadístico
        df["Normality_Score"] = 1.0
        df["Diversity_Score"] = 1.0
        df["Consistency_Score"] = 1.0

        for i, col in enumerate(feature_cols):
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 3:
                    # Normalidad
                    feature_skew = abs(skew(values))
                    feature_kurt = abs(kurtosis(values))
                    normality = max(0, 1 - (feature_skew + feature_kurt) / 12)
                    df["Normality_Score"] *= normality

                    # Diversidad
                    cv = values.std() / values.mean() if values.mean() != 0 else 0
                    diversity = min(cv / 2.5, 1)
                    df["Diversity_Score"] *= diversity

        # ML Score final
        df["ML_Score"] = (
            0.30 * df["Normality_Score"]
            + 0.25 * df["Diversity_Score"]
            + 0.20 * df["Outlier_Factor"]
            + 0.25 * df["Consistency_Score"]
        )

        return df

    def calculate_penalties(self, row):
        """
        Calcula las penalizaciones dinámicas
        """
        penalty = 1.0

        # Penalización por número de trades
        num_trades = row.get("# of trades", 0)
        if num_trades < 30:
            penalty *= 0.4
        elif num_trades < 50:
            penalty *= 0.6
        elif num_trades < 100:
            penalty *= 0.8
        elif num_trades >= 500:
            penalty *= 1.1  # Bonus

        # Penalización por Profit Factor
        pf_avg = row.get("PF_Average", 1)
        if pf_avg < 1.05:
            penalty *= 0.3
        elif pf_avg < 1.20:
            penalty *= 0.7
        elif pf_avg < 1.40:
            penalty *= 0.9

        # Penalización por Calmar Ratio
        calmar = row.get("Calmar_Ratio_Weighted", 0)
        if calmar < 1.0:
            penalty *= 0.5
        elif calmar < 1.5:
            penalty *= 0.8
        elif calmar >= 4.0:
            penalty *= 1.1  # Bonus

        # Penalizaciones específicas por régimen
        regime = row.get("Market_Regime", "sideways")
        max_dd = row.get("Max_DD_Total", 0)
        cagr = row.get("CAGR_Average", 0)

        if regime in ["crisis", "bear"]:
            if max_dd > 3.0:
                penalty *= 0.4
            elif max_dd > 2.0:
                penalty *= 0.7
            elif max_dd > 1.0:
                penalty *= 0.9
        elif regime == "bull":
            if cagr < 3:
                penalty *= 0.8
            elif cagr < 8:
                penalty *= 0.95

        # Bonus para bear market con buen control de riesgo
        if regime == "bear":
            if max_dd < 0.5:
                penalty *= 1.2
            elif max_dd < 1.0:
                penalty *= 1.1

        return penalty

    def intelligent_sigmoid(self, score, regime):
        """
        Normalización sigmoidea inteligente basada en régimen
        """
        params = self.sigmoid_params.get(regime, self.sigmoid_params["sideways"])

        normalized = params["scale"] / (
            1 + np.exp(-params["steepness"] * (score - params["midpoint"]))
        )

        return max(0, min(normalized, 10.0))

    def calculate_factor_k_elite(self, df):
        """
        Calcula el Factor K Elite 9.6 para todas las estrategias
        """
        print("Calculando Factor K Elite 9.6...")

        # Calcular percentiles para normalización adaptativa
        percentiles = {}
        key_metrics = [
            "Sharpe_Average",
            "CAGR_Average",
            "Calmar_Ratio_Weighted",
            "Max_DD_Total",
            "Net_Profit_Total",
            "Stability_IS",
            "Ret_DD_Ratio_Average",
        ]

        for metric in key_metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                percentiles[metric] = {
                    20: np.percentile(values, 20),
                    35: np.percentile(values, 35),
                    50: np.percentile(values, 50),
                    65: np.percentile(values, 65),
                    75: np.percentile(values, 75),
                    85: np.percentile(values, 85),
                    95: np.percentile(values, 95),
                }

        # Calcular componentes para cada estrategia
        factor_k_scores = []

        for idx, row in df.iterrows():
            regime = row.get("Market_Regime", "sideways")
            weights = self.regime_weights.get(regime, self.regime_weights["sideways"])

            # Calcular componentes
            stability = self.calculate_stability_enhanced(row, percentiles)
            growth = self.calculate_growth_enhanced(row, percentiles)
            efficiency = self.calculate_efficiency_enhanced(row, percentiles)
            consistency = self.calculate_consistency_enhanced(row, percentiles)

            # Score base
            base_score = (
                weights["stability"] * stability
                + weights["growth"] * growth
                + weights["efficiency"] * efficiency
                + weights["consistency"] * consistency
            )

            # Aplicar ML Score
            ml_score = row.get("ML_Score", 1.0)

            # Aplicar penalizaciones
            penalty = self.calculate_penalties(row)

            # Score con ajustes
            adjusted_score = base_score * ml_score * penalty

            # Normalización sigmoidea inteligente
            final_score = self.intelligent_sigmoid(adjusted_score, regime)

            factor_k_scores.append(final_score)

        df["Factor_K_Elite_96"] = factor_k_scores

        return df

    def assign_rank_categories(self, df):
        """
        Asigna categorías de ranking basadas en Factor K Elite 9.6
        """
        categories = []

        for score in df["Factor_K_Elite_96"]:
            category = "Very Poor"
            for cat_name, cat_info in self.rank_categories.items():
                if cat_info["min"] <= score <= cat_info["max"]:
                    category = cat_name
                    break
            categories.append(category)

        df["Rank_Category"] = categories

        # Calcular percentiles
        df["Percentile"] = df["Factor_K_Elite_96"].rank(pct=True) * 100

        return df

    def process_strategies(self, input_file, output_file="factor_k_elite_results.csv"):
        """
        Proceso completo: carga datos, calcula Factor K Elite 9.6 y guarda resultados
        """
        print(f"Procesando archivo: {input_file}")
        print("=" * 60)

        # Cargar datos
        try:
            # Intentar diferentes separadores
            try:
                df = pd.read_csv(input_file, sep=";", decimal=".")
            except:
                df = pd.read_csv(input_file)
        except Exception as e:
            print(f"Error al cargar archivo: {e}")
            return None

        print(f"Estrategias cargadas: {len(df)}")

        # 1. Calcular métricas derivadas
        df = self.calculate_derived_metrics(df)

        # 2. Detectar régimen de mercado
        df = self.detect_market_regime(df)

        # 3. Calcular ML Score
        df = self.calculate_ml_score(df)

        # 4. Calcular Factor K Elite 9.6
        df = self.calculate_factor_k_elite(df)

        # 5. Asignar categorías y ranking
        df = self.assign_rank_categories(df)

        # 6. Ordenar por Factor K descendente
        df = df.sort_values("Factor_K_Elite_96", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)

        # 7. Seleccionar columnas para output
        output_columns = [
            "Rank",
            "Strategy Name",
            "Factor_K_Elite_96",
            "Rank_Category",
            "Percentile",
            "Market_Regime",
            "Net_Profit_Total",
            "Calmar_Ratio_Weighted",
            "Sharpe_Average",
            "Max_DD_Total",
            "# of trades",
            "CAGR_Average",
        ]

        # Verificar qué columnas existen
        output_columns = [col for col in output_columns if col in df.columns]

        # Crear DataFrame de salida
        output_df = df[output_columns].copy()

        # Formatear valores
        if "Factor_K_Elite_96" in output_df.columns:
            output_df["Factor_K_Elite_96"] = output_df["Factor_K_Elite_96"].round(3)
        if "Percentile" in output_df.columns:
            output_df["Percentile"] = output_df["Percentile"].round(1)
        if "Calmar_Ratio_Weighted" in output_df.columns:
            output_df["Calmar_Ratio_Weighted"] = output_df[
                "Calmar_Ratio_Weighted"
            ].round(3)
        if "Sharpe_Average" in output_df.columns:
            output_df["Sharpe_Average"] = output_df["Sharpe_Average"].round(3)
        if "Max_DD_Total" in output_df.columns:
            output_df["Max_DD_Total"] = output_df["Max_DD_Total"].round(3)
        if "CAGR_Average" in output_df.columns:
            output_df["CAGR_Average"] = output_df["CAGR_Average"].round(3)

        # Guardar resultados
        output_df.to_csv(output_file, index=False)
        print(f"\nResultados guardados en: {output_file}")

        # Mostrar resumen
        print("\nResumen de resultados:")
        print("-" * 40)
        print(f"Factor K promedio: {output_df['Factor_K_Elite_96'].mean():.2f}")
        print(f"Factor K máximo: {output_df['Factor_K_Elite_96'].max():.2f}")
        print(f"Factor K mínimo: {output_df['Factor_K_Elite_96'].min():.2f}")

        print("\nDistribución por categorías:")
        category_counts = output_df["Rank_Category"].value_counts()
        for category in [
            "Elite",
            "Excellent",
            "Very Good",
            "Good",
            "Average",
            "Below Average",
            "Poor",
            "Very Poor",
        ]:
            if category in category_counts.index:
                print(f"{category}: {category_counts[category]} estrategias")

        print("\nTop 10 estrategias:")
        print(
            output_df.head(10)[
                ["Rank", "Strategy Name", "Factor_K_Elite_96", "Rank_Category"]
            ]
        )

        return output_df


# Función principal para ejecutar el sistema
def main():
    """
    Función principal para ejecutar Factor K Elite 9.6
    """
    import sys

    # Verificar argumentos
    if len(sys.argv) < 2:
        print(
            "Uso: python factor_k_elite_96.py <archivo_entrada.csv> [archivo_salida.csv]"
        )
        print("\nEjemplo: python factor_k_elite_96.py strategies.csv results.csv")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "factor_k_elite_results.csv"

    # Crear instancia y procesar
    fk96 = FactorKElite96()
    results = fk96.process_strategies(input_file, output_file)

    if results is not None:
        print("\n¡Proceso completado exitosamente!")


if __name__ == "__main__":
    main()
