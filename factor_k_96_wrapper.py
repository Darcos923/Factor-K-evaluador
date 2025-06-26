#!/usr/bin/env python3
"""
Factor K Elite 9.6 Enhanced - Versión Simplificada
Diseñada para trabajar directamente con el CSV completo de Strategy Quant
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings("ignore")

# Importar las clases base
from factor_k_96 import FactorKElite96
from mapper import ColumnMapper


class FactorKElite96Simplified(FactorKElite96):
    """
    Versión simplificada que trabaja directamente con el CSV completo
    """

    def __init__(self):
        super().__init__()
        self.mapper = ColumnMapper()

    def load_and_prepare_data(self, input_file):
        """
        Carga y prepara los datos del CSV de Strategy Quant
        """
        print(f"Cargando datos desde: {input_file}")

        # Usar el mapper para preparar los datos
        df = self.mapper.prepare_dataframe(input_file)

        if df is None:
            raise ValueError("Error al cargar los datos")

        return df

    def quick_process(self, input_file, output_file="factor_k_results.csv"):
        """
        Proceso rápido y directo para calcular Factor K Elite 9.6
        """
        print("\n" + "=" * 80)
        print("FACTOR K ELITE 9.6 ENHANCED - PROCESO RÁPIDO")
        print("=" * 80)

        try:
            # 1. Cargar y preparar datos
            print("\n[1/5] Cargando y preparando datos...")
            df = self.load_and_prepare_data(input_file)

            # 2. Calcular métricas derivadas
            print("\n[2/5] Calculando métricas derivadas...")
            df = self.calculate_derived_metrics(df)

            # 3. Detectar régimen de mercado
            print("\n[3/5] Detectando regímenes de mercado...")
            df = self.detect_market_regime(df)

            # 4. Calcular ML Score
            print("\n[4/5] Calculando ML Score...")
            df = self.calculate_ml_score(df)

            # 5. Calcular Factor K Elite 9.6
            print("\n[5/5] Calculando Factor K Elite 9.6...")
            df = self.calculate_factor_k_elite(df)

            # 6. Asignar categorías y ranking
            df = self.assign_rank_categories(df)

            # 7. Ordenar y preparar output
            df = df.sort_values("Factor_K_Elite_96", ascending=False).reset_index(
                drop=True
            )
            df["Rank"] = range(1, len(df) + 1)

            # 8. Seleccionar columnas para output
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
                "Stability_IS",
                "PF_Average",
                "WinRate_Average",
                # Añadir columnas para análisis de consistencia IS/OOS
                "Net_Profit_OOS_Proportion",
                "Net_Profit_IS_Proportion",
                "Sharpe_Consistency",
                "CAGR_Consistency",
                "PF_Consistency",
                "WinRate_Consistency",
            ]

            # Solo incluir columnas que existen
            output_columns = [col for col in output_columns if col in df.columns]
            output_df = df[output_columns].copy()

            # Formatear valores numéricos
            numeric_formats = {
                "Factor_K_Elite_96": 3,
                "Percentile": 1,
                "Calmar_Ratio_Weighted": 3,
                "Sharpe_Average": 3,
                "Max_DD_Total": 3,
                "CAGR_Average": 3,
                "Net_Profit_Total": 2,
                "Stability_IS": 4,
                "PF_Average": 3,
                "WinRate_Average": 1,
                # Formatos para columnas de consistencia IS/OOS
                "Net_Profit_OOS_Proportion": 3,
                "Net_Profit_IS_Proportion": 3,
                "Sharpe_Consistency": 3,
                "CAGR_Consistency": 3,
                "PF_Consistency": 3,
                "WinRate_Consistency": 3,
            }

            for col, decimals in numeric_formats.items():
                if col in output_df.columns:
                    output_df[col] = output_df[col].round(decimals)

            # Guardar resultados
            output_df.to_csv(output_file, index=False, sep=";", decimal=".")
            print(f"\n✓ Resultados guardados en: {output_file}")

            # Mostrar resumen
            self.show_summary(output_df)

            # Generar reporte detallado
            self.generate_detailed_report(df, output_df, output_file)

            return output_df

        except Exception as e:
            print(f"\n❌ Error durante el procesamiento: {e}")
            import traceback

            traceback.print_exc()
            return None

    def show_summary(self, df):
        """
        Muestra un resumen de los resultados
        """
        print("\n" + "=" * 80)
        print("RESUMEN DE RESULTADOS")
        print("=" * 80)

        # Estadísticas generales
        print(f"\nEstadísticas del Factor K Elite 9.6:")
        print(f"  • Media: {df['Factor_K_Elite_96'].mean():.3f}")
        print(f"  • Mediana: {df['Factor_K_Elite_96'].median():.3f}")
        print(f"  • Desv. Est.: {df['Factor_K_Elite_96'].std():.3f}")
        print(f"  • Mínimo: {df['Factor_K_Elite_96'].min():.3f}")
        print(f"  • Máximo: {df['Factor_K_Elite_96'].max():.3f}")

        # Distribución por categorías
        print(f"\nDistribución por categorías:")
        category_order = [
            "Elite",
            "Excellent",
            "Very Good",
            "Good",
            "Average",
            "Below Average",
            "Poor",
            "Very Poor",
        ]

        for category in category_order:
            count = len(df[df["Rank_Category"] == category])
            if count > 0:
                percentage = (count / len(df)) * 100
                bar = "█" * int(percentage / 2)
                print(f"  {category:15} {count:3} ({percentage:5.1f}%) {bar}")

        # Top 10
        print(f"\nTOP 10 Estrategias:")
        print("-" * 80)
        print(
            f"{'Rank':>4} {'Estrategia':35} {'Factor K':>10} {'Calmar':>10} {'Sharpe':>10}"
        )
        print("-" * 80)

        for _, row in df.head(10).iterrows():
            name = row["Strategy Name"]
            if len(name) > 33:
                name = name[:30] + "..."

            calmar = row.get("Calmar_Ratio_Weighted", 0)
            sharpe = row.get("Sharpe_Average", 0)

            print(
                f"{row['Rank']:>4} {name:35} {row['Factor_K_Elite_96']:>10.3f} "
                f"{calmar:>10.3f} {sharpe:>10.3f}"
            )

    def generate_detailed_report(self, full_df, output_df, base_filename):
        """
        Genera un reporte detallado en formato texto
        """
        report_filename = base_filename.replace(".csv", "_report.txt")

        with open(report_filename, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("FACTOR K ELITE 9.6 ENHANCED - REPORTE DETALLADO\n")
            f.write("=" * 80 + "\n\n")

            # Información general
            f.write(
                f"Fecha del análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total de estrategias analizadas: {len(output_df)}\n")
            f.write(f"Archivo de entrada: {base_filename}\n\n")

            # Estadísticas del Factor K
            f.write("ESTADÍSTICAS DEL FACTOR K ELITE 9.6\n")
            f.write("-" * 40 + "\n")
            f.write(f"Media: {output_df['Factor_K_Elite_96'].mean():.3f}\n")
            f.write(f"Mediana: {output_df['Factor_K_Elite_96'].median():.3f}\n")
            f.write(
                f"Desviación estándar: {output_df['Factor_K_Elite_96'].std():.3f}\n"
            )
            f.write(f"Mínimo: {output_df['Factor_K_Elite_96'].min():.3f}\n")
            f.write(f"Máximo: {output_df['Factor_K_Elite_96'].max():.3f}\n")
            f.write(
                f"Rango intercuartílico: {output_df['Factor_K_Elite_96'].quantile(0.75) - output_df['Factor_K_Elite_96'].quantile(0.25):.3f}\n\n"
            )

            # Distribución por categorías
            f.write("DISTRIBUCIÓN POR CATEGORÍAS\n")
            f.write("-" * 40 + "\n")
            category_order = [
                "Elite",
                "Excellent",
                "Very Good",
                "Good",
                "Average",
                "Below Average",
                "Poor",
                "Very Poor",
            ]

            for category in category_order:
                subset = output_df[output_df["Rank_Category"] == category]
                if len(subset) > 0:
                    f.write(
                        f"{category:15}: {len(subset):3} estrategias ({len(subset)/len(output_df)*100:5.1f}%)\n"
                    )

            # Análisis por régimen de mercado
            f.write("\nANÁLISIS POR RÉGIMEN DE MERCADO\n")
            f.write("-" * 40 + "\n")
            regime_stats = (
                output_df.groupby("Market_Regime")
                .agg(
                    {
                        "Factor_K_Elite_96": ["mean", "std", "count"],
                        "Calmar_Ratio_Weighted": "mean",
                        "Sharpe_Average": "mean",
                    }
                )
                .round(3)
            )

            for regime in regime_stats.index:
                f.write(f"\n{regime.upper()}:\n")
                f.write(
                    f"  - Estrategias: {int(regime_stats.loc[regime, ('Factor_K_Elite_96', 'count')])}\n"
                )
                f.write(
                    f"  - Factor K promedio: {regime_stats.loc[regime, ('Factor_K_Elite_96', 'mean')]:.3f}\n"
                )
                f.write(
                    f"  - Calmar promedio: {regime_stats.loc[regime, ('Calmar_Ratio_Weighted', 'mean')]:.3f}\n"
                )
                f.write(
                    f"  - Sharpe promedio: {regime_stats.loc[regime, ('Sharpe_Average', 'mean')]:.3f}\n"
                )

            # Top 20 estrategias
            f.write("\n\nTOP 20 ESTRATEGIAS\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Rank':>4} {'Estrategia':40} {'Factor K':>10} {'Categoría':15}\n"
            )
            f.write("-" * 80 + "\n")

            for _, row in output_df.head(20).iterrows():
                name = (
                    row["Strategy Name"][:38] + ".."
                    if len(row["Strategy Name"]) > 40
                    else row["Strategy Name"]
                )
                f.write(
                    f"{row['Rank']:>4} {name:40} {row['Factor_K_Elite_96']:>10.3f} {row['Rank_Category']:15}\n"
                )

            # Estrategias con advertencias
            f.write("\n\nADVERTENCIAS Y CONSIDERACIONES\n")
            f.write("-" * 40 + "\n")

            # Estrategias con alto drawdown
            high_dd = output_df[output_df["Max_DD_Total"] > 3.0]
            if len(high_dd) > 0:
                f.write(f"\n⚠️  Estrategias con Max DD > 3%: {len(high_dd)}\n")
                for _, row in high_dd.head(5).iterrows():
                    f.write(
                        f"   - {row['Strategy Name']}: {row['Max_DD_Total']:.2f}%\n"
                    )

            # Estrategias con pocos trades
            low_trades = output_df[output_df["# of trades"] < 100]
            if len(low_trades) > 0:
                f.write(f"\n⚠️  Estrategias con < 100 trades: {len(low_trades)}\n")
                for _, row in low_trades.head(5).iterrows():
                    f.write(
                        f"   - {row['Strategy Name']}: {row['# of trades']} trades\n"
                    )

            # Recomendaciones finales
            f.write("\n\nRECOMENDACIONES\n")
            f.write("-" * 40 + "\n")

            elite_count = len(output_df[output_df["Rank_Category"] == "Elite"])
            excellent_count = len(output_df[output_df["Rank_Category"] == "Excellent"])

            if elite_count > 0:
                f.write(
                    f"✓ Tienes {elite_count} estrategias Elite - considera estas para trading real\n"
                )

            if excellent_count > 0:
                f.write(
                    f"✓ Tienes {excellent_count} estrategias Excellent - buenas candidatas para portfolio\n"
                )

            if elite_count == 0 and excellent_count == 0:
                f.write("⚠️  No hay estrategias Elite o Excellent. Recomendaciones:\n")
                f.write("   - Revisar los parámetros de generación de estrategias\n")
                f.write("   - Considerar períodos de prueba más largos\n")
                f.write("   - Ajustar los criterios de filtrado inicial\n")

        print(f"\n✓ Reporte detallado guardado en: {report_filename}")


def main():
    """
    Función principal simplificada
    """
    import sys

    print("\n" + "=" * 80)
    print("FACTOR K ELITE 9.6 ENHANCED - VERSIÓN SIMPLIFICADA")
    print("Sistema Avanzado de Evaluación de Estrategias de Trading")
    print("=" * 80)

    if len(sys.argv) < 2:
        print(
            "\nUso: python factor_k_elite_simplified.py <archivo_csv> [archivo_salida]"
        )
        print("\nEjemplo: python factor_k_elite_simplified.py DatabankExport.csv")
        print("\nEl sistema procesará automáticamente tu archivo CSV de Strategy Quant")
        print("y generará:")
        print("  • Archivo de resultados con Factor K Elite 9.6")
        print("  • Reporte detallado en formato texto")
        print("  • Análisis estadístico completo")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "factor_k_results.csv"

    # Crear instancia y procesar
    processor = FactorKElite96Simplified()
    results = processor.quick_process(input_file, output_file)

    if results is not None:
        print("\n" + "=" * 80)
        print("✓ PROCESO COMPLETADO EXITOSAMENTE")
        print("=" * 80)

        # Mostrar archivos generados
        print("\nArchivos generados:")
        print(f"  • Resultados: {output_file}")
        print(f"  • Reporte: {output_file.replace('.csv', '_report.txt')}")

        # Sugerencias basadas en los resultados
        print("\nSUGERENCIAS BASADAS EN EL ANÁLISIS:")

        elite = results[results["Rank_Category"] == "Elite"]
        excellent = results[results["Rank_Category"] == "Excellent"]
        very_good = results[results["Rank_Category"] == "Very Good"]

        total_high_quality = len(elite) + len(excellent) + len(very_good)

        if len(elite) > 0:
            print(f"\n✓ {len(elite)} Estrategias ELITE encontradas:")
            for _, row in elite.head(3).iterrows():
                print(
                    f"   - {row['Strategy Name']} (Factor K: {row['Factor_K_Elite_96']:.3f})"
                )
            if len(elite) > 3:
                print(f"   ... y {len(elite) - 3} más")

        if total_high_quality >= 10:
            print(
                f"\n✓ Tienes {total_high_quality} estrategias de alta calidad (Very Good o mejor)"
            )
            print(
                "  Recomendación: Considera crear un portfolio diversificado con las mejores"
            )
        elif total_high_quality >= 5:
            print(f"\n✓ Tienes {total_high_quality} estrategias de calidad aceptable")
            print("  Recomendación: Analiza en detalle las top 5 para selección final")
        else:
            print(f"\n⚠️  Solo {total_high_quality} estrategias de alta calidad")
            print(
                "  Recomendación: Considera generar más estrategias o ajustar parámetros"
            )

        # Análisis de riesgo
        high_risk = results[results["Max_DD_Total"] > 2.5]
        if len(high_risk) > len(results) * 0.3:
            print("\n⚠️  Más del 30% de las estrategias tienen Max DD > 2.5%")
            print(
                "  Considera ajustar los parámetros de generación para reducir el riesgo"
            )

        # Consistencia IS/OOS
        if "Net_Profit_OOS_Proportion" in results.columns:
            good_consistency = results[results["Net_Profit_OOS_Proportion"] > 0.4]
            if len(good_consistency) < len(results) * 0.2:
                print("\n⚠️  Pocas estrategias muestran buena consistencia IS/OOS")
                print("  Esto podría indicar sobreoptimización")

        print("\n" + "=" * 80)
        print("Para más detalles, consulta el reporte completo generado")
        print("=" * 80)

    else:
        print("\n❌ El proceso no se completó correctamente")
        print("Verifica que el archivo CSV tenga el formato correcto")


if __name__ == "__main__":
    main()
