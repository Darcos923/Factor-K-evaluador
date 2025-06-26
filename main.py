#!/usr/bin/env python3
"""
Script principal para ejecutar Factor K Elite 9.6
Con tu CSV actualizado de Strategy Quant
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Importar el procesador simplificado
from factor_k_96_wrapper import FactorKElite96Simplified


def create_visualizations(results_df, output_dir="factor_k_output"):
    """
    Crea visualizaciones de los resultados
    """
    print("\nGenerando visualizaciones...")

    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configurar estilo
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. Distribución del Factor K
    plt.figure(figsize=(10, 6))
    plt.hist(
        results_df["Factor_K_Elite_96"],
        bins=30,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
    )
    plt.axvline(
        results_df["Factor_K_Elite_96"].mean(),
        color="red",
        linestyle="--",
        label=f'Media: {results_df["Factor_K_Elite_96"].mean():.2f}',
    )
    plt.axvline(
        results_df["Factor_K_Elite_96"].median(),
        color="green",
        linestyle="--",
        label=f'Mediana: {results_df["Factor_K_Elite_96"].median():.2f}',
    )
    plt.xlabel("Factor K Elite 9.6")
    plt.ylabel("Frecuencia")
    plt.title("Distribución del Factor K Elite 9.6")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "factor_k_distribution.png"), dpi=300)
    plt.close()

    # 2. Factor K vs Calmar Ratio
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        results_df["Calmar_Ratio_Weighted"],
        results_df["Factor_K_Elite_96"],
        c=results_df["Percentile"],
        cmap="viridis",
        alpha=0.6,
        s=50,
    )
    plt.colorbar(scatter, label="Percentil")
    plt.xlabel("Calmar Ratio Weighted")
    plt.ylabel("Factor K Elite 9.6")
    plt.title("Factor K vs Calmar Ratio")

    # Añadir línea de tendencia
    z = np.polyfit(
        results_df["Calmar_Ratio_Weighted"], results_df["Factor_K_Elite_96"], 1
    )
    p = np.poly1d(z)
    plt.plot(
        results_df["Calmar_Ratio_Weighted"].sort_values(),
        p(results_df["Calmar_Ratio_Weighted"].sort_values()),
        "r--",
        alpha=0.8,
        label=f"Tendencia: y={z[0]:.2f}x+{z[1]:.2f}",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "factor_k_vs_calmar.png"), dpi=300)
    plt.close()

    # 3. Distribución por Categorías
    plt.figure(figsize=(10, 6))
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
    category_counts = results_df["Rank_Category"].value_counts()
    category_counts = category_counts.reindex(category_order, fill_value=0)

    colors = [
        "gold",
        "silver",
        "#CD7F32",
        "green",
        "yellow",
        "orange",
        "red",
        "darkred",
    ]
    bars = plt.bar(
        range(len(category_counts)),
        category_counts.values,
        color=colors,
        edgecolor="black",
    )

    # Añadir valores en las barras
    for i, (idx, value) in enumerate(category_counts.items()):
        if value > 0:
            plt.text(i, value + 0.5, str(value), ha="center", va="bottom")

    plt.xticks(range(len(category_counts)), category_counts.index, rotation=45)
    plt.xlabel("Categoría")
    plt.ylabel("Número de Estrategias")
    plt.title("Distribución de Estrategias por Categoría")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_distribution.png"), dpi=300)
    plt.close()

    # 4. Heatmap de correlaciones
    plt.figure(figsize=(10, 8))
    numeric_cols = [
        "Factor_K_Elite_96",
        "Calmar_Ratio_Weighted",
        "Sharpe_Average",
        "CAGR_Average",
        "Max_DD_Total",
        "PF_Average",
        "WinRate_Average",
    ]
    available_cols = [col for col in numeric_cols if col in results_df.columns]

    if len(available_cols) > 2:
        corr_matrix = results_df[available_cols].corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Matriz de Correlación - Métricas Principales")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
        plt.close()

    # 5. Top 15 Estrategias
    plt.figure(figsize=(10, 8))
    top_15 = results_df.head(15)
    y_pos = np.arange(len(top_15))

    bars = plt.barh(
        y_pos, top_15["Factor_K_Elite_96"], color="lightgreen", edgecolor="black"
    )

    # Colorear según categoría
    category_colors = {
        "Elite": "gold",
        "Excellent": "silver",
        "Very Good": "#CD7F32",
        "Good": "green",
    }

    for i, (idx, row) in enumerate(top_15.iterrows()):
        color = category_colors.get(row["Rank_Category"], "lightgreen")
        bars[i].set_color(color)

    plt.yticks(
        y_pos,
        [
            (
                f"{row['Rank']}. {row['Strategy Name'][:30]}..."
                if len(row["Strategy Name"]) > 30
                else f"{row['Rank']}. {row['Strategy Name']}"
            )
            for _, row in top_15.iterrows()
        ],
    )
    plt.xlabel("Factor K Elite 9.6")
    plt.title("Top 15 Estrategias por Factor K Elite 9.6")
    plt.gca().invert_yaxis()

    # Añadir valores en las barras
    for i, (idx, row) in enumerate(top_15.iterrows()):
        plt.text(
            row["Factor_K_Elite_96"] + 0.05,
            i,
            f"{row['Factor_K_Elite_96']:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_15_strategies.png"), dpi=300)
    plt.close()

    print(f"✓ Visualizaciones guardadas en: {output_dir}/")

    # Crear un resumen HTML con todas las visualizaciones
    create_html_report(results_df, output_dir)


def create_html_report(results_df, output_dir):
    """
    Crea un reporte HTML con todas las visualizaciones y estadísticas
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Factor K Elite 9.6 - Reporte</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #333;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-box {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #007bff;
            }}
            .stat-label {{
                font-size: 14px;
                color: #6c757d;
                margin-top: 5px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #007bff;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .elite {{
                background-color: #ffd700;
            }}
            .excellent {{
                background-color: #c0c0c0;
            }}
            .very-good {{
                background-color: #cd7f32;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Factor K Elite 9.6 - Reporte de Análisis</h1>
            <p>Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Estadísticas Generales</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{len(results_df)}</div>
                    <div class="stat-label">Total Estrategias</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results_df['Factor_K_Elite_96'].mean():.3f}</div>
                    <div class="stat-label">Factor K Promedio</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results_df['Factor_K_Elite_96'].max():.3f}</div>
                    <div class="stat-label">Factor K Máximo</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(results_df[results_df['Rank_Category'] == 'Elite'])}</div>
                    <div class="stat-label">Estrategias Elite</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results_df['Calmar_Ratio_Weighted'].mean():.2f}</div>
                    <div class="stat-label">Calmar Promedio</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results_df['Sharpe_Average'].mean():.2f}</div>
                    <div class="stat-label">Sharpe Promedio</div>
                </div>
            </div>
            
            <h2>Distribución del Factor K</h2>
            <img src="factor_k_distribution.png" alt="Distribución Factor K">
            
            <h2>Distribución por Categorías</h2>
            <img src="category_distribution.png" alt="Distribución por Categorías">
            
            <h2>Factor K vs Calmar Ratio</h2>
            <img src="factor_k_vs_calmar.png" alt="Factor K vs Calmar">
            
            <h2>Matriz de Correlación</h2>
            <img src="correlation_matrix.png" alt="Matriz de Correlación">
            
            <h2>Top 15 Estrategias</h2>
            <img src="top_15_strategies.png" alt="Top 15 Estrategias">
            
            <h2>Tabla de Top 20 Estrategias</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Estrategia</th>
                    <th>Factor K</th>
                    <th>Categoría</th>
                    <th>Calmar Ratio</th>
                    <th>Sharpe</th>
                    <th>Max DD %</th>
                </tr>
    """

    # Añadir top 20 estrategias a la tabla
    for _, row in results_df.head(20).iterrows():
        category_class = row["Rank_Category"].lower().replace(" ", "-")
        html_content += f"""
                <tr class="{category_class}">
                    <td>{row['Rank']}</td>
                    <td>{row['Strategy Name']}</td>
                    <td>{row['Factor_K_Elite_96']:.3f}</td>
                    <td>{row['Rank_Category']}</td>
                    <td>{row['Calmar_Ratio_Weighted']:.3f}</td>
                    <td>{row['Sharpe_Average']:.3f}</td>
                    <td>{row['Max_DD_Total']:.2f}%</td>
                </tr>
        """

    html_content += """
            </table>
        </div>
    </body>
    </html>
    """

    # Guardar HTML
    html_file = os.path.join(output_dir, "factor_k_report.html")
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✓ Reporte HTML guardado en: {html_file}")


def main():
    """
    Función principal para ejecutar el análisis completo
    """
    print("\n" + "=" * 80)
    print("FACTOR K ELITE 9.6 ENHANCED - ANÁLISIS COMPLETO")
    print("=" * 80)

    # Verificar argumentos
    if len(sys.argv) < 2:
        print(
            "\nUso: python run_factor_k.py <archivo_csv> [--visualize] [--output-dir <dir>]"
        )
        print("\nOpciones:")
        print("  --visualize     Genera gráficos y reporte HTML")
        print(
            "  --output-dir    Directorio para guardar resultados (default: factor_k_output)"
        )
        print("\nEjemplo:")
        print("  python run_factor_k.py DatabankExport.csv --visualize")
        return

    # Parsear argumentos
    input_file = sys.argv[1]
    visualize = "--visualize" in sys.argv

    # Directorio de salida
    output_dir = "factor_k_output"
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]

    # Crear directorio de salida
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Archivo de salida
    output_file = os.path.join(output_dir, "factor_k_results.csv")

    # Ejecutar el análisis
    print(f"\nProcesando: {input_file}")
    print(f"Directorio de salida: {output_dir}")

    processor = FactorKElite96Simplified()
    results = processor.quick_process(input_file, output_file)

    if results is not None:
        # Generar visualizaciones si se solicita
        if visualize:
            try:
                import numpy as np

                create_visualizations(results, output_dir)
            except ImportError:
                print(
                    "\n⚠️  No se pueden generar visualizaciones. Instala matplotlib y seaborn:"
                )
                print("    pip install matplotlib seaborn")

        # Análisis adicional
        print("\n" + "=" * 80)
        print("ANÁLISIS ADICIONAL")
        print("=" * 80)

        # Análisis de consistencia IS/OOS
        if "Net_Profit_OOS_Proportion" in results.columns:
            print("\nAnálisis de Consistencia IS/OOS:")
            consistency_bins = [0, 0.2, 0.3, 0.4, 0.5, 1.0]
            consistency_labels = ["Muy Baja", "Baja", "Media", "Alta", "Muy Alta"]
            results["Consistency_Level"] = pd.cut(
                results["Net_Profit_OOS_Proportion"],
                bins=consistency_bins,
                labels=consistency_labels,
            )

            consistency_dist = results["Consistency_Level"].value_counts()
            for level in consistency_labels:
                if level in consistency_dist.index:
                    count = consistency_dist[level]
                    print(
                        f"  {level:10}: {count:3} estrategias ({count/len(results)*100:5.1f}%)"
                    )

        # Análisis de diversificación por régimen
        print("\nDiversificación por Régimen de Mercado:")
        regime_quality = (
            results.groupby(["Market_Regime", "Rank_Category"])
            .size()
            .unstack(fill_value=0)
        )

        for regime in regime_quality.index:
            total_regime = regime_quality.loc[regime].sum()
            if total_regime > 0:
                # Check what categories are actually available
                available_categories = regime_quality.columns.tolist()
                high_quality_categories = [
                    cat
                    for cat in ["Elite", "Excellent", "Very Good"]
                    if cat in available_categories
                ]

                if high_quality_categories:
                    high_quality = regime_quality.loc[regime][
                        high_quality_categories
                    ].sum()
                    print(
                        f"  {regime:10}: {high_quality}/{total_regime} de alta calidad ({high_quality/total_regime*100:.1f}%)"
                    )
                else:
                    # If no high quality categories exist, show all categories
                    print(f"  {regime:10}: 0/{total_regime} de alta calidad (0.0%)")
                    print(f"    Available categories: {available_categories}")

        # Recomendaciones de portfolio
        print("\nRECOMENDACIONES DE PORTFOLIO:")

        # Seleccionar mejores por régimen
        portfolio_candidates = []
        for regime in results["Market_Regime"].unique():
            regime_strategies = results[results["Market_Regime"] == regime]
            top_regime = regime_strategies.nlargest(3, "Factor_K_Elite_96")
            portfolio_candidates.extend(top_regime.index.tolist())

        portfolio_df = results.loc[portfolio_candidates].drop_duplicates()

        print(f"\nPortfolio diversificado sugerido ({len(portfolio_df)} estrategias):")
        for _, row in portfolio_df.head(10).iterrows():
            print(
                f"  • {row['Strategy Name'][:40]:40} "
                f"FK: {row['Factor_K_Elite_96']:6.3f} "
                f"[{row['Market_Regime']}]"
            )

        # Métricas del portfolio sugerido
        if len(portfolio_df) > 0:
            print(f"\nMétricas del portfolio sugerido:")
            print(
                f"  - Factor K promedio: {portfolio_df['Factor_K_Elite_96'].mean():.3f}"
            )
            print(
                f"  - Calmar promedio: {portfolio_df['Calmar_Ratio_Weighted'].mean():.3f}"
            )
            print(f"  - Sharpe promedio: {portfolio_df['Sharpe_Average'].mean():.3f}")
            print(f"  - Max DD promedio: {portfolio_df['Max_DD_Total'].mean():.2f}%")

        # Guardar portfolio sugerido
        portfolio_file = os.path.join(output_dir, "portfolio_suggested.csv")
        portfolio_df.to_csv(portfolio_file, index=False, sep=";", decimal=".")
        print(f"\n✓ Portfolio sugerido guardado en: {portfolio_file}")

        # Resumen final
        print("\n" + "=" * 80)
        print("RESUMEN DE ARCHIVOS GENERADOS:")
        print("=" * 80)

        files_generated = [
            f"Resultados principales: {output_file}",
            f"Reporte detallado: {output_file.replace('.csv', '_report.txt')}",
            f"Portfolio sugerido: {portfolio_file}",
        ]

        if visualize:
            files_generated.extend(
                [
                    f"Visualizaciones: {output_dir}/*.png",
                    f"Reporte HTML: {os.path.join(output_dir, 'factor_k_report.html')}",
                ]
            )

        for file_desc in files_generated:
            print(f"  • {file_desc}")

        print("\n✅ ¡Análisis completado exitosamente!")

    else:
        print("\n❌ Error en el procesamiento. Verifica el archivo de entrada.")


if __name__ == "__main__":
    main()
