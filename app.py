#!/usr/bin/env python3
"""
Factor K Elite 9.6 - Aplicación Streamlit Interactiva
Visualización completa e interactiva de resultados
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import base64
from io import BytesIO

# Importar las clases del sistema
from factor_k_96_wrapper import FactorKElite96Simplified

# Configuración de la página
st.set_page_config(
    page_title="Factor K Elite 9.6 - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .category-elite {
        background-color: #ffd700;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }
    .category-excellent {
        background-color: #c0c0c0;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }
    .category-very-good {
        background-color: #cd7f32;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ========== FUNCIONES AUXILIARES ==========


def ensure_proper_dtypes(df):
    """Asegura que los tipos de datos sean compatibles con Streamlit/Arrow"""
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                # Intentar convertir a numérico si es posible
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except:
                pass
    return df


@st.cache_data
def process_file(uploaded_file):
    """Procesa el archivo CSV y calcula Factor K Elite 9.6"""
    # Guardar temporalmente el archivo
    temp_file = "temp_upload.csv"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Procesar con Factor K Elite 9.6
    processor = FactorKElite96Simplified()
    results = processor.quick_process(temp_file, "temp_results.csv")

    # Limpiar archivos temporales
    if os.path.exists(temp_file):
        os.remove(temp_file)
    if os.path.exists("temp_results.csv"):
        os.remove("temp_results.csv")

    return results


def get_category_color(category):
    """Retorna el color para cada categoría"""
    colors = {
        "Elite": "#FFD700",
        "Excellent": "#C0C0C0",
        "Very Good": "#CD7F32",
        "Good": "#90EE90",
        "Average": "#FFFF00",
        "Below Average": "#FFA500",
        "Poor": "#FF6B6B",
        "Very Poor": "#8B0000",
    }
    return colors.get(category, "#808080")


def create_download_link(df, filename="data.csv"):
    """Crea un link de descarga para un DataFrame"""
    csv = df.to_csv(index=False, sep=";", decimal=".")
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Descargar {filename}</a>'


def generate_text_report(results_df, filtered_df):
    """Genera un reporte completo en formato texto"""
    report = []
    report.append("=" * 80)
    report.append("FACTOR K ELITE 9.6 ENHANCED - REPORTE COMPLETO")
    report.append("=" * 80)
    report.append(
        f"\nFecha del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    report.append(f"Total de estrategias analizadas: {len(results_df)}")
    report.append(f"Estrategias después de filtros: {len(filtered_df)}")

    # Estadísticas generales
    report.append("\n\nESTADÍSTICAS GENERALES DEL FACTOR K")
    report.append("-" * 40)
    report.append(f"Media: {results_df['Factor_K_Elite_96'].mean():.3f}")
    report.append(f"Mediana: {results_df['Factor_K_Elite_96'].median():.3f}")
    report.append(f"Desviación estándar: {results_df['Factor_K_Elite_96'].std():.3f}")
    report.append(f"Mínimo: {results_df['Factor_K_Elite_96'].min():.3f}")
    report.append(f"Máximo: {results_df['Factor_K_Elite_96'].max():.3f}")

    # Distribución por categorías
    report.append("\n\nDISTRIBUCIÓN POR CATEGORÍAS")
    report.append("-" * 40)
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
        count = len(results_df[results_df["Rank_Category"] == category])
        if count > 0:
            percentage = (count / len(results_df)) * 100
            report.append(f"{category:15}: {count:3} estrategias ({percentage:5.1f}%)")

    # Top 20 estrategias
    report.append("\n\nTOP 20 ESTRATEGIAS")
    report.append("-" * 80)
    report.append(f"{'Rank':>4} {'Estrategia':40} {'Factor K':>10} {'Categoría':15}")
    report.append("-" * 80)

    for _, row in results_df.head(20).iterrows():
        name = (
            row["Strategy Name"][:38] + ".."
            if len(row["Strategy Name"]) > 40
            else row["Strategy Name"]
        )
        report.append(
            f"{row['Rank']:>4} {name:40} {row['Factor_K_Elite_96']:>10.3f} {row['Rank_Category']:15}"
        )

    return "\n".join(report)


def generate_executive_summary(results_df, filtered_df):
    """Genera un resumen ejecutivo"""
    summary = []
    summary.append("FACTOR K ELITE 9.6 - RESUMEN EJECUTIVO")
    summary.append("=" * 50)
    summary.append(f"\nFecha: {datetime.now().strftime('%Y-%m-%d')}")

    # Hallazgos clave
    summary.append("\n\nHALLAZGOS CLAVE:")

    elite_count = len(results_df[results_df["Rank_Category"] == "Elite"])
    excellent_count = len(results_df[results_df["Rank_Category"] == "Excellent"])
    high_quality = elite_count + excellent_count

    summary.append(
        f"\n✓ {high_quality} estrategias de alta calidad (Elite + Excellent)"
    )
    summary.append(f"✓ Factor K promedio: {results_df['Factor_K_Elite_96'].mean():.2f}")
    summary.append(f"✓ {len(results_df)} estrategias analizadas en total")

    # Recomendaciones
    summary.append("\n\nRECOMENDACIONES:")

    if elite_count > 0:
        summary.append(
            f"\n1. Considerar las {elite_count} estrategias Elite para trading real"
        )

    if high_quality >= 10:
        summary.append(
            "\n2. Crear un portfolio diversificado con las mejores estrategias"
        )
    else:
        summary.append("\n2. Generar más estrategias para ampliar opciones de calidad")

    # Advertencias
    high_dd = len(results_df[results_df["Max_DD_Total"] > 3.0])
    if high_dd > 0:
        summary.append(
            f"\n⚠️  {high_dd} estrategias tienen Max DD > 3% - Evaluar riesgo"
        )

    return "\n".join(summary)


def setup_sidebar_info():
    """Configura información adicional en la barra lateral"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ Información")

        with st.expander("Componentes del Factor K"):
            st.markdown(
                """
            **Factor K Elite 9.6** evalúa:
            
            1. **Estabilidad (32-48%)**
               - Retorno ajustado por riesgo
               - Resistencia al drawdown
               - Calmar Ratio
            
            2. **Crecimiento (8-33%)**
               - CAGR normalizado
               - Consistencia del crecimiento
               
            3. **Eficiencia (20-25%)**
               - Profit Factor
               - Win Rate
               
            4. **Consistencia (15-20%)**
               - Ratios IS/OOS
               - Degradación temporal
            """
            )

        with st.expander("Interpretación"):
            st.markdown(
                """
            **Rangos del Factor K:**
            - 9.2-10.0: Elite 🏆
            - 8.2-9.1: Excellent ⭐
            - 7.2-8.1: Very Good 👍
            - 6.2-7.1: Good ✓
            - 5.2-6.1: Average
            - < 5.2: Por debajo del promedio
            """
            )

        st.markdown("---")
        st.markdown("### 👨‍💻 Desarrollado por")
        st.markdown("**Diego Arcos de las Heras**")
        st.markdown("Factor K Elite 9.6 Enhanced")
        st.markdown("© 2025")
        st.markdown("---")


# ========== FUNCIÓN PRINCIPAL ==========


def main():
    # Header
    st.markdown(
        '<h1 class="main-header">Factor K Elite 9.6 Dashboard</h1>',
        unsafe_allow_html=True,
    )
    # Sidebar
    with st.sidebar:
        # Cargar archivo
        st.markdown("### 📁 Cargar Datos")
        uploaded_file = st.file_uploader(
            "Selecciona tu archivo CSV de Strategy Quant",
            type=["csv"],
            help="El archivo debe contener las columnas IS/OOS de Strategy Quant",
        )

        if uploaded_file is not None:
            st.success(f"✅ Archivo cargado: {uploaded_file.name}")
            st.markdown(f"Tamaño: {uploaded_file.size / 1024:.1f} KB")

        st.markdown("---")

        # Opciones de visualización
        st.markdown("### ⚙️ Opciones")
        show_raw_data = st.checkbox("Mostrar datos crudos", value=False)
        top_n = st.slider("Top N estrategias a mostrar", 5, 50, 20)

        # Filtros
        st.markdown("### 🔍 Filtros")

    # Contenido principal
    if uploaded_file is None:
        # Pantalla de bienvenida
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info(
                "👆 Por favor, carga un archivo CSV de Strategy Quant en el panel lateral"
            )

            st.markdown("### ¿Qué es Factor K Elite 9.6?")
            st.markdown(
                """
            Factor K Elite 9.6 es un sistema avanzado de evaluación de estrategias de trading que combina:
            
            - ✅ **Análisis Temporal IS/OOS** - Evalúa la consistencia entre períodos
            - ✅ **Machine Learning** - Detección automática de outliers y regímenes
            - ✅ **Calmar Ratio Integrado** - Estándar de la industria
            - ✅ **16 Métricas Derivadas** - Cálculo automático completo
            - ✅ **Normalización Inteligente** - Adaptativa por régimen de mercado
            """
            )

            st.markdown("### Categorías de Ranking")
            categories_df = pd.DataFrame(
                {
                    "Categoría": [
                        "Elite",
                        "Excellent",
                        "Very Good",
                        "Good",
                        "Average",
                        "Below Average",
                        "Poor",
                        "Very Poor",
                    ],
                    "Rango Factor K": [
                        "9.2-10.0",
                        "8.2-9.1",
                        "7.2-8.1",
                        "6.2-7.1",
                        "5.2-6.1",
                        "4.2-5.1",
                        "3.2-4.1",
                        "0.0-3.1",
                    ],
                    "Descripción": [
                        "Estrategias excepcionales",
                        "Estrategias sobresalientes",
                        "Estrategias muy buenas",
                        "Estrategias buenas",
                        "Estrategias promedio",
                        "Por debajo del promedio",
                        "Estrategias pobres",
                        "Estrategias muy pobres",
                    ],
                }
            )
            st.table(categories_df)

    else:
        # Procesar archivo
        with st.spinner("Procesando estrategias... esto puede tomar unos momentos"):
            try:
                results_df = process_file(uploaded_file)

                if results_df is not None:
                    st.success("✅ Procesamiento completado exitosamente!")

                    # Aplicar filtros del sidebar
                    with st.sidebar:
                        # Filtro por categoría
                        categories = results_df["Rank_Category"].unique()
                        selected_categories = st.multiselect(
                            "Filtrar por categoría", categories, default=categories
                        )

                        # Filtro por régimen de mercado
                        regimes = results_df["Market_Regime"].unique()
                        selected_regimes = st.multiselect(
                            "Filtrar por régimen", regimes, default=regimes
                        )

                        # Filtro por Factor K
                        min_fk, max_fk = float(
                            results_df["Factor_K_Elite_96"].min()
                        ), float(results_df["Factor_K_Elite_96"].max())
                        fk_range = st.slider(
                            "Rango de Factor K",
                            min_fk,
                            max_fk,
                            (min_fk, max_fk),
                            step=0.1,
                        )

                    # Aplicar filtros
                    filtered_df = results_df[
                        (results_df["Rank_Category"].isin(selected_categories))
                        & (results_df["Market_Regime"].isin(selected_regimes))
                        & (results_df["Factor_K_Elite_96"] >= fk_range[0])
                        & (results_df["Factor_K_Elite_96"] <= fk_range[1])
                    ]

                    # Tabs principales
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                        [
                            "📊 Resumen General",
                            "📈 Análisis de Factor K",
                            "🎯 Top Estrategias",
                            "🔄 Consistencia IS/OOS",
                            "💼 Portfolio Sugerido",
                            "📥 Descargas",
                        ]
                    )

                    # ========== TAB 1: RESUMEN GENERAL ==========
                    with tab1:
                        # Métricas principales
                        st.markdown("### 📊 Métricas Principales")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Total Estrategias",
                                len(filtered_df),
                                (
                                    f"{len(filtered_df) - len(results_df)} filtradas"
                                    if len(filtered_df) < len(results_df)
                                    else None
                                ),
                            )

                        with col2:
                            st.metric(
                                "Factor K Promedio",
                                f"{filtered_df['Factor_K_Elite_96'].mean():.3f}",
                                f"Max: {filtered_df['Factor_K_Elite_96'].max():.3f}",
                            )

                        with col3:
                            elite_count = len(
                                filtered_df[filtered_df["Rank_Category"] == "Elite"]
                            )
                            st.metric(
                                "Estrategias Elite",
                                elite_count,
                                (
                                    f"{elite_count/len(filtered_df)*100:.1f}%"
                                    if len(filtered_df) > 0
                                    else "0%"
                                ),
                            )

                        with col4:
                            st.metric(
                                "Calmar Promedio",
                                f"{filtered_df['Calmar_Ratio_Weighted'].mean():.2f}",
                                f"Max: {filtered_df['Calmar_Ratio_Weighted'].max():.2f}",
                            )

                        # Distribución por categorías
                        st.markdown("### 🎨 Distribución por Categorías")

                        category_counts = filtered_df["Rank_Category"].value_counts()
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
                        category_counts = category_counts.reindex(
                            category_order, fill_value=0
                        )

                        fig_categories = go.Figure(
                            data=[
                                go.Bar(
                                    x=category_counts.index,
                                    y=category_counts.values,
                                    marker_color=[
                                        get_category_color(cat)
                                        for cat in category_counts.index
                                    ],
                                    text=category_counts.values,
                                    textposition="auto",
                                )
                            ]
                        )
                        fig_categories.update_layout(
                            title="Distribución de Estrategias por Categoría",
                            xaxis_title="Categoría",
                            yaxis_title="Número de Estrategias",
                            showlegend=False,
                            height=400,
                        )
                        st.plotly_chart(fig_categories, use_container_width=True)

                        # Distribución por régimen de mercado
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### 🌍 Distribución por Régimen de Mercado")
                            regime_counts = filtered_df["Market_Regime"].value_counts()

                            fig_regime = go.Figure(
                                data=[
                                    go.Pie(
                                        labels=regime_counts.index,
                                        values=regime_counts.values,
                                        hole=0.3,
                                    )
                                ]
                            )
                            fig_regime.update_layout(height=300)
                            st.plotly_chart(fig_regime, use_container_width=True)

                        with col2:
                            st.markdown("### 📊 Factor K por Régimen")
                            regime_stats = filtered_df.groupby("Market_Regime")[
                                "Factor_K_Elite_96"
                            ].agg(["mean", "std", "count"])

                            fig_regime_fk = go.Figure(
                                data=[
                                    go.Bar(
                                        x=regime_stats.index,
                                        y=regime_stats["mean"],
                                        error_y=dict(
                                            type="data", array=regime_stats["std"]
                                        ),
                                        text=[
                                            f"{val:.2f}" for val in regime_stats["mean"]
                                        ],
                                        textposition="auto",
                                    )
                                ]
                            )
                            fig_regime_fk.update_layout(
                                yaxis_title="Factor K Promedio", height=300
                            )
                            st.plotly_chart(fig_regime_fk, use_container_width=True)

                    # ========== TAB 2: ANÁLISIS DE FACTOR K ==========
                    with tab2:
                        st.markdown("### 📈 Análisis Detallado del Factor K")

                        # Histograma de Factor K
                        fig_hist = px.histogram(
                            filtered_df,
                            x="Factor_K_Elite_96",
                            nbins=30,
                            title="Distribución del Factor K Elite 9.6",
                            color="Rank_Category",
                            color_discrete_map={
                                cat: get_category_color(cat) for cat in category_order
                            },
                        )
                        fig_hist.update_layout(height=400)
                        fig_hist.add_vline(
                            x=filtered_df["Factor_K_Elite_96"].mean(),
                            line_dash="dash",
                            line_color="white",
                            annotation_text=f"Media: {filtered_df['Factor_K_Elite_96'].mean():.2f}",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # Scatter plots
                        col1, col2 = st.columns(2)
                        with col1:
                            # Factor K vs Calmar Ratio
                            fig_scatter1 = px.scatter(
                                filtered_df,
                                x="Calmar_Ratio_Weighted",
                                y="Factor_K_Elite_96",
                                color="Percentile",
                                size="Net_Profit_Total",
                                hover_data=["Strategy Name", "Rank_Category"],
                                title="Factor K vs Calmar Ratio",
                                color_continuous_scale="viridis",
                            )
                            fig_scatter1.update_layout(height=400)
                            st.plotly_chart(fig_scatter1, use_container_width=True)

                            fig_scatter3 = px.scatter(
                                filtered_df,
                                x="Max_DD_Total",
                                y="Factor_K_Elite_96",
                                color="Percentile",
                                size="Net_Profit_Total",
                                hover_data=["Strategy Name", "Rank_Category"],
                                title="Factor K vs Max DD Total",
                                color_continuous_scale="burgyl",
                            )
                            fig_scatter3.update_layout(height=400)
                            st.plotly_chart(fig_scatter3, use_container_width=True)

                        with col2:
                            # Factor K vs Sharpe
                            fig_scatter2 = px.scatter(
                                filtered_df,
                                x="PF_Consistency",
                                y="Factor_K_Elite_96",
                                color="Market_Regime",
                                size="Net_Profit_Total",
                                hover_data=["Strategy Name", "Rank_Category"],
                                title="Factor K vs PF Consistency",
                            )
                            fig_scatter2.update_layout(height=400)
                            st.plotly_chart(fig_scatter2, use_container_width=True)

                            fig_scatter4 = px.scatter(
                                filtered_df,
                                x="CAGR_Average",
                                y="Factor_K_Elite_96",
                                color="Market_Regime",
                                size="Net_Profit_Total",
                                hover_data=["Strategy Name", "Rank_Category"],
                                title="Factor K vs CAGR Average",
                            )
                            fig_scatter4.update_layout(height=400)
                            st.plotly_chart(fig_scatter4, use_container_width=True)

                        # Matriz de correlación
                        st.markdown("### 🔗 Matriz de Correlación")

                        numeric_cols = [
                            "Factor_K_Elite_96",
                            "Calmar_Ratio_Weighted",
                            "Sharpe_Average",
                            "CAGR_Average",
                            "Max_DD_Total",
                            "PF_Average",
                            "WinRate_Average",
                        ]
                        available_cols = [
                            col for col in numeric_cols if col in filtered_df.columns
                        ]

                        if len(available_cols) > 2:
                            corr_matrix = filtered_df[available_cols].corr()

                            fig_corr = go.Figure(
                                data=go.Heatmap(
                                    z=corr_matrix.values,
                                    x=corr_matrix.columns,
                                    y=corr_matrix.columns,
                                    colorscale="RdBu",
                                    zmid=0,
                                    text=corr_matrix.values.round(2),
                                    texttemplate="%{text}",
                                    textfont={"size": 10},
                                )
                            )
                            fig_corr.update_layout(
                                title="Matriz de Correlación de Métricas", height=500
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)

                    # ========== TAB 3: TOP ESTRATEGIAS ==========
                    with tab3:
                        st.markdown(f"### 🏆 Top {top_n} Estrategias")

                        # Selector de métrica para ordenar
                        sort_metric = st.selectbox(
                            "Ordenar por:",
                            [
                                "Factor_K_Elite_96",
                                "Calmar_Ratio_Weighted",
                                "Sharpe_Average",
                                "Net_Profit_Total",
                            ],
                            format_func=lambda x: x.replace("_", " ").title(),
                        )

                        # Top estrategias
                        top_strategies = filtered_df.nlargest(top_n, sort_metric)

                        # Gráfico de barras horizontales
                        fig_top = go.Figure()

                        # Añadir barras con colores por categoría
                        for category in category_order:
                            cat_data = top_strategies[
                                top_strategies["Rank_Category"] == category
                            ]
                            if not cat_data.empty:
                                fig_top.add_trace(
                                    go.Bar(
                                        y=[
                                            (
                                                f"{row['Rank']}. {row['Strategy Name'][:30]}..."
                                                if len(row["Strategy Name"]) > 30
                                                else f"{row['Rank']}. {row['Strategy Name']}"
                                            )
                                            for _, row in cat_data.iterrows()
                                        ],
                                        x=cat_data[sort_metric],
                                        name=category,
                                        orientation="h",
                                        marker_color=get_category_color(category),
                                        text=cat_data[sort_metric].round(3),
                                        textposition="auto",
                                    )
                                )

                        fig_top.update_layout(
                            title=f"Top {top_n} Estrategias por {sort_metric.replace('_', ' ').title()}",
                            xaxis_title=sort_metric.replace("_", " ").title(),
                            height=max(400, top_n * 25),
                            showlegend=True,
                            yaxis={"categoryorder": "total ascending"},
                        )
                        st.plotly_chart(fig_top, use_container_width=True)

                        # Tabla detallada
                        st.markdown("### 📋 Tabla Detallada")

                        display_cols = [
                            "Rank",
                            "Strategy Name",
                            "Factor_K_Elite_96",
                            "Rank_Category",
                            "Calmar_Ratio_Weighted",
                            "Sharpe_Average",
                            "Max_DD_Total",
                            "Market_Regime",
                            "# of trades",
                        ]

                        # Formatear tabla
                        top_table = top_strategies[display_cols].copy()
                        top_table["Factor_K_Elite_96"] = top_table[
                            "Factor_K_Elite_96"
                        ].round(3)
                        top_table["Calmar_Ratio_Weighted"] = top_table[
                            "Calmar_Ratio_Weighted"
                        ].round(3)
                        top_table["Sharpe_Average"] = top_table["Sharpe_Average"].round(
                            3
                        )
                        top_table["Max_DD_Total"] = top_table["Max_DD_Total"].round(2)

                        # Asegurar tipos de datos correctos
                        top_table = ensure_proper_dtypes(top_table)

                        st.dataframe(
                            top_table,
                            use_container_width=True,
                            height=min(600, (top_n + 1) * 35),
                        )

                    # ========== TAB 4: CONSISTENCIA IS/OOS ==========
                    with tab4:
                        st.markdown("### 🔄 Análisis de Consistencia IS/OOS")

                        # Consistencia Temporal Thresholds
                        st.markdown("#### 📊 Umbrales de Consistencia Temporal")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.info("**Rango:** 0.0 - 2.0")
                        with col2:
                            st.success("**Óptimo:** ~1.0 (Consistente IS/OOS)")
                        with col3:
                            st.warning("**Alerta:** < 0.6 (Posible sobreajuste)")

                        if "Net_Profit_OOS_Proportion" in filtered_df.columns:
                            # Histograma de proporción OOS
                            fig_oos = px.histogram(
                                filtered_df,
                                x="Net_Profit_OOS_Proportion",
                                nbins=20,
                                title="Distribución de Proporción de Beneficios OOS",
                                labels={"Net_Profit_OOS_Proportion": "Proporción OOS"},
                            )
                            fig_oos.add_vline(
                                x=0.3,
                                line_dash="dash",
                                line_color="white",
                                annotation_text="30% (Ideal)",
                            )
                            fig_oos.update_layout(height=400)
                            st.plotly_chart(fig_oos, use_container_width=True)

                            # Detección de sobreajuste y alertas
                            st.markdown("#### ⚠️ Análisis de Sobreajuste")

                            # Calcular consistencia temporal (usando proporción OOS como proxy)
                            filtered_df["Temporal_Consistency"] = (
                                filtered_df["Net_Profit_OOS_Proportion"] * 2.0
                            )

                            # Detectar estrategias con posible sobreajuste
                            overfitting_strategies = filtered_df[
                                filtered_df["Temporal_Consistency"] < 0.6
                            ]

                            if len(overfitting_strategies) > 0:
                                st.error(
                                    f"🚨 **ALERTA:** {len(overfitting_strategies)} estrategias con posible sobreajuste detectadas (Consistencia < 0.6)"
                                )

                                # Mostrar las estrategias problemáticas
                                overfitting_display = overfitting_strategies[
                                    [
                                        "Strategy Name",
                                        "Temporal_Consistency",
                                        "Factor_K_Elite_96",
                                        "Net_Profit_OOS_Proportion",
                                    ]
                                ].copy()
                                overfitting_display["Temporal_Consistency"] = (
                                    overfitting_display["Temporal_Consistency"].round(3)
                                )
                                overfitting_display["Factor_K_Elite_96"] = (
                                    overfitting_display["Factor_K_Elite_96"].round(3)
                                )
                                overfitting_display["Net_Profit_OOS_Proportion"] = (
                                    overfitting_display["Net_Profit_OOS_Proportion"]
                                    * 100
                                ).round(1)
                                overfitting_display.columns = [
                                    "Estrategia",
                                    "Consistencia Temporal",
                                    "Factor K",
                                    "OOS %",
                                ]

                                st.dataframe(
                                    overfitting_display, use_container_width=True
                                )
                            else:
                                st.success(
                                    "✅ No se detectaron estrategias con sobreajuste significativo"
                                )

                            # Análisis por categorías de consistencia
                            col1, col2 = st.columns(2)

                            with col1:
                                # Definir categorías de consistencia basadas en consistencia temporal
                                consistency_bins = [0, 0.6, 0.8, 1.2, 1.5, 2.0]
                                consistency_labels = [
                                    "Sobreajuste",
                                    "Baja",
                                    "Buena",
                                    "Excelente",
                                    "Muy Alta",
                                ]
                                filtered_df["Consistency_Level"] = pd.cut(
                                    filtered_df["Temporal_Consistency"],
                                    bins=consistency_bins,
                                    labels=consistency_labels,
                                )

                                consistency_counts = filtered_df[
                                    "Consistency_Level"
                                ].value_counts()

                                # Colores para categorías de consistencia
                                consistency_colors = {
                                    "Sobreajuste": "#FF4444",
                                    "Baja": "#FFA500",
                                    "Buena": "#90EE90",
                                    "Excelente": "#32CD32",
                                    "Muy Alta": "#FFD700",
                                }

                                fig_consist = go.Figure(
                                    data=[
                                        go.Pie(
                                            labels=consistency_counts.index,
                                            values=consistency_counts.values,
                                            hole=0.3,
                                            marker=dict(
                                                colors=[
                                                    consistency_colors.get(
                                                        cat, "#808080"
                                                    )
                                                    for cat in consistency_counts.index
                                                ]
                                            ),
                                        )
                                    ]
                                )
                                fig_consist.update_layout(
                                    title="Niveles de Consistencia Temporal", height=350
                                )
                                st.plotly_chart(fig_consist, use_container_width=True)

                            with col2:
                                # Mejores estrategias por consistencia
                                st.markdown("#### 🎯 Estrategias más Consistentes")

                                consistent_strategies = filtered_df.nlargest(
                                    10, "Temporal_Consistency"
                                )

                                consist_display = consistent_strategies[
                                    [
                                        "Strategy Name",
                                        "Temporal_Consistency",
                                        "Factor_K_Elite_96",
                                        "Consistency_Level",
                                    ]
                                ].copy()
                                consist_display["Temporal_Consistency"] = (
                                    consist_display["Temporal_Consistency"].round(3)
                                )
                                consist_display["Factor_K_Elite_96"] = consist_display[
                                    "Factor_K_Elite_96"
                                ].round(3)
                                consist_display.columns = [
                                    "Estrategia",
                                    "Consistencia Temporal",
                                    "Factor K",
                                    "Nivel",
                                ]

                                # Asegurar tipos de datos correctos
                                consist_display["Consistencia Temporal"] = (
                                    consist_display["Consistencia Temporal"].astype(
                                        float
                                    )
                                )
                                consist_display["Factor K"] = consist_display[
                                    "Factor K"
                                ].astype(float)

                                st.dataframe(
                                    consist_display,
                                    use_container_width=True,
                                    height=350,
                                )

                        # Análisis de degradación
                        st.markdown("### 📉 Análisis de Degradación IS/OOS")

                        metrics_comparison = []

                        if "Sharpe_Consistency" in filtered_df.columns:
                            metrics_comparison.append(
                                ("Sharpe Ratio", "Sharpe_Consistency")
                            )
                        if "CAGR_Consistency" in filtered_df.columns:
                            metrics_comparison.append(("CAGR", "CAGR_Consistency"))
                        if "PF_Consistency" in filtered_df.columns:
                            metrics_comparison.append(
                                ("Profit Factor", "PF_Consistency")
                            )

                        if metrics_comparison:
                            fig_degrad = make_subplots(
                                rows=1,
                                cols=len(metrics_comparison),
                                subplot_titles=[m[0] for m in metrics_comparison],
                            )

                            for i, (metric_name, metric_col) in enumerate(
                                metrics_comparison
                            ):
                                fig_degrad.add_trace(
                                    go.Histogram(
                                        x=filtered_df[metric_col],
                                        name=metric_name,
                                        nbinsx=20,
                                    ),
                                    row=1,
                                    col=i + 1,
                                )
                                # Línea de referencia para consistencia perfecta
                                fig_degrad.add_vline(
                                    x=1.0,
                                    line_dash="dash",
                                    line_color="green",
                                    row=1,
                                    col=i + 1,
                                    annotation_text="Consistencia Perfecta",
                                )
                                # Línea de alerta para posible sobreajuste
                                fig_degrad.add_vline(
                                    x=0.6,
                                    line_dash="dot",
                                    line_color="red",
                                    row=1,
                                    col=i + 1,
                                    annotation_text="Alerta",
                                )

                            fig_degrad.update_layout(
                                title="Consistencia de Métricas IS/OOS (1.0 = Sin degradación)",
                                height=400,
                                showlegend=False,
                            )
                            st.plotly_chart(fig_degrad, use_container_width=True)

                    # ========== TAB 5: PORTFOLIO SUGERIDO ==========
                    with tab5:
                        st.markdown("### 💼 Portfolio Diversificado Sugerido")

                        # Estrategia de selección
                        st.info(
                            """
                        **Estrategia de Selección:**
                        - Top 3 estrategias de cada régimen de mercado
                        - Prioridad a estrategias Elite y Excellent
                        - Diversificación por características
                        """
                        )

                        # Seleccionar mejores por régimen
                        portfolio_list = []

                        for regime in filtered_df["Market_Regime"].unique():
                            regime_strategies = filtered_df[
                                filtered_df["Market_Regime"] == regime
                            ]
                            top_regime = regime_strategies.nlargest(
                                3, "Factor_K_Elite_96"
                            )
                            portfolio_list.append(top_regime)

                        if portfolio_list:
                            portfolio_df = pd.concat(portfolio_list).drop_duplicates()
                            portfolio_df = portfolio_df.sort_values(
                                "Factor_K_Elite_96", ascending=False
                            ).reset_index(drop=True)

                            # Métricas del portfolio
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Estrategias en Portfolio", len(portfolio_df))

                            with col2:
                                st.metric(
                                    "Factor K Promedio",
                                    f"{portfolio_df['Factor_K_Elite_96'].mean():.3f}",
                                )

                            with col3:
                                st.metric(
                                    "Calmar Promedio",
                                    f"{portfolio_df['Calmar_Ratio_Weighted'].mean():.2f}",
                                )

                            with col4:
                                st.metric(
                                    "Max DD Promedio",
                                    f"{portfolio_df['Max_DD_Total'].mean():.2f}%",
                                )

                            # Composición del portfolio
                            col1, col2 = st.columns(2)

                            with col1:
                                # Por régimen
                                regime_comp = portfolio_df[
                                    "Market_Regime"
                                ].value_counts()
                                fig_regime_comp = go.Figure(
                                    data=[
                                        go.Pie(
                                            labels=regime_comp.index,
                                            values=regime_comp.values,
                                            hole=0.3,
                                            title="Composición por Régimen",
                                        )
                                    ]
                                )
                                st.plotly_chart(
                                    fig_regime_comp, use_container_width=True
                                )

                            with col2:
                                # Por categoría
                                cat_comp = portfolio_df["Rank_Category"].value_counts()
                                fig_cat_comp = go.Figure(
                                    data=[
                                        go.Pie(
                                            labels=cat_comp.index,
                                            values=cat_comp.values,
                                            hole=0.3,
                                            title="Composición por Categoría",
                                        )
                                    ]
                                )
                                st.plotly_chart(fig_cat_comp, use_container_width=True)

                            # Tabla del portfolio
                            st.markdown("### 📋 Estrategias del Portfolio")

                            portfolio_display = portfolio_df[
                                [
                                    "Strategy Name",
                                    "Factor_K_Elite_96",
                                    "Rank_Category",
                                    "Market_Regime",
                                    "Calmar_Ratio_Weighted",
                                    "Max_DD_Total",
                                ]
                            ].copy()

                            portfolio_display["Factor_K_Elite_96"] = portfolio_display[
                                "Factor_K_Elite_96"
                            ].round(3)
                            portfolio_display["Calmar_Ratio_Weighted"] = (
                                portfolio_display["Calmar_Ratio_Weighted"].round(3)
                            )
                            portfolio_display["Max_DD_Total"] = portfolio_display[
                                "Max_DD_Total"
                            ].round(2)

                            # Asegurar tipos de datos correctos
                            portfolio_display = ensure_proper_dtypes(portfolio_display)

                            st.dataframe(portfolio_display, use_container_width=True)

                            # Análisis de correlación del portfolio
                            if len(portfolio_df) > 3:
                                st.markdown(
                                    "### 🔗 Correlación entre Estrategias del Portfolio"
                                )

                                # Aquí podrías añadir análisis de correlación de returns si tuvieras esos datos
                                st.info(
                                    "Para un análisis completo de correlación, se necesitarían los returns históricos de cada estrategia."
                                )

                    # ========== TAB 6: DESCARGAS ==========
                    with tab6:
                        st.markdown("### 📥 Descargar Resultados")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### 📊 Resultados Principales")

                            # Resultados completos
                            st.markdown(
                                create_download_link(
                                    results_df,
                                    f"factor_k_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                ),
                                unsafe_allow_html=True,
                            )

                            # Top 100 estrategias
                            top_100 = results_df.head(100)
                            st.markdown(
                                create_download_link(
                                    top_100,
                                    f"top_100_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                ),
                                unsafe_allow_html=True,
                            )

                            # Portfolio sugerido
                            if "portfolio_df" in locals():
                                st.markdown(
                                    create_download_link(
                                        portfolio_df,
                                        f"portfolio_suggested_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    ),
                                    unsafe_allow_html=True,
                                )

                        with col2:
                            st.markdown("#### 📄 Reportes")

                            # Generar reporte en texto
                            report_text = generate_text_report(results_df, filtered_df)
                            b64_text = base64.b64encode(report_text.encode()).decode()
                            href_text = f'<a href="data:text/plain;base64,{b64_text}" download="factor_k_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt">📥 Descargar Reporte Completo (TXT)</a>'
                            st.markdown(href_text, unsafe_allow_html=True)

                            # Generar resumen ejecutivo
                            exec_summary = generate_executive_summary(
                                results_df, filtered_df
                            )
                            b64_exec = base64.b64encode(exec_summary.encode()).decode()
                            href_exec = f'<a href="data:text/plain;base64,{b64_exec}" download="executive_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt">📥 Descargar Resumen Ejecutivo</a>'
                            st.markdown(href_exec, unsafe_allow_html=True)

                        # Estadísticas para exportar
                        st.markdown("#### 📈 Estadísticas para Exportar")

                        # Crear DataFrame con estadísticas - todos los valores como strings
                        stats_data = {
                            "Métrica": [
                                "Total Estrategias",
                                "Factor K Promedio",
                                "Factor K Máximo",
                                "Factor K Mínimo",
                                "Estrategias Elite",
                                "Estrategias Excellent",
                                "Estrategias Very Good",
                                "Calmar Ratio Promedio",
                                "Sharpe Ratio Promedio",
                                "Max DD Promedio (%)",
                            ],
                            "Valor": [
                                str(len(results_df)),
                                f"{results_df['Factor_K_Elite_96'].mean():.3f}",
                                f"{results_df['Factor_K_Elite_96'].max():.3f}",
                                f"{results_df['Factor_K_Elite_96'].min():.3f}",
                                str(
                                    len(
                                        results_df[
                                            results_df["Rank_Category"] == "Elite"
                                        ]
                                    )
                                ),
                                str(
                                    len(
                                        results_df[
                                            results_df["Rank_Category"] == "Excellent"
                                        ]
                                    )
                                ),
                                str(
                                    len(
                                        results_df[
                                            results_df["Rank_Category"] == "Very Good"
                                        ]
                                    )
                                ),
                                f"{results_df['Calmar_Ratio_Weighted'].mean():.3f}",
                                f"{results_df['Sharpe_Average'].mean():.3f}",
                                f"{results_df['Max_DD_Total'].mean():.2f}",
                            ],
                        }

                        stats_df = pd.DataFrame(stats_data)
                        # Asegurar que la columna Valor sea tipo string
                        stats_df["Valor"] = stats_df["Valor"].astype(str)
                        st.dataframe(stats_df, use_container_width=True)

                        st.markdown(
                            create_download_link(
                                stats_df,
                                f"statistics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            ),
                            unsafe_allow_html=True,
                        )

                    # Mostrar datos crudos si está seleccionado
                    if show_raw_data:
                        st.markdown("---")
                        st.markdown("### 📊 Datos Crudos")
                        # Asegurar tipos de datos correctos
                        filtered_df_display = ensure_proper_dtypes(filtered_df.copy())
                        st.dataframe(filtered_df_display, use_container_width=True)

                else:
                    st.error(
                        "❌ Error al procesar el archivo. Verifica que tenga el formato correcto."
                    )

            except Exception as e:
                st.error(f"❌ Error durante el procesamiento: {str(e)}")
                st.exception(e)


# ========== EJECUTAR LA APLICACIÓN ==========
if __name__ == "__main__":
    setup_sidebar_info()
    main()
