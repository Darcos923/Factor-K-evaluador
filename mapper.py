#!/usr/bin/env python3
"""
Mapeador de columnas actualizado para el CSV completo de Strategy Quant
con todas las columnas necesarias para Factor K Elite 9.6
"""

import pandas as pd
import numpy as np


class ColumnMapper:
    """
    Mapea las columnas del CSV exportado de Strategy Quant
    a las columnas esperadas por el Factor K Elite 9.6
    """

    def __init__(self):
        # Mapeo de columnas del CSV a las esperadas por el sistema
        # Ahora el CSV tiene todas las columnas necesarias
        self.column_mapping = {
            # Columnas directas
            "Strategy Name": "Strategy Name",
            "# of trades": "# of trades",
            "Avg. Trade": "Avg trade",
            "Win/Loss ratio": "Win/loss ratio",
            "SQN": "SQN",
            "R Expectancy": "R Expectancy",
            "Stagnation": "Stagnation",
            "Avg. Profit Per Month": "Avg profit per month (%)",
            # Métricas IS (In-Sample)
            "Net profit (IS)": "Net profit (IS)",
            "Sharpe Ratio (IS)": "Sharpe (IS)",
            "CAGR (IS)": "CAGR % (IS)",
            "Profit factor (IS)": "Profit factor (IS)",
            "Max DD % (IS)": "Max DD % (IS)",
            "Winning Percent (IS)": "Win rate (IS)",
            "Stability (IS)": "Stability (IS)",
            "CalmarRatio (IS)": "Calmar Ratio (IS)",
            "Ret/DD Ratio (IS)": "Return/DD ratio (IS)",
            # Métricas OOS (Out-of-Sample)
            "Net profit (OOS)": "Net profit (OOS)",
            "Sharpe Ratio (OOS)": "Sharpe (OOS)",
            "CAGR (OOS)": "CAGR % (OOS)",
            "Profit factor (OOS)": "Profit factor (OOS)",
            "Max DD % (OOS)": "Max DD % (OOS)",
            "Winning Percent (OOS)": "Win rate (OOS)",
            "CalmarRatio (OOS)": "Calmar Ratio (OOS)",
            "Ret/DD Ratio (OOS)": "Return/DD ratio (OOS)",
        }

        # Ya no necesitamos calcular columnas faltantes
        # pero aún necesitamos estimar Drawdown % que no está en el CSV
        self.derived_columns = {
            "Drawdown % (IS)": "estimate_from_max_dd",
            "Drawdown % (OOS)": "estimate_from_max_dd",
        }

    def map_columns(self, df):
        """
        Mapea las columnas del DataFrame de entrada
        """
        print("Mapeando columnas del CSV...")
        print("-" * 60)

        # Crear nuevo DataFrame con columnas mapeadas
        mapped_df = pd.DataFrame()

        # Mapear columnas existentes
        mapped_count = 0
        missing_count = 0

        for orig_col, new_col in self.column_mapping.items():
            if orig_col in df.columns:
                mapped_df[new_col] = df[orig_col]
                mapped_count += 1
            else:
                print(f"⚠️  Columna no encontrada: {orig_col}")
                missing_count += 1

        print(f"\n✓ Columnas mapeadas: {mapped_count}")
        if missing_count > 0:
            print(f"⚠️  Columnas faltantes: {missing_count}")

        # Estimar columnas derivadas
        print("\nCalculando columnas derivadas...")

        # Estimar Drawdown % como 80% del Max DD (aproximación conservadora)
        if "Max DD % (IS)" in mapped_df.columns:
            mapped_df["Drawdown % (IS)"] = mapped_df["Max DD % (IS)"] * 0.8
            print("✓ Estimado: Drawdown % (IS) = Max DD % (IS) * 0.8")

        if "Max DD % (OOS)" in mapped_df.columns:
            mapped_df["Drawdown % (OOS)"] = mapped_df["Max DD % (OOS)"] * 0.8
            print("✓ Estimado: Drawdown % (OOS) = Max DD % (OOS) * 0.8")

        return mapped_df

    def validate_data(self, df):
        """
        Valida y limpia los datos
        """
        print("\nValidando y limpiando datos...")
        print("-" * 60)

        # Reemplazar valores infinitos con NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Convertir columnas numéricas
        numeric_columns = df.select_dtypes(include=["object"]).columns
        for col in numeric_columns:
            if col != "Strategy Name":
                try:
                    # Intentar convertir a numérico
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except:
                    pass

        # Validar rangos típicos
        validations = {
            "Sharpe (IS)": (-5, 10),
            "Sharpe (OOS)": (-5, 10),
            "Max DD % (IS)": (0, 100),
            "Max DD % (OOS)": (0, 100),
            "Win rate (IS)": (0, 100),
            "Win rate (OOS)": (0, 100),
            "Profit factor (IS)": (0, 10),
            "Profit factor (OOS)": (0, 10),
            "Calmar Ratio (IS)": (-10, 20),
            "Calmar Ratio (OOS)": (-10, 20),
        }

        for col, (min_val, max_val) in validations.items():
            if col in df.columns:
                # Contar valores fuera de rango
                mask = (df[col] < min_val) | (df[col] > max_val)
                out_of_range = mask.sum()

                if out_of_range > 0:
                    print(
                        f"⚠️  {col}: {out_of_range} valores fuera de rango [{min_val}, {max_val}]"
                    )
                    # Mostrar algunos ejemplos
                    if out_of_range <= 5:
                        problem_values = df.loc[mask, ["Strategy Name", col]]
                        for idx, row in problem_values.iterrows():
                            print(f"    - {row['Strategy Name']}: {row[col]}")

                    # Opción de clipping más conservadora
                    df.loc[mask & (df[col] < min_val), col] = min_val
                    df.loc[mask & (df[col] > max_val), col] = max_val

        # Verificar valores faltantes en columnas críticas
        critical_columns = [
            "Sharpe (IS)",
            "Sharpe (OOS)",
            "Max DD % (IS)",
            "Max DD % (OOS)",
            "Calmar Ratio (IS)",
            "Calmar Ratio (OOS)",
        ]

        for col in critical_columns:
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    print(f"⚠️  {col}: {missing} valores faltantes")

        # Llenar valores faltantes con valores por defecto conservadores
        fill_values = {
            "Sharpe (IS)": 0,
            "Sharpe (OOS)": 0,
            "Calmar Ratio (IS)": 0,
            "Calmar Ratio (OOS)": 0,
            "Win rate (IS)": 50,
            "Win rate (OOS)": 50,
            "Profit factor (IS)": 1,
            "Profit factor (OOS)": 1,
            "Stability (IS)": 0.5,
            "SQN": 0,
            "R Expectancy": 0,
        }

        for col, default_val in fill_values.items():
            if col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(default_val)

        print("\n✓ Validación completada")

        return df

    def prepare_dataframe(self, input_file):
        """
        Prepara el DataFrame completo para el Factor K Elite 9.6
        """
        print(f"\nPreparando datos desde: {input_file}")
        print("=" * 60)

        # Leer el CSV
        try:
            # Intentar con diferentes encodings
            try:
                df = pd.read_csv(input_file, sep=";", decimal=".", encoding="utf-8")
            except:
                df = pd.read_csv(input_file, sep=";", decimal=".", encoding="latin-1")

            print(f"✓ Archivo cargado exitosamente")
            print(f"✓ Estrategias encontradas: {len(df)}")
            print(f"✓ Columnas encontradas: {len(df.columns)}")

        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
            return None

        # Verificar estructura del CSV
        if len(df.columns) == 1:
            # El CSV podría estar mal parseado
            print("\n⚠️  Detectado posible problema de parsing")
            print("Intentando re-parsear con diferentes delimitadores...")

            # Intentar leer la primera línea para detectar el delimitador real
            with open(input_file, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line.count(";") > first_line.count(","):
                    print("✓ Usando ';' como delimitador")
                    df = pd.read_csv(input_file, sep=";", decimal=".")
                else:
                    print("✓ Usando ',' como delimitador")
                    df = pd.read_csv(input_file, sep=",")

        # Mostrar primeras columnas para verificar
        print("\nPrimeras 5 columnas encontradas:")
        for i, col in enumerate(df.columns[:5]):
            print(f"  {i+1}. {col}")

        # Mapear columnas
        df_mapped = self.map_columns(df)

        # Validar y limpiar datos
        df_final = self.validate_data(df_mapped)

        # Resumen final
        print(f"\nResumen final:")
        print(f"- Estrategias procesadas: {len(df_final)}")
        print(f"- Columnas disponibles: {len(df_final.columns)}")
        print(f"- Valores faltantes totales: {df_final.isnull().sum().sum()}")

        # Estadísticas básicas
        if "Calmar Ratio (IS)" in df_final.columns:
            print(f"\nEstadísticas de Calmar Ratio IS:")
            print(f"  - Media: {df_final['Calmar Ratio (IS)'].mean():.2f}")
            print(f"  - Mediana: {df_final['Calmar Ratio (IS)'].median():.2f}")
            print(f"  - Máximo: {df_final['Calmar Ratio (IS)'].max():.2f}")

        return df_final


# Función de utilidad para preparar datos antes de Factor K
def prepare_strategy_quant_data(input_file, output_file="prepared_data.csv"):
    """
    Prepara los datos de Strategy Quant para el Factor K Elite 9.6

    Args:
        input_file: Archivo CSV exportado de Strategy Quant
        output_file: Archivo de salida con datos preparados

    Returns:
        DataFrame preparado
    """
    mapper = ColumnMapper()
    df_prepared = mapper.prepare_dataframe(input_file)

    if df_prepared is not None:
        # Guardar datos preparados
        df_prepared.to_csv(output_file, index=False, sep=";", decimal=".")
        print(f"\n✓ Datos preparados guardados en: {output_file}")
        print("\nAhora puedes usar este archivo con el Factor K Elite 9.6:")
        print(f"python factor_k_elite_96.py {output_file}")

        # Mostrar preview de los datos
        print("\nPreview de las primeras 3 estrategias:")
        preview_cols = [
            "Strategy Name",
            "Calmar Ratio (IS)",
            "Sharpe (IS)",
            "Max DD % (IS)",
        ]
        available_preview_cols = [
            col for col in preview_cols if col in df_prepared.columns
        ]
        if available_preview_cols:
            print(df_prepared[available_preview_cols].head(3))

    return df_prepared


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "prepared_data.csv"
        prepare_strategy_quant_data(input_file, output_file)
    else:
        print("Uso: python column_mapper.py <archivo_entrada.csv> [archivo_salida.csv]")
        print("\nEjemplo: python column_mapper.py DatabankExport.csv")
