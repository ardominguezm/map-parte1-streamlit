import builtins as _b
# Restaurar builtins por si fueron "pisados" en el entorno (evita TypeError: 'str' object is not callable)
str=_b.str; list=_b.list; dict=_b.dict; set=_b.set

import streamlit as st
import pandas as pd
import datetime as _dt
import io

# (Opcional) Este guard evita ejecutar el script en "bare mode" (p.ej., desde un notebook).
# En Streamlit Cloud normalmente no es necesario; si te causa problemas, coméntalo.
def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

if not _in_streamlit():
    print("Ejecuta esta app con: streamlit run app.py")
    # raise SystemExit  # <-- comenta esta línea si vas a ejecutarlo en entornos no-Streamlit

st.set_page_config(page_title="MAP — 2024 Q1", layout="wide")

@st.cache_data
def load_data():
    # Serie histórica
    serie = pd.read_csv("outputs_parte1/serie_nacional_mensual.csv")
    s_date_col = next((c for c in serie.columns if str(c).lower() in ("date","fecha","index","unnamed: 0")), None)
    if s_date_col is None:
        raise ValueError("No se encontró columna de fecha en serie_nacional_mensual.csv")
    serie["date"] = pd.to_datetime(serie[s_date_col])
    serie = serie.set_index("date").sort_index()
    s_val_col = next((c for c in serie.columns if c != s_date_col), None)
    if s_val_col is None:
        serie.columns = ["hist"]
    else:
        serie = serie.rename(columns={s_val_col: "hist"})
    serie = serie[["hist"]]

    # Forecast Q1
    fc = pd.read_csv("outputs_parte1/forecast_nacional_Q1_2024.csv")
    f_date_col = next((c for c in fc.columns if str(c).lower() in ("date","fecha","index","unnamed: 0")), None)
    if f_date_col is None:
        raise ValueError("No se encontró columna de fecha en forecast_nacional_Q1_2024.csv")
    fc["date"] = pd.to_datetime(fc[f_date_col])
    fc = fc.set_index("date").sort_index()
    if "forecast" not in fc.columns:
        num_cols = [c for c in fc.columns if pd.api.types.is_numeric_dtype(fc[c])]
        if not num_cols:
            raise ValueError("No se encontró columna numérica para 'forecast' en el CSV de forecast.")
        fc = fc.rename(columns={num_cols[0]: "forecast"})
    for col in ("lower_95","upper_95"):
        if col not in fc.columns:
            fc[col] = pd.NA

    # Departamentos Q1
    dept = pd.read_csv("outputs_parte1/forecast_depto_Q1_2024.csv")
    if "departamento" in dept.columns:
        dept = dept.rename(columns={"departamento":"Departamento"})
    if "pred_Q1_2024" in dept.columns:
        dept = dept.rename(columns={"pred_Q1_2024":"Pred_Q1"})
    if not {"Departamento","Pred_Q1"}.issubset(dept.columns):
        raise ValueError("El CSV de departamentos debe tener columnas 'Departamento' y 'Pred_Q1'.")

    return serie, fc, dept

# ==== Carga de datos
serie, fc, dept_q1 = load_data()

# ==== Título
st.title("Víctimas por Minas Antipersonal — Pronóstico 2024-Q1")

# ==== Pestañas (único lugar donde se muestran las visualizaciones)
tab_viz, tab_dl = st.tabs(["📊 Visualizaciones", "📥 Descargas"])

with tab_viz:
    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("Histórico y pronóstico (nacional)")
        to_plot = pd.concat([serie["hist"], fc["forecast"].rename("forecast")], axis=1).sort_index()
        to_plot.index.name = "date"
        st.line_chart(to_plot)
        st.caption("Nota: IC≈95% calculado a partir del error de validación del modelo seleccionado.")

    with col2:
        st.subheader("Top-10 departamentos (Q1-2024, predicho)")
        top10 = dept_q1.sort_values("Pred_Q1", ascending=False).head(10).copy()
        top10 =
