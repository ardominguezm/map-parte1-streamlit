import builtins as _b
# Evita errores si algún builtin fue "pisado" (p. ej., str=list=...)
str = _b.str; list = _b.list; dict = _b.dict; set = _b.set

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import datetime as _dt

APP_VERSION = "v1.2-2025-09-12"

# ===========================
#  Configuración de la página
# ===========================
st.set_page_config(
    page_title="MAP — 2024 Q1",
    page_icon="📊",
    layout="wide",
)

# ===========================
#  Personalización / Branding
# ===========================
AUTHOR_NAME  = "Andy Domínguez"
AUTHOR_EMAIL = "ardominguezm@gmail.com"
LOGO_CANDIDATES = ["assets/logo_3is.png"]

def render_header():
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        logo_found = False
        for p in LOGO_CANDIDATES:
            if Path(p).exists():
                try:    
                    st.image(p, use_container_width=True) 
                except TypeError:     # Compatibilidad con versiones antiguas    
                    st.image(p, use_column_width=True)
                logo_found = True
                break
        if not logo_found:
            st.markdown("### 📊")
    with col_title:
        st.markdown("# Víctimas por Minas Antipersonal — Pronóstico 2024-Q1")
        st.markdown(
            f"**Realizado por:** {AUTHOR_NAME} · "
            f"[{AUTHOR_EMAIL}](mailto:{AUTHOR_EMAIL})"
        )

# ===========================
#  Carga de datos (cacheada)
# ===========================
@st.cache_data
def load_data():
    # Serie histórica nacional
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

    # Forecast nacional Q1
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
    for col in ("lower_95", "upper_95"):
        if col not in fc.columns:
            fc[col] = pd.NA

    # Predicción por departamento (Q1)
    dept = pd.read_csv("outputs_parte1/forecast_depto_Q1_2024.csv")
    if "departamento" in dept.columns:
        dept = dept.rename(columns={"departamento":"Departamento"})
    if "pred_Q1_2024" in dept.columns:
        dept = dept.rename(columns={"pred_Q1_2024":"Pred_Q1"})
    if not {"Departamento","Pred_Q1"}.issubset(dept.columns):
        raise ValueError("El CSV de departamentos debe tener columnas 'Departamento' y 'Pred_Q1'.")

    return serie, fc, dept

# ===========================
#  UI
# ===========================
try:
    serie, fc, dept_q1 = load_data()
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# Encabezado con logo + autor
render_header()

# Pestañas
tab_viz, tab_dl = st.tabs(["📊 Visualizaciones", "📥 Descargas"])

with tab_viz:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Histórico y pronóstico (nacional)")
        to_plot = pd.concat([serie["hist"], fc["forecast"].rename("forecast")], axis=1).sort_index()
        to_plot.index.name = "date"
        st.line_chart(to_plot)
        st.caption("Nota: IC≈95% calculado a partir del error de validación del modelo seleccionado.")

    with col2:
        st.subheader("Top-10 departamentos (Q1-2024, predicho)")
        top10 = (
            dept_q1
            .sort_values("Pred_Q1", ascending=False)
            .head(10)
            .copy()
            .set_index("Departamento")["Pred_Q1"]
        )
        st.bar_chart(top10)

with tab_dl:
    st.write("Descarga los datos usados por el dashboard:")
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Serie nacional (CSV)",
        serie.reset_index().rename(columns={"index": "date"}).to_csv(index=False).encode("utf-8"),
        file_name="serie_nacional_mensual.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Forecast Q1-2024 (CSV)",
        fc.reset_index().rename(columns={"index": "date"}).to_csv(index=False).encode("utf-8"),
        file_name="forecast_nacional_Q1_2024.csv",
        mime="text/csv",
    )
    c3.download_button(
        "Departamentos Q1-2024 (CSV)",
        dept_q1.to_csv(index=False).encode("utf-8"),
        file_name="forecast_depto_Q1_2024.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Realizado por: **Andy Domínguez** "
    f"([{AUTHOR_EMAIL}](mailto:{AUTHOR_EMAIL})) · "
    f"Generado: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} · {APP_VERSION}"
)
