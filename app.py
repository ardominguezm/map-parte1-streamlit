# app.py
# -----------------------------------------------------------------------------
# MAP — Parte 1 (Streamlit)
# Autor: Andy Domínguez · ardominguezm@gmail.com
# Requiere: streamlit, pandas, numpy, plotly
# Datos esperados (carpeta outputs_parte1/):
#   - serie_nacional_mensual.csv
#   - forecast_nacional_Q1_2024.csv
#   - forecast_depto_Q1_2024.csv
# GeoJSON esperado:
#   - data/colombia_departamentos.geojson  (con propiedades NOMBRE_DPT, DPTO_CODE, name_norm)
# -----------------------------------------------------------------------------

import builtins as _b
# Restaurar builtins por si fueron "pisados" en el entorno
str=_b.str; list=_b.list; dict=_b.dict; set=_b.set

import json, re, unicodedata, os
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ------------------------- Configuración general ------------------------------
st.set_page_config(page_title="MAP — 2024 Q1", layout="wide")

APP_TITLE = "Parte 1- Víctimas por Minas Antipersona — Pronóstico 2024-Q1"
AUTHOR_NAME = "Andy Domínguez"
AUTHOR_EMAIL = "ardominguezm@gmail.com"

# Rutas de datos
OUTPUTS_DIRS = ["outputs_parte1"]  # intenta en este orden (por si cambia el nombre)
GEOJSON_PATH = "data/colombia_departamentos.geojson"
LOGO_PATH = "assets/logo_3is.png"   # súbelo con este nombre/carpeta

# ------------------------- Utilidades -----------------------------------------
def _norm(s: str) -> str:
    """Normaliza nombres para empatar (sin acentos, mayúsculas, espacios compactos)."""
    s = unicodedata.normalize("NFD", str(s)).encode("ascii","ignore").decode().upper()
    s = re.sub(r"[.,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    repl = {
        "BOGOTA D C":"BOGOTA DC","BOGOTA":"BOGOTA DC",
        "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA":"SAN ANDRES Y PROVIDENCIA",
        "SAN ANDRES":"SAN ANDRES Y PROVIDENCIA",
        "VALLE":"VALLE DEL CAUCA",
        "N DE SANTANDER":"NORTE DE SANTANDER","NORTE SANTANDER":"NORTE DE SANTANDER",
    }
    return repl.get(s, s)

def _find_first_existing(*paths) -> Path | None:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

def _read_csv_flexible(filename: str) -> pd.DataFrame:
    """Lee un CSV buscando en OUTPUTS_DIRS y tolerando nombres/encabezados distintos."""
    candidates = [str(Path(d)/filename) for d in OUTPUTS_DIRS]
    p = _find_first_existing(*candidates)
    if p is None:
        raise FileNotFoundError(f"No se encontró {filename} en {OUTPUTS_DIRS}")
    df = pd.read_csv(p)
    return df

# ------------------------- Carga de datos (con cache) -------------------------
@st.cache_data
def load_series_and_forecasts():
    # Serie nacional mensual
    serie = _read_csv_flexible("serie_nacional_mensual.csv")
    s_date_col = next((c for c in serie.columns if c.lower() in ("date","fecha","index","unnamed: 0")), None)
    if s_date_col is None:
        # intenta parsear primera columna como fecha
        s_date_col = serie.columns[0]
    serie["date"] = pd.to_datetime(serie[s_date_col], errors="coerce")
    serie = serie.dropna(subset=["date"]).set_index("date").sort_index()
    # Detectar columna numérica
    s_val_col = next((c for c in serie.columns if pd.api.types.is_numeric_dtype(serie[c])), None)
    if s_val_col is None:
        raise ValueError("No se encontró columna numérica en serie_nacional_mensual.csv")
    serie = serie.rename(columns={s_val_col:"hist"})[["hist"]]

    # Forecast nacional Q1
    fc = _read_csv_flexible("forecast_nacional_Q1_2024.csv")
    f_date_col = next((c for c in fc.columns if c.lower() in ("date","fecha","index","unnamed: 0")), None)
    if f_date_col is None: f_date_col = fc.columns[0]
    fc["date"] = pd.to_datetime(fc[f_date_col], errors="coerce")
    fc = fc.dropna(subset=["date"]).set_index("date").sort_index()

    # asegurar columnas forecast y límites
    if "forecast" not in fc.columns:
        num_cols = [c for c in fc.columns if pd.api.types.is_numeric_dtype(fc[c])]
        fc = fc.rename(columns={num_cols[0]:"forecast"}) if num_cols else fc.assign(forecast=np.nan)
    for col in ("lower_95", "upper_95"):
        if col not in fc.columns:
            fc[col] = np.nan

    # Forecast departamental Q1
    dept = _read_csv_flexible("forecast_depto_Q1_2024.csv")
    # normalizar encabezados
    if "departamento" in dept.columns: dept = dept.rename(columns={"departamento":"Departamento"})
    if "Departamento" not in dept.columns:
        # intentar detectar
        name_col = next((c for c in dept.columns if re.search(r"depar|nombre", c, re.I)), None)
        if name_col: dept = dept.rename(columns={name_col:"Departamento"})
    if "pred_Q1_2024" in dept.columns: dept = dept.rename(columns={"pred_Q1_2024":"Pred_Q1"})
    if "Pred_Q1" not in dept.columns:
        # toma primera numérica como predicción
        num_cols = [c for c in dept.columns if pd.api.types.is_numeric_dtype(dept[c])]
        if num_cols:
            dept = dept.rename(columns={num_cols[0]:"Pred_Q1"})
        else:
            raise ValueError("No se encontró columna numérica de predicción para departamentos.")

    # Dataset histórico 12m por departamento (opcional)
    hist12 = None
    try:
        hist12 = _read_csv_flexible("historico_12m_por_depto.csv")
        if "Departamento" not in hist12.columns:
            # autodetect
            name_col = next((c for c in hist12.columns if re.search(r"depar|nombre", c, re.I)), None)
            if name_col: hist12 = hist12.rename(columns={name_col:"Departamento"})
        val_col = next((c for c in hist12.columns if pd.api.types.is_numeric_dtype(hist12[c])), None)
        hist12 = hist12.rename(columns={val_col:"Valor_12m"})[["Departamento","Valor_12m"]]
    except Exception:
        pass  # si no existe, lo tratamos abajo

    last_hist_date = (serie.index.max() if len(serie) else pd.NaT)
    return serie, fc, dept, hist12, last_hist_date

@st.cache_data
def load_geojson_with_norm(path: str):
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    # Asegurar properties.name_norm en cada feature
    for ft in gj.get("features", []):
        props = ft.setdefault("properties", {})
        name = (
            props.get("name_norm")
            or props.get("NOMBRE_DPT")
            or props.get("DEPARTAMEN")
            or props.get("DEPARTAMENTO")
            or props.get("NAME")
            or next((v for v in props.values() if isinstance(v, str)), "")
        )
        props["name_norm"] = _norm(name)
        ft["properties"] = props
    names_geo = {ft["properties"]["name_norm"] for ft in gj["features"]}
    return gj, names_geo

# ---------------------------- Layout: Sidebar ---------------------------------
with st.sidebar:
    st.header("MAP — Parte 1")
    st.caption("Dashboard de resultados (pronóstico Q1-2024)")

    if st.button("🔄 Recargar datos"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    # Logo compacto al final de la barra lateral
    bottom = st.container()
    with bottom:
        st.markdown(
            f"**Realizado por:** {AUTHOR_NAME} · "
            f"[{AUTHOR_EMAIL}](mailto:{AUTHOR_EMAIL})"
        )
        if Path(LOGO_PATH).exists():
            st.image(LOGO_PATH, caption="3iS · information · innovation · impact",
                     use_container_width=True)
        else:
            st.caption("Sube el logo en `assets/logo.png` para verlo aquí.")

# ----------------------------- Encabezado -------------------------------------
st.title(APP_TITLE)
st.write(f"**Realizado por:** {AUTHOR_NAME} · [{AUTHOR_EMAIL}](mailto:{AUTHOR_EMAIL})")

# ----------------------------- Carga ------------------------------------------
try:
    serie, fc, dept_q1, hist12, last_hist_date = load_series_and_forecasts()
except Exception as e:
    st.error(f"Error al cargar CSVs: {e}")
    st.stop()

# ----------------------------- Visualizaciones --------------------------------
tab_viz, tab_map, tab_dl = st.tabs(["📊 Visualizaciones", "🗺️ Mapa por departamento", "📥 Descargas"])

with tab_viz:
    col1, col2 = st.columns([1,1])

    # Serie + forecast nacional
    with col1:
        st.subheader("Histórico y pronóstico (nacional)")
        to_plot = pd.concat([serie["hist"].rename("hist"), fc["forecast"].rename("forecast")], axis=1).sort_index()
        st.line_chart(to_plot, use_container_width=True)
        if {"lower_95","upper_95"}.issubset(fc.columns):
            st.caption("Nota: IC≈95% calculado a partir del error de validación del modelo seleccionado (aprox).")

    # Top-10 departamental (predicción)
    with col2:
        st.subheader("Top-10 departamentos (Q1-2024, predicho)")
        top10 = dept_q1.rename(columns={"Pred_Q1":"valor"}).sort_values("valor", ascending=False).head(10)
        st.bar_chart(top10.set_index("Departamento")["valor"], use_container_width=True)

    st.divider()
    st.write("**Metodología (resumen):** PM-12 vs SARIMA vs GBR con split temporal (train ≤2022, valid=2023). Limpieza de etiquetas y normalización de fechas/códigos. Reparto proporcional por participación reciente para estimar Q1 por departamento.")

# ----------------------------- Mapa por departamento ---------------------------
with tab_map:
    st.subheader("Mapa choropleth por departamento")
    geojson_file = _find_first_existing(GEOJSON_PATH)
    if not geojson_file:
        st.error(f"No se encontró el GeoJSON en `{GEOJSON_PATH}`. Súbelo para ver el mapa.")
        st.stop()

    geojson, names_geo = load_geojson_with_norm(str(geojson_file))

    opt = st.radio("Selecciona capa a visualizar:",
                   ["Histórico (últimos 12 meses)", "Pronóstico Q1-2024"],
                   horizontal=True)

    if opt.startswith("Histórico") and hist12 is not None:
        df_map = hist12.rename(columns={"Valor_12m":"valor"}).copy()
        title = f"Histórico (total 12m hasta {last_hist_date.date() if pd.notna(last_hist_date) else '…'})"
    else:
        df_map = dept_q1.rename(columns={"Pred_Q1":"valor"}).copy()
        if opt.startswith("Histórico"):
            st.info("No se encontró `historico_12m_por_depto.csv`. Mostrando pronóstico Q1-2024 como alternativa.")
        title = "Pronóstico Q1-2024"

    # normalizar nombres para el join
    df_map["name_norm"] = df_map["Departamento"].map(_norm)

    # aviso de faltantes reales
    faltan = sorted(set(df_map["name_norm"]) - names_geo)
    if faltan:
        st.info("Departamentos no encontrados en el GeoJSON (normalizados): " + ", ".join(faltan))

    # choropleth
    fig = px.choropleth(
        df_map,
        geojson=geojson,
        locations="name_norm",
        featureidkey="properties.name_norm",
        color="valor",
        color_continuous_scale="Blues",
        labels={"valor":"Víctimas"},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Descargas --------------------------------------
with tab_dl:
    st.write("Descarga los datos usados por el dashboard:")
    c1, c2, c3 = st.columns(3)

    # Serie nacional
    c1.download_button(
        "Serie nacional (CSV)",
        serie.reset_index().rename(columns={"index":"date"}).to_csv(index=False).encode("utf-8"),
        file_name="serie_nacional_mensual.csv",
        mime="text/csv",
    )
    # Forecast nacional
    c2.download_button(
        "Forecast Q1-2024 (CSV)",
        fc.reset_index().rename(columns={"index":"date"}).to_csv(index=False).encode("utf-8"),
        file_name="forecast_nacional_Q1_2024.csv",
        mime="text/csv",
    )
    # Departamentos Q1
    c3.download_button(
        "Departamentos Q1-2024 (CSV)",
        dept_q1.to_csv(index=False).encode("utf-8"),
        file_name="forecast_depto_Q1_2024.csv",
        mime="text/csv",
    )
