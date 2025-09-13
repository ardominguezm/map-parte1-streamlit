# app.py — Dark Edition
# -----------------------------------------------------------------------------
# MAP — Parte 1 (Streamlit)
# Autor: Andy Domínguez · ardominguezm@gmail.com
# Requiere: streamlit, pandas, numpy, plotly
#
# Datos esperados (carpeta outputs_parte1/):
#   - serie_nacional_mensual.csv
#   - forecast_nacional_Q1_2024.csv
#   - forecast_depto_Q1_2024.csv          # puede venir subset (Top-10). El mapa completa a 33 con 0
#   - (opcional) serie_departamental_mensual.csv  # date, Departamento, victims (mensual)
#
# GeoJSON esperado:
#   - data/colombia_departamentos.geojson  (propiedades NOMBRE_DPT, DPTO_CODE, name_norm)
# -----------------------------------------------------------------------------

import builtins as _b
str=_b.str; list=_b.list; dict=_b.dict; set=_b.set

import json, re, unicodedata
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Config general
st.set_page_config(page_title="MAP — 2024 Q1", layout="wide")
px.defaults.template = "plotly_dark"  # plotly en modo oscuro

APP_TITLE = "Dashboard Modelo Predictivo Víctimas por Minas Antipersona — Pronóstico 2024-Q1"
AUTHOR_NAME = "Andy Domínguez"
AUTHOR_EMAIL = "ardominguezm@gmail.com"

OUTPUTS_DIR = Path("outputs_parte1")
GEOJSON_PATH = "data/colombia_departamentos.geojson"
LOGO_PATH = "assets/logo_3is.png"

# ---- CSS para pulir modo oscuro (banner, enlaces, tabs)
st.markdown("""
<style>
/* Banner oscuro */
.hero {
  background: linear-gradient(90deg,#0b0c10 0%, #16181d 50%, #0B5FFF 100%);
  border-radius: 14px; padding: 18px 22px; margin: 6px 0 12px 0;
}
.hero h1, .hero p, .hero a { color: #fff !important; margin: 0; }
.hero p { opacity: .95; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
a { color: #9bc3ff !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------- Utilidades -----------------------------------------
def norm_name(s: str) -> str:
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

def read_csv_safe(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None

# ------------------------- Carga de insumos (cache) ---------------------------
@st.cache_data
def load_inputs(_outputs_dir: str, _geojson_path: str):
    outdir = Path(_outputs_dir)

    # Serie nacional
    serie = read_csv_safe(outdir / "serie_nacional_mensual.csv")
    if serie is None:
        raise FileNotFoundError("Falta outputs_parte1/serie_nacional_mensual.csv")
    s_date_col = next((c for c in serie.columns if c.lower() in ("date","fecha","index","unnamed: 0")), serie.columns[0])
    serie["date"] = pd.to_datetime(serie[s_date_col], errors="coerce")
    serie = serie.dropna(subset=["date"]).set_index("date").sort_index()
    s_val_col = next((c for c in serie.columns if pd.api.types.is_numeric_dtype(serie[c])), None)
    if s_val_col is None:
        raise ValueError("No se encontró columna numérica en serie_nacional_mensual.csv")
    serie = serie.rename(columns={s_val_col:"hist"})[["hist"]]

    # Forecast nacional Q1
    fc = read_csv_safe(outdir / "forecast_nacional_Q1_2024.csv")
    if fc is None:
        raise FileNotFoundError("Falta outputs_parte1/forecast_nacional_Q1_2024.csv")
    f_date_col = next((c for c in fc.columns if c.lower() in ("date","fecha","index","unnamed: 0")), fc.columns[0])
    fc["date"] = pd.to_datetime(fc[f_date_col], errors="coerce")
    fc = fc.dropna(subset=["date"]).set_index("date").sort_index()
    if "forecast" not in fc.columns:
        num_cols = [c for c in fc.columns if pd.api.types.is_numeric_dtype(fc[c])]
        fc = fc.rename(columns={num_cols[0]:"forecast"}) if num_cols else fc.assign(forecast=np.nan)
    for col in ("lower_95","upper_95"):
        if col not in fc.columns:
            fc[col] = np.nan

    # Forecast departamental Q1 (puede ser subset)
    fc_dep = read_csv_safe(outdir / "forecast_depto_Q1_2024.csv")
    if fc_dep is None:
        raise FileNotFoundError("Falta outputs_parte1/forecast_depto_Q1_2024.csv")
    if "departamento" in fc_dep.columns: fc_dep = fc_dep.rename(columns={"departamento":"Departamento"})
    if "Pred_Q1" not in fc_dep.columns:
        if "pred_Q1_2024" in fc_dep.columns:
            fc_dep = fc_dep.rename(columns={"pred_Q1_2024":"Pred_Q1"})
        else:
            num_cols = [c for c in fc_dep.columns if pd.api.types.is_numeric_dtype(fc_dep[c])]
            if num_cols: fc_dep = fc_dep.rename(columns={num_cols[0]:"Pred_Q1"})
    if "Departamento" not in fc_dep.columns or "Pred_Q1" not in fc_dep.columns:
        raise ValueError("forecast_depto_Q1_2024.csv debe incluir 'Departamento' y 'Pred_Q1'.")

    # Serie departamental mensual (opcional)
    serie_dep = read_csv_safe(outdir / "serie_departamental_mensual.csv")
    if serie_dep is not None:
        dcol = next((c for c in serie_dep.columns if c.lower() in ("date","fecha")), serie_dep.columns[0])
        serie_dep["date"] = pd.to_datetime(serie_dep[dcol], errors="coerce")
        serie_dep = serie_dep.dropna(subset=["date"])
        if "Departamento" not in serie_dep.columns:
            cand = next((c for c in serie_dep.columns if re.search(r"depar|nombre", c, re.I)), None)
            if cand: serie_dep = serie_dep.rename(columns={cand:"Departamento"})
        vcol = next((c for c in serie_dep.columns if pd.api.types.is_numeric_dtype(serie_dep[c]) and c != "date"), None)
        if vcol is None:
            raise ValueError("serie_departamental_mensual.csv debe tener columna numérica mensual.")
        serie_dep = serie_dep.rename(columns={vcol:"victims"})[["date","Departamento","victims"]]
        serie_dep["Departamento_norm"] = serie_dep["Departamento"].map(norm_name)

    # GeoJSON + base de 33 dptos
    with open(_geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    base_rows = []
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
        props["name_norm"] = norm_name(name)
        label = props.get("NOMBRE_DPT") or props.get("DEPARTAMEN") or props.get("DEPARTAMENTO") or name
        base_rows.append({"Departamento_norm": props["name_norm"], "Departamento_label": str(label)})
        ft["properties"] = props
    base_depts = pd.DataFrame(base_rows).drop_duplicates(subset=["Departamento_norm"]).sort_values("Departamento_label")
    feat_norm_key = "name_norm"

    last_hist_date = (serie.index.max() if len(serie) else pd.NaT)
    return serie, fc, fc_dep, serie_dep, gj, feat_norm_key, base_depts, last_hist_date

# ----------------------------- Sidebar ----------------------------------------
with st.sidebar:
    st.header("MAP — Parte 1")
    st.caption("Dashboard (modo oscuro)")

    if st.button("🔄 Recargar datos"):
        st.cache_data.clear()
        st.rerun()

    st.subheader("🎨 Apariencia")
    palette = st.selectbox(
        "Paleta",
        ["Viridis","Cividis","Plasma","Inferno","Magma","Turbo",
         "Blues","Greens","Greys","Oranges","Purples","Reds",
         "YlGn","YlGnBu","YlOrBr","YlOrRd","GnBu","BuGn","PuBu","PuBuGn","BuPu","RdPu","PuRd","OrRd",
         "RdBu","RdYlBu","RdYlGn","BrBG","PRGn","PiYG","PuOr","Spectral"],
        index=0,
    )
    rev = st.toggle("Invertir paleta", False)
    outline = st.slider("Borde del mapa", 0.2, 2.0, 0.6, 0.1)
    show_scale = st.toggle("Mostrar barra de color", True)

    st.markdown("---")
    st.markdown(
        f"**Realizado por:** {AUTHOR_NAME} · "
        f"[{AUTHOR_EMAIL}](mailto:{AUTHOR_EMAIL})"
    )
    if Path(LOGO_PATH).exists():
        st.image(LOGO_PATH, caption="3iS · information · innovation · impact",
                 use_container_width=True)

# ----------------------------- Banner -----------------------------------------
st.markdown(f"""
<div class="hero">
  <h1>{APP_TITLE}</h1>
  <p>Realizado por: {AUTHOR_NAME} · <a href="mailto:{AUTHOR_EMAIL}">{AUTHOR_EMAIL}</a></p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Carga ------------------------------------------
if not Path(GEOJSON_PATH).exists():
    st.error(f"No se encontró el GeoJSON en `{GEOJSON_PATH}`. Súbelo para ver el mapa.")
    st.stop()

try:
    serie, fc, fc_dep, serie_dep, geojson, feat_norm_key, base_depts, last_hist_date = load_inputs(str(OUTPUTS_DIR), GEOJSON_PATH)
except Exception as e:
    st.error(f"Error al cargar insumos: {e}")
    st.stop()

# ----------------------------- Pestañas ---------------------------------------
tab_viz, tab_map, tab_dl = st.tabs(["📊 Visualizaciones", "🗺️ Mapa por departamento", "📥 Descargas"])

with tab_viz:
    col1, col2 = st.columns([1,1])

    # Serie + forecast nacional
    with col1:
        st.subheader("Histórico y pronóstico (nacional)")
        to_plot = pd.concat([serie["hist"].rename("hist"), fc["forecast"].rename("forecast")], axis=1).sort_index()
        st.line_chart(to_plot, use_container_width=True)
        if {"lower_95","upper_95"}.issubset(fc.columns):
            st.caption("IC≈95% basado en error de validación (aprox).")

    # Top-10 departamental (predicción)
    with col2:
        st.subheader("Top-10 departamentos (Q1-2024, predicho)")
        top10 = fc_dep.rename(columns={"Pred_Q1":"valor"}).sort_values("valor", ascending=False).head(10)
        st.bar_chart(top10.set_index("Departamento")["valor"], use_container_width=True)

    st.divider()
    st.write("**Metodología (resumen):** PM-12 vs SARIMA vs GBR con split temporal (train ≤2022, valid=2023). Limpieza de etiquetas/códigos y reparto proporcional por participación reciente para estimar Q1 por departamento.")

# ----------------------------- Mapa -------------------------------------------
with tab_map:
    st.subheader("Mapa choropleth por departamento")

    has_hist = (serie_dep is not None)
    if has_hist:
        opt = st.radio(
            "Selecciona capa:",
            ["Histórico (últimos 12 meses)", "Pronóstico Q1-2024"],
            horizontal=True,
            index=1,
        )
    else:
        opt = "Pronóstico Q1-2024"

    if has_hist and opt.startswith("Histórico"):
        last_date = serie_dep["date"].max()
        start_date = (last_date - pd.DateOffset(months=11)).normalize()
        mask = (serie_dep["date"] >= start_date) & (serie_dep["date"] <= last_date)
        tmp = (serie_dep.loc[mask]
               .groupby(["Departamento","Departamento_norm"], as_index=False)["victims"].sum())
        tmp = tmp.rename(columns={"victims":"valor"})
        df_map = base_depts.merge(tmp[["Departamento_norm","valor"]], on="Departamento_norm", how="left")
        df_map["valor"] = df_map["valor"].fillna(0.0)
        df_map["Departamento"] = df_map["Departamento_label"]
        layer_name = f"Histórico (total 12m hasta {last_date.date()})"
    else:
        tmp = fc_dep.copy()
        tmp["Departamento_norm"] = tmp["Departamento"].map(norm_name)
        tmp = tmp.rename(columns={"Pred_Q1":"valor"})
        df_map = base_depts.merge(tmp[["Departamento_norm","valor"]], on="Departamento_norm", how="left")
        df_map["valor"] = df_map["valor"].fillna(0.0)
        df_map["Departamento"] = df_map["Departamento_label"]
        layer_name = "Pronóstico Q1-2024"

    fig = px.choropleth(
        df_map,
        geojson=geojson,
        locations="Departamento_norm",
        featureidkey=f"properties.{feat_norm_key}",
        color="valor",
        color_continuous_scale=palette,
        labels={"valor":"Víctimas"},
        hover_name="Departamento",
        hover_data={"Departamento_norm": False, "valor": ":.2f"},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_traces(marker_line_width=outline, marker_line_color="rgba(255,255,255,.55)")
    fig.update_layout(
        coloraxis_reversescale=rev,
        coloraxis_showscale=show_scale,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(title="Víctimas"),
        title=layer_name,
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Descargas --------------------------------------
with tab_dl:
    st.write("Descarga los datos usados por el dashboard:")
    c1, c2, c3, c4 = st.columns(4)

    c1.download_button(
        "Serie nacional (CSV)",
        serie.reset_index().rename(columns={"index":"date"}).to_csv(index=False).encode("utf-8"),
        file_name="serie_nacional_mensual.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Forecast nacional (CSV)",
        fc.reset_index().rename(columns={"index":"date"}).to_csv(index=False).encode("utf-8"),
        file_name="forecast_nacional_Q1_2024.csv",
        mime="text/csv",
    )
    c3.download_button(
        "Departamentos Q1-2024 (CSV)",
        fc_dep.to_csv(index=False).encode("utf-8"),
        file_name="forecast_depto_Q1_2024.csv",
        mime="text/csv",
    )
    if serie_dep is not None:
        c4.download_button(
            "Serie departamental mensual (CSV)",
            serie_dep.to_csv(index=False).encode("utf-8"),
            file_name="serie_departamental_mensual.csv",
            mime="text/csv",
        )
    else:
        c4.write("—")
