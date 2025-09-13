# ===========================
#  APP: MAP — 2024 Q1 (Streamlit)
#  Autor: Andy Domínguez (ardominguezm@gmail.com)
#  Logo lateral compacto + Mapa por departamento (histórico & pronóstico)
# ===========================

import builtins as _b
str = _b.str; list = _b.list; dict = _b.dict; set = _b.set

import json, unicodedata, base64
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as _dt

APP_VERSION = "v1.4-2025-09-12"

# -------- Config página --------
st.set_page_config(
    page_title="MAP — 2024 Q1",
    page_icon="📊",
    layout="wide",
)

# -------- Personalización --------
AUTHOR_NAME  = "Andy Domínguez"
AUTHOR_EMAIL = "ardominguezm@gmail.com"

# Archivos esperados
GEOJSON_PATH = "data/colombia_departamentos.geojson"  # súbelo al repo
CSV_SERIE_NAC = "outputs_parte1/serie_nacional_mensual.csv"
CSV_FC_NAC    = "outputs_parte1/forecast_nacional_Q1_2024.csv"
CSV_FC_DEP    = "outputs_parte1/forecast_depto_Q1_2024.csv"
CSV_SERIE_DEP = "outputs_parte1/serie_departamental_mensual.csv"  # <- nuevo (ver notas abajo)

# Rutas de logo candidatas
LOGO_CANDIDATES = ["assets/logo_3is.png"]

# ===========================
#  Utilidades
# ===========================
def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def norm_name(s: str) -> str:
    s = _strip_accents(str(s).upper())
    # normaliza separadores y variantes comunes
    s = s.replace(".", " ").replace(",", " ").replace("  ", " ").strip()
    # unifica algunos casos problemáticos
    repl = {
        "BOGOTA D C": "BOGOTA DC",
        "BOGOTA": "BOGOTA DC",
        "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "SAN ANDRES Y PROVIDENCIA",
        "SAN ANDRES": "SAN ANDRES Y PROVIDENCIA",
        "VALLE": "VALLE DEL CAUCA",
        "VALLE DEL CAUCA": "VALLE DEL CAUCA",
        "N DE SANTANDER": "NORTE DE SANTANDER",
        "NORTE SANTANDER": "NORTE DE SANTANDER",
    }
    return repl.get(s, s)

def try_read_csv(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"No se pudo leer {path}: {e}")
        return None

def load_geojson(path: str):
    p = Path(path)
    if not p.exists():
        return None, None  # geojson, feature_name_key
    with open(p, "r", encoding="utf-8") as f:
        gj = json.load(f)
    # intenta detectar el nombre del departamento en las propiedades
    candidates = [
        "NOMBRE_DPT", "NOMBRE_DEP", "DPTO_CNMBR", "NOMBRE", "DEPARTAMEN", "name",
        "dpt_name", "NOMBRE_DEPARTAMENTO"
    ]
    props = gj["features"][0]["properties"]
    key = next((k for k in candidates if k in props), None)
    if key is None:
        # si no detecta, toma el primer string que parezca nombre
        for k, v in props.items():
            if isinstance(v, str):
                key = k; break
    # agrega campo normalizado para join robusto
    for ft in gj["features"]:
        nm = ft["properties"].get(key, "")
        ft["properties"]["name_norm"] = norm_name(nm)
    return gj, "name_norm"

def logo_data_url():
    for p in LOGO_CANDIDATES:
        if Path(p).exists():
            with open(p, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    return None

# ===========================
#  Carga de datos (cache)
# ===========================
@st.cache_data
def load_all():
    # Serie nacional
    serie = try_read_csv(CSV_SERIE_NAC)
    if serie is None: raise FileNotFoundError(f"Falta {CSV_SERIE_NAC}")
    s_date_col = next((c for c in serie.columns if str(c).lower() in ("date","fecha","index","unnamed: 0")), None)
    if s_date_col is None: raise ValueError("No se encontró columna de fecha en serie_nacional_mensual.csv")
    serie["date"] = pd.to_datetime(serie[s_date_col])
    s_val_col = next((c for c in serie.columns if c != s_date_col), None)
    if s_val_col is None: serie = serie.rename(columns={serie.columns[0]: "date", serie.columns[1]: "hist"})
    else: serie = serie.rename(columns={s_val_col: "hist"})
    serie = serie[["date","hist"]].sort_values("date").set_index("date")

    # Forecast nacional
    fc_nac = try_read_csv(CSV_FC_NAC)
    if fc_nac is None: raise FileNotFoundError(f"Falta {CSV_FC_NAC}")
    f_date_col = next((c for c in fc_nac.columns if str(c).lower() in ("date","fecha","index","unnamed: 0")), None)
    if f_date_col is None: raise ValueError("No se encontró columna de fecha en forecast_nacional_Q1_2024.csv")
    fc_nac["date"] = pd.to_datetime(fc_nac[f_date_col])
    if "forecast" not in fc_nac.columns:
        num_cols = [c for c in fc_nac.columns if pd.api.types.is_numeric_dtype(fc_nac[c])]
        if not num_cols: raise ValueError("No hay columna numérica para 'forecast'.")
        fc_nac = fc_nac.rename(columns={num_cols[0]: "forecast"})
    for col in ("lower_95","upper_95"):
        if col not in fc_nac.columns: fc_nac[col] = pd.NA
    fc_nac = fc_nac[["date","forecast","lower_95","upper_95"]].set_index("date").sort_index()

    # Forecast departamental Q1
    fc_dep = try_read_csv(CSV_FC_DEP)
    if fc_dep is None: raise FileNotFoundError(f"Falta {CSV_FC_DEP}")
    if "departamento" in fc_dep.columns: fc_dep = fc_dep.rename(columns={"departamento":"Departamento"})
    if "pred_Q1_2024" in fc_dep.columns: fc_dep = fc_dep.rename(columns={"pred_Q1_2024":"Pred_Q1"})
    if not {"Departamento","Pred_Q1"}.issubset(fc_dep.columns):
        raise ValueError("CSV de departamentos debe tener columnas 'Departamento' y 'Pred_Q1'.")

    # Serie departamental mensual (para histórico por mapa) — opcional
    serie_dep = try_read_csv(CSV_SERIE_DEP)
    if serie_dep is not None:
        date_col = next((c for c in serie_dep.columns if str(c).lower() in ("date","fecha","index","unnamed: 0")), None)
        if date_col is None: raise ValueError("No se encontró columna de fecha en serie_departamental_mensual.csv")
        serie_dep["date"] = pd.to_datetime(serie_dep[date_col])
        # estandariza nombres de columnas
        cand_dep = [c for c in serie_dep.columns if c.lower() in ("departamento","dept","depto","dpto","nombre_departamento")]
        if not cand_dep: raise ValueError("serie_departamental_mensual.csv debe tener columna de departamento.")
        dep_col = cand_dep[0]
        val_col = next((c for c in serie_dep.columns if c not in (date_col, dep_col)), None)
        if val_col is None: raise ValueError("serie_departamental_mensual.csv debe tener columna de conteos.")
        serie_dep = serie_dep.rename(columns={dep_col: "Departamento", val_col: "victims"})
        serie_dep["Departamento_norm"] = serie_dep["Departamento"].map(norm_name)
        serie_dep = serie_dep[["date", "Departamento", "Departamento_norm", "victims"]].sort_values(["date","Departamento"])
    # GeoJSON
    geojson, feat_norm_key = load_geojson(GEOJSON_PATH)

    return serie, fc_nac, fc_dep, serie_dep, geojson, feat_norm_key

# ===========================
#  Header + Sidebar brand
# ===========================
def render_header():
    st.markdown("# Víctimas por Minas Antipersonal — Pronóstico 2024-Q1")
    st.markdown(
        f"**Realizado por:** {AUTHOR_NAME} · "
        f"[{AUTHOR_EMAIL}](mailto:{AUTHOR_EMAIL})"
    )

def sidebar_brand():
    # CSS para anclar al fondo
    st.sidebar.markdown(
        """
        <style>
        section[data-testid="stSidebar"] > div { display: flex; flex-direction: column; height: 100%; }
        .sidebar-spacer { flex: 1 1 auto; }
        .sidebar-brand { padding: 10px 12px; border-top: 1px solid #e5e7eb; font-size: 13px; color: #374151; }
        .sidebar-brand img { width: 120px; display: block; margin: 0 auto 6px; }
        .center { text-align: center; line-height: 1.2; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
    src = logo_data_url()
    img = f'<img src="{src}" alt="logo"/>' if src else ""
    st.sidebar.markdown(
        f'<div class="sidebar-brand">{img}<div class="center"><strong>{AUTHOR_NAME}</strong><br/>'
        f'<a href="mailto:{AUTHOR_EMAIL}">{AUTHOR_EMAIL}</a></div></div>',
        unsafe_allow_html=True
    )

# ===========================
#  Main
# ===========================
try:
    serie, fc_nac, fc_dep, serie_dep, geojson, feat_norm_key = load_all()
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

render_header()
sidebar_brand()

# -------- Tabs --------
tab_viz, tab_map, tab_dl = st.tabs(["📊 Visualizaciones", "🗺️ Mapa por departamento", "📥 Descargas"])

with tab_viz:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Histórico y pronóstico (nacional)")
        to_plot = pd.concat([serie["hist"], fc_nac["forecast"].rename("forecast")], axis=1).sort_index()
        to_plot.index.name = "date"
        st.line_chart(to_plot, use_container_width=True)
        st.caption("Nota: IC≈95% calculado a partir del error de validación del modelo seleccionado.")
    with col2:
        st.subheader("Top-10 departamentos (Q1-2024, predicho)")
        top10 = (
            fc_dep.sort_values("Pred_Q1", ascending=False)
            .head(10).copy()
            .set_index("Departamento")["Pred_Q1"]
        )
        st.bar_chart(top10, use_container_width=True)

# -------- MAPA --------
with tab_map:
    st.subheader("Mapa choropleth por departamento")

    if geojson is None:
        st.warning(
            "Falta el archivo GeoJSON de departamentos. "
            f"Sube uno como **{GEOJSON_PATH}** (propiedad de nombre de dpto estándar)."
        )
    else:
        # Selector de capa
        capa = st.radio(
            "Selecciona capa a visualizar:",
            ["Histórico (últimos 12 meses)", "Pronóstico Q1-2024"],
            horizontal=True,
        )

        # Construir dataframe para mapa
        if capa.startswith("Histórico"):
            if serie_dep is None:
                st.warning(
                    "Para el histórico por departamento se requiere **outputs_parte1/serie_departamental_mensual.csv** "
                    "con columnas: date, Departamento, victims (mensual)."
                )
                df_map = None
            else:
                # últimos 12 meses disponibles en la serie departamental
                last_date = serie_dep["date"].max()
                start_date = (last_date - pd.DateOffset(months=11)).normalize()
                mask = (serie_dep["date"] >= start_date) & (serie_dep["date"] <= last_date)
                tmp = (serie_dep.loc[mask]
                       .groupby(["Departamento","Departamento_norm"], as_index=False)["victims"].sum())
                tmp = tmp.rename(columns={"victims": "valor"})
                layer_name = f"Histórico (total 12m hasta {last_date.date()})"
                df_map = tmp
        else:
            tmp = fc_dep.copy()
            tmp["Departamento_norm"] = tmp["Departamento"].map(norm_name)
            tmp = tmp.rename(columns={"Pred_Q1": "valor"})
            layer_name = "Pronóstico Q1-2024 (total)"
            df_map = tmp[["Departamento","Departamento_norm","valor"]]

        if df_map is not None:
            # Prepara GeoJSON con clave normalizada ya creada (feat_norm_key)
            # Genera choropleth
            fig = px.choropleth(
                df_map,
                geojson=geojson,
                locations="Departamento_norm",
                featureidkey=f"properties.{feat_norm_key}",
                color="valor",
                color_continuous_scale="Blues",
                labels={"valor": "Víctimas"},
                hover_name="Departamento",
                hover_data={"Departamento_norm": False, "valor": ":.1f"},
            )
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                coloraxis_colorbar=dict(title="Víctimas"),
                title=layer_name,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Resumen de mapeo / faltantes
            # construye set de nombres de geojson
            gj_names = {norm_name(ft["properties"][feat_norm_key]) for ft in geojson["features"]}
            df_names = set(df_map["Departamento_norm"].unique())
            missing_in_geo = sorted(df_names - gj_names)
            if missing_in_geo:
                st.info(
                    "Departamentos no encontrados en el GeoJSON (normalizados): "
                    + ", ".join(missing_in_geo)
                )

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
        "Forecast nacional Q1-2024 (CSV)",
        fc_nac.reset_index().rename(columns={"index": "date"}).to_csv(index=False).encode("utf-8"),
        file_name="forecast_nacional_Q1_2024.csv",
        mime="text/csv",
    )
    c3.download_button(
        "Departamentos Q1-2024 (CSV)",
        fc_dep.to_csv(index=False).encode("utf-8"),
        file_name="forecast_depto_Q1_2024.csv",
        mime="text/csv",
    )

# Pie + versión
st.divider()
st.caption(
    "Realizado por: **Andy Domínguez** "
    f"([{AUTHOR_EMAIL}](mailto:{AUTHOR_EMAIL})) · "
    f"Generado: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} · {APP_VERSION}"
)


