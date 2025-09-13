# MAP — Parte 1 (Streamlit)  
**Víctimas por Minas Antipersonal — Pronóstico 2024-Q1**

Dashboard en **Streamlit** para visualizar el histórico nacional de víctimas por minas antipersonal, el **pronóstico Q1-2024** y un **mapa coroplético por departamento** (33/33 departamentos garantizados).

> **Autor:** Andy Domínguez · [ardominguezm@gmail.com](mailto:ardominguezm@gmail.com)

---

## ✨ Características

- **Gráfico nacional**: histórico vs pronóstico en colores diferenciados, línea punteada para el forecast, **banda de IC≈95%** y sombreado del tramo de pronóstico.
- **Mapa por departamento**: choropleth con cobertura **100%** (si el CSV trae solo Top-10, la app completa faltantes con **0**).
- **Tema oscuro** (fondo negro, texto blanco) + **widgets**: paleta, invertir paleta, grosor del borde, barra de color.
- **Descargas** de los CSV usados por el dashboard.

---

## 🗂️ Estructura del repositorio


├── app.py
├── requirements.txt
├── .streamlit/
│ └── config.toml
├── assets/
│ └── logo_3is.png # mostrado en la barra lateral
├── data/
│ └── colombia_departamentos.geojson
└── outputs_parte1/
├── serie_nacional_mensual.csv
├── forecast_nacional_Q1_2024.csv
├── forecast_depto_Q1_2024.csv
└── serie_departamental_mensual.csv # 

---

## 📦 Requisitos

`requirements.txt`:

```txt
streamlit
pandas
numpy
plotly
