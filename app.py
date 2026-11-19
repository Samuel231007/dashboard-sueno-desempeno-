"""
Dashboard — Hábitos de sueño y desempeño académico percibido
Tercer semestre de Ingeniería · Universidad de La Sabana · 2026-2

Ejecutar:
    pip install -r requirements.txt
    streamlit run app.py

Datos: lee base_procesada.xlsx (o BaseDeDatos.xlsx, hoja Base_datos) en la misma carpeta,
o un archivo que se suba desde la barra lateral. Solo usa las 26 respuestas codificadas;
todas las variables derivadas se recalculan aquí.

Criterios: las respuestas son categorías ordenadas (se analizan por rangos), las
asociaciones no implican causalidad y la muestra es voluntaria (no probabilística).
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Sueño y desempeño · Ingeniería 2026-2", page_icon="🌙", layout="wide")

# ------------------------------------------------------------------ diccionario
FREC = {1: "Nunca", 2: "Rara vez", 3: "Algunas veces", 4: "Frecuentemente", 5: "Siempre"}
ACOST = {1: "Antes de 9 p. m.", 2: "9–10 p. m.", 3: "10–11 p. m.", 4: "11 p. m.–12 a. m.", 5: "Después de 12 a. m."}
HORAS = {1: "< 4 h", 2: "4 a < 6 h", 3: "6 a < 7 h", 4: "7 a < 8 h", 5: "8 a < 9 h", 6: "≥ 9 h"}
CB = {
    "PROGRAMA": ("P1", "Programa", "nominal", {1: "Ciencia de Datos", 2: "Ing. Civil", 3: "Ing. en Bioproducción",
                                            4: "Ing. en Diseño", 5: "Ing. Industrial", 6: "Ing. Informática",
                                            7: "Ing. Mecánica", 8: "Ing. Química"}, 0),
    "EDAD": ("P2", "Edad", "ordinal", {1: "< 17", 2: "17–18", 3: "19–20", 4: "21–22", 5: "> 22"}, 0),
    "SEXO": ("P3", "Sexo", "nominal", {1: "Femenino", 2: "Masculino", 3: "Prefiere no responder"}, 0),
    "HR_DORMIR_SEM": ("P4", "Hora de acostarse (semana)", "ordinal", ACOST, 0),
    "HR_LEVANTAR_SEM": ("P5", "Hora de levantarse (semana)", "ordinal",
                        {1: "Antes de 5 a. m.", 2: "5–6 a. m.", 3: "6–7 a. m.", 4: "7–8 a. m.", 5: "8–9 a. m.",
                         6: "Después de 9 a. m."}, 0),
    "HR_DORMIR_FIN": ("P6", "Hora de acostarse (fin de semana)", "ordinal", ACOST, 0),
    "HR_LEVANTAR_FIN": ("P7", "Hora de levantarse (fin de semana)", "ordinal",
                        {1: "Antes de 7 a. m.", 2: "7–8 a. m.", 3: "8–9 a. m.", 4: "9–10 a. m.", 5: "10–11 a. m.",
                         6: "Después de 11 a. m."}, 0),
    "HORAS_SUENO_SEM": ("P8", "Horas de sueño (semana)", "ordinal", HORAS, 1),
    "HORAS_SUENO_FIN": ("P9", "Horas de sueño (fin de semana)", "ordinal", HORAS, 0),
    "FREC_SIESTA": ("P10", "Siestas", "ordinal", FREC, 0),
    "FREC_DISPOSITIVOS": ("P11", "Dispositivos antes de dormir", "ordinal", FREC, 0),
    "FREC_CAFEINA": ("P12", "Cafeína después de 6 p. m.", "ordinal", FREC, 0),
    "FREC_CARGA_ACAD": ("P13", "Carga académica altera el horario", "ordinal", FREC, 0),
    "REGULARIDAD_HORARIO": ("P14", "Horarios regulares entre semana", "ordinal", FREC, 1),
    "CALIDAD_SUENO": ("P15", "Calidad global del sueño", "ordinal",
                      {1: "Muy mala", 2: "Mala", 3: "Regular", 4: "Buena", 5: "Muy buena"}, 1),
    "FREC_CONCILIAR": ("P16", "Le cuesta conciliar el sueño", "ordinal", FREC, -1),
    "FREC_DESPERTAR_NOCHE": ("P17", "Despertares nocturnos", "ordinal", FREC, -1),
    "FREC_SUENO_NO_REPARADOR": ("P18", "Sueño no reparador", "ordinal", FREC, -1),
    "FREC_SOMNOLENCIA_DIA": ("P19", "Somnolencia diurna", "ordinal", FREC, -1),
    "FREC_DIFICULTAD_CONCENTRACION": ("P20", "Dificultad para concentrarse", "ordinal", FREC, -1),
    "PRODUCTIVIDAD_ACAD": ("P21", "Productividad académica", "ordinal",
                           {1: "Muy baja", 2: "Baja", 3: "Regular", 4: "Alta", 5: "Muy alta"}, 1),
    "FREC_ERRORES_CANSANCIO": ("P22", "Errores por cansancio", "ordinal", FREC, -1),
    "AFECTO_EVALUACIONES": ("P23", "Falta de sueño afectó evaluaciones", "nominal", {1: "Sí", 2: "No", 3: "Tal vez"}, 0),
    "DESEMPENO_ACAD": ("P24", "Desempeño académico percibido", "ordinal",
                       {1: "Muy bajo", 2: "Bajo", 3: "Regular", 4: "Alto", 5: "Muy alto"}, 1),
    "INFLUENCIA_SUENO_DESEMPENO": ("P25", "Influencia del sueño en el desempeño", "ordinal",
                                   {1: "Nada", 2: "Poco", 3: "Moderadamente", 4: "Mucho", 5: "Demasiado"}, 0),
    "FACTOR_PRINCIPAL_SUENO": ("P26", "Principal factor que afecta el sueño", "nominal",
                               {1: "Carga académica", 2: "Dispositivos", 3: "Horarios de clase", 4: "Estrés",
                                5: "Cafeína / energizantes", 6: "Actividades sociales", 7: "Rutinas personales",
                                8: "Otro"}, 0),
}
VARS = list(CB)
ORD = [v for v in VARS if CB[v][2] == "ordinal"]
ET = {v: CB[v][3] for v in VARS}
LAB = {v: f"{CB[v][0]}. {CB[v][1]}" for v in VARS}
N_H = {1: 27, 2: 15, 3: 12, 4: 40, 5: 41, 6: 36, 7: 32, 8: 32}  # marco al 29/10/2026 (N = 235)
N_POB, N_PLAN = 235, 147
MIN_CELDA = 5  # privacidad: no se muestran resultados de grupos con menos de 5 respuestas

AZUL, AZUL_CL, NAR, GRIS, NAVY = "#2B64A8", "#8DB4E2", "#E07B2E", "#C9C8C2", "#14213D"
LIKERT = ["#B5432F", "#EBA393", "#D3D1CA", "#8DB4E2", "#2B64A8"]


def fmt(x, k=1):
    return f"{x:,.{k}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ------------------------------------------------------------------ datos
def preparar(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = [str(c).strip().upper() for c in raw.columns]
    faltan = [v for v in VARS if v not in raw.columns]
    if faltan:
        raise ValueError(f"Faltan columnas del diccionario: {', '.join(faltan)}")
    df = raw[[c for c in ["ID_ENCUESTA"] + VARS if c in raw.columns]].copy()
    for v in VARS:
        x = pd.to_numeric(df[v], errors="coerce")
        df[v] = x.where(x.isin(list(ET[v])))
    df = df.dropna(subset=VARS).copy()
    for v in VARS:
        df[v] = df[v].astype(int)
        df[v + "_ETQ"] = pd.Categorical(df[v].map(ET[v]), categories=list(ET[v].values()),
                                        ordered=CB[v][2] == "ordinal")
        if CB[v][4] == -1:
            df["R_" + v] = 6 - df[v]
        elif CB[v][4] == 1:
            df["R_" + v] = df[v]
    df["MENOS_7H"] = (df["HORAS_SUENO_SEM"] <= 3).astype(int)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cargar(contenido: bytes | None, nombre: str | None):
    import io
    if contenido is not None:
        fuente = io.BytesIO(contenido)
        etiqueta = nombre
    else:
        aqui = Path(__file__).parent
        candidatos = [aqui / "base_procesada.xlsx", aqui / "BaseDeDatos.xlsx", aqui / "data" / "base_procesada.xlsx"]
        ruta = next((p for p in candidatos if p.exists()), None)
        if ruta is None:
            return None, "sin archivo"
        fuente, etiqueta = ruta, ruta.name
    hojas = pd.ExcelFile(fuente).sheet_names
    hoja = "Base_datos" if "Base_datos" in hojas else hojas[0]
    return preparar(pd.read_excel(fuente, sheet_name=hoja)), etiqueta


# ------------------------------------------------------------------ estadística
def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    aj = np.empty(n)
    acc = 1.0
    for i in range(n - 1, -1, -1):
        acc = min(acc, p[o[i]] * n / (i + 1))
        aj[o[i]] = acc
    return aj


def fuerza(r):
    a = abs(r)
    return "muy débil" if a < .10 else "débil" if a < .30 else "moderada" if a < .50 else "fuerte"


@st.cache_data(show_spinner=False)
def spearman_ic(x: tuple, y: tuple, B: int = 2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(2026)
    boots = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) > 1 and len(np.unique(y[i])) > 1:
            boots.append(stats.spearmanr(x[i], y[i])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    tau = stats.kendalltau(x, y, variant="b")[0]
    return dict(rho=rho, p=p, lo=lo, hi=hi, tau=tau, n=len(x))


def pct(df, v, cods):
    return df[v].isin(cods).mean() * 100 if len(df) else np.nan


# ------------------------------------------------------------------ gráficos
def likert(df, variables, titulo):
    """Barras divergentes centradas en la categoría intermedia (la mitad a cada lado)."""
    filas = []
    for v in variables:
        pr = df[v].value_counts(normalize=True).reindex(range(1, 6), fill_value=0).values * 100
        if CB[v][4] != 1:  # frecuencia de algo negativo: "Siempre" es desfavorable
            pr = pr[::-1]
        filas.append((LAB[v], pr))
    y = [f for f, _ in filas]
    nombres = ["Muy desfavorable", "Desfavorable", "Intermedio", "Favorable", "Muy favorable"]
    fig = go.Figure()
    # Lado negativo: se apila desde 0 hacia la izquierda en el orden de las trazas.
    for j, signo, mitad in [(2, -1, True), (1, -1, False), (0, -1, False), (2, 1, True), (3, 1, False), (4, 1, False)]:
        val = [p[j] / (2 if mitad else 1) for _, p in filas]
        fig.add_bar(y=y, x=[signo * x for x in val], orientation="h", name=nombres[j], marker_color=LIKERT[j],
                    showlegend=not (j == 2 and signo == 1), legendgroup=nombres[j],
                    customdata=[p[j] for _, p in filas],
                    hovertemplate="%{y}<br>" + nombres[j] + ": %{customdata:.1f} %<extra></extra>")
    fig.update_layout(barmode="relative", title=titulo, height=110 + 46 * len(variables),
                      xaxis=dict(title="% (izquierda: desfavorable · derecha: favorable)", range=[-100, 100],
                                 tickvals=[-100, -50, 0, 50, 100], ticktext=["100", "50", "0", "50", "100"]),
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.25, traceorder="normal"),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def dist_100(df, v, por, titulo):
    t = pd.crosstab(df[por + "_ETQ"], df[v + "_ETQ"], normalize="index") * 100
    n = df[por + "_ETQ"].value_counts()
    t = t[[c for c in ET[v].values() if c in t.columns]]
    k = len(t.columns)
    paleta = [AZUL] if k == 1 else [px.colors.sample_colorscale([[0, "#E8EEF6"], [1, NAVY]], i / (k - 1))[0] for i in range(k)]
    fig = go.Figure()
    for c, col in zip(t.columns, paleta):
        fig.add_bar(y=[f"{i} (n={n.get(i, 0)})" for i in t.index], x=t[c], orientation="h", name=str(c), marker_color=col,
                    hovertemplate="%{y}<br>" + str(c) + ": %{x:.1f} %<extra></extra>")
    fig.update_layout(barmode="stack", title=titulo, xaxis=dict(title="% dentro de cada grupo", range=[0, 100]),
                      yaxis=dict(autorange="reversed"), height=110 + 42 * len(t), legend=dict(orientation="h", y=-0.25),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ------------------------------------------------------------------ interfaz
st.title("Hábitos de sueño y desempeño académico percibido")
st.caption("Tercer semestre de Ingeniería · Universidad de La Sabana · 2026-2 · Recolección de Datos")

with st.sidebar:
    st.header("Datos")
    sub = st.file_uploader("Base (.xlsx) — opcional", type=["xlsx"],
                           help="Si no se sube nada, se usa base_procesada.xlsx de la carpeta del app.")
    df_all, origen = cargar(sub.getvalue() if sub else None, sub.name if sub else None)
    if df_all is None:
        st.error("No se encontró la base. Sube el Excel con la hoja Base_datos o coloca base_procesada.xlsx junto a app.py.")
        st.stop()
    st.caption(f"Fuente: {origen} · {len(df_all)} respuestas válidas")
    st.header("Filtros")
    progs = st.multiselect("Programa", list(ET["PROGRAMA"].values()), default=list(ET["PROGRAMA"].values()))
    sexos = st.multiselect("Sexo", list(ET["SEXO"].values()), default=list(ET["SEXO"].values()))
    edades = st.multiselect("Edad", list(ET["EDAD"].values()), default=list(ET["EDAD"].values()))
    st.divider()
    st.caption("Muestra voluntaria y autoseleccionada: los resultados describen a quienes respondieron. "
               "Asociaciones, no causalidad. Por privacidad no se muestran grupos con menos de "
               f"{MIN_CELDA} respuestas.")

df = df_all[df_all["PROGRAMA_ETQ"].isin(progs) & df_all["SEXO_ETQ"].isin(sexos) & df_all["EDAD_ETQ"].isin(edades)]
n = len(df)
filtrado = n < len(df_all)
if n < MIN_CELDA:
    st.warning(f"La selección tiene {n} respuestas (menos de {MIN_CELDA}). Amplía los filtros para ver resultados.")
    st.stop()
if n < 20:
    st.info(f"La selección tiene solo {n} respuestas: los porcentajes y correlaciones son muy inestables.")

tabs = st.tabs(["Resumen", "Hábitos", "Calidad y desempeño", "Asociaciones", "Programas", "Método y datos"])

# ---------------------------------------------------------------- 1 Resumen
with tabs[0]:
    def kpi(col, titulo, valor, total, sufijo=" %", k=1):
        delta = None if not filtrado or np.isnan(valor) else f"{fmt(valor - total, k)} pp vs. total"
        col.metric(titulo, fmt(valor, k) + sufijo, delta, delta_color="off")

    c = st.columns(5)
    c[0].metric("Respuestas en la selección", f"{n}", f"de {len(df_all)} · marco {N_POB}", delta_color="off")
    kpi(c[1], "Duerme < 7 h entre semana", pct(df, "HORAS_SUENO_SEM", [1, 2, 3]), pct(df_all, "HORAS_SUENO_SEM", [1, 2, 3]))
    kpi(c[2], "Sueño malo o muy malo", pct(df, "CALIDAD_SUENO", [1, 2]), pct(df_all, "CALIDAD_SUENO", [1, 2]))
    kpi(c[3], "Dispositivos frec./siempre", pct(df, "FREC_DISPOSITIVOS", [4, 5]), pct(df_all, "FREC_DISPOSITIVOS", [4, 5]))
    kpi(c[4], "Desempeño alto o muy alto", pct(df, "DESEMPENO_ACAD", [4, 5]), pct(df_all, "DESEMPENO_ACAD", [4, 5]))

    r = spearman_ic(tuple(df["CALIDAD_SUENO"]), tuple(df["DESEMPENO_ACAD"]))
    st.subheader("Relación principal: calidad del sueño (P15) × desempeño percibido (P24)")
    if r:
        a, b = st.columns([1, 2])
        a.metric("rho de Spearman", fmt(r["rho"], 2), f"asociación {fuerza(r['rho'])}", delta_color="off")
        a.caption(f"IC 95 % bootstrap [{fmt(r['lo'], 2)}; {fmt(r['hi'], 2)}] · p = {fmt(r['p'], 3)} · "
                  f"tau-b = {fmt(r['tau'], 2)} · n = {r['n']}")
        tab = pd.crosstab(df["CALIDAD_SUENO_ETQ"], df["DESEMPENO_ACAD_ETQ"]).reindex(
            index=list(ET["CALIDAD_SUENO"].values()), columns=list(ET["DESEMPENO_ACAD"].values()), fill_value=0)
        fig = px.imshow(tab, text_auto=True, color_continuous_scale=[[0, "#F2F5FA"], [1, NAVY]], aspect="auto",
                        labels=dict(x="P24. Desempeño percibido", y="P15. Calidad del sueño", color="Estudiantes"))
        fig.update_yaxes(autorange=True)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        b.plotly_chart(fig, use_container_width=True)
    st.caption("Lectura: quienes califican mejor su sueño tienden a percibir mejor su desempeño. No implica que dormir mejor "
               "cause un mejor desempeño.")

# ---------------------------------------------------------------- 2 Hábitos
with tabs[1]:
    a, b = st.columns(2)
    sem = df["HORAS_SUENO_SEM"].value_counts(normalize=True).reindex(range(1, 7), fill_value=0) * 100
    fin = df["HORAS_SUENO_FIN"].value_counts(normalize=True).reindex(range(1, 7), fill_value=0) * 100
    fig = go.Figure()
    fig.add_bar(x=list(HORAS.values()), y=sem, name="Entre semana", marker_color=AZUL,
                text=[fmt(v, 0) for v in sem], textposition="outside")
    fig.add_bar(x=list(HORAS.values()), y=fin, name="Fin de semana", marker_color=NAR,
                text=[fmt(v, 0) for v in fin], textposition="outside")
    fig.update_layout(title="Horas de sueño por noche (franjas)", yaxis_title="% de estudiantes", barmode="group",
                      height=400, legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=50, b=10))
    a.plotly_chart(fig, use_container_width=True)

    sem = df["HR_DORMIR_SEM"].value_counts(normalize=True).reindex(range(1, 6), fill_value=0) * 100
    fin = df["HR_DORMIR_FIN"].value_counts(normalize=True).reindex(range(1, 6), fill_value=0) * 100
    fig = go.Figure()
    fig.add_bar(x=list(ACOST.values()), y=sem, name="Entre semana", marker_color=AZUL,
                text=[fmt(v, 0) for v in sem], textposition="outside")
    fig.add_bar(x=list(ACOST.values()), y=fin, name="Fin de semana", marker_color=NAR,
                text=[fmt(v, 0) for v in fin], textposition="outside")
    fig.update_layout(title="Hora de acostarse", yaxis_title="% de estudiantes", barmode="group", height=400,
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=50, b=10))
    b.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(likert(df, ["FREC_DISPOSITIVOS", "FREC_CARGA_ACAD", "FREC_CAFEINA", "FREC_SIESTA",
                                "REGULARIDAD_HORARIO"], "Factores asociados al sueño (P10–P14)"),
                    use_container_width=True)
    st.caption("En P10–P13, «Siempre» se muestra como desfavorable; en P14 (regularidad), «Siempre» es favorable. "
               "Las franjas son categorías, no horas exactas.")

# ---------------------------------------------------------------- 3 Calidad y desempeño
with tabs[2]:
    st.plotly_chart(likert(df, ["CALIDAD_SUENO", "FREC_CONCILIAR", "FREC_DESPERTAR_NOCHE", "FREC_SUENO_NO_REPARADOR",
                                "FREC_SOMNOLENCIA_DIA"], "Calidad del sueño y somnolencia (P15–P19)"),
                    use_container_width=True)
    st.plotly_chart(likert(df, ["FREC_DIFICULTAD_CONCENTRACION", "PRODUCTIVIDAD_ACAD", "FREC_ERRORES_CANSANCIO",
                                "DESEMPENO_ACAD"], "Desempeño académico percibido (P20–P24)"),
                    use_container_width=True)
    a, b = st.columns(2)
    t = df["FACTOR_PRINCIPAL_SUENO_ETQ"].value_counts().sort_values()
    fig = px.bar(x=t.values, y=t.index.astype(str), orientation="h", text=t.values,
                 labels=dict(x="Estudiantes", y=""), title="P26. Principal factor que afecta el sueño")
    fig.update_traces(marker_color=AZUL)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    a.plotly_chart(fig, use_container_width=True)
    t = df["AFECTO_EVALUACIONES_ETQ"].value_counts().reindex(list(ET["AFECTO_EVALUACIONES"].values()), fill_value=0)
    fig = px.bar(x=t.index.astype(str), y=t.values / n * 100, text=[fmt(v / n * 100) + " %" for v in t.values],
                 labels=dict(x="", y="% de estudiantes"), title="P23. ¿La falta de sueño afectó evaluaciones?")
    fig.update_traces(marker_color=[NAVY, GRIS, AZUL_CL])
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    b.plotly_chart(fig, use_container_width=True)
    b.caption("P23 presupone falta de sueño y puede sobrestimar el «Sí»; se interpreta con cautela.")

    with st.expander("Tabla de frecuencias de una pregunta"):
        v = st.selectbox("Pregunta", VARS, format_func=lambda x: LAB[x], key="freq")
        t = df[v + "_ETQ"].value_counts().reindex(list(ET[v].values()), fill_value=0)
        out = pd.DataFrame({"Categoría": t.index.astype(str), "n": t.values, "%": (t.values / n * 100).round(1)})
        if CB[v][2] == "ordinal":
            out["% acumulado"] = out["%"].cumsum().round(1)
        st.dataframe(out, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 4 Asociaciones
with tabs[3]:
    sueno = ["R_HORAS_SUENO_SEM", "R_REGULARIDAD_HORARIO", "R_CALIDAD_SUENO", "R_FREC_CONCILIAR",
             "R_FREC_DESPERTAR_NOCHE", "R_FREC_SUENO_NO_REPARADOR", "R_FREC_SOMNOLENCIA_DIA"]
    desem = ["R_FREC_DIFICULTAD_CONCENTRACION", "R_PRODUCTIVIDAD_ACAD", "R_FREC_ERRORES_CANSANCIO", "R_DESEMPENO_ACAD"]
    corto = {"R_HORAS_SUENO_SEM": "P8 Horas de sueño", "R_REGULARIDAD_HORARIO": "P14 Regularidad",
             "R_CALIDAD_SUENO": "P15 Calidad global", "R_FREC_CONCILIAR": "P16 Concilia sin dificultad",
             "R_FREC_DESPERTAR_NOCHE": "P17 Sin despertares", "R_FREC_SUENO_NO_REPARADOR": "P18 Sueño reparador",
             "R_FREC_SOMNOLENCIA_DIA": "P19 Sin somnolencia", "R_FREC_DIFICULTAD_CONCENTRACION": "P20 Concentración",
             "R_PRODUCTIVIDAD_ACAD": "P21 Productividad", "R_FREC_ERRORES_CANSANCIO": "P22 Sin errores",
             "R_DESEMPENO_ACAD": "P24 Desempeño"}
    filas = []
    for s in sueno:
        for d_ in desem:
            if df[s].nunique() > 1 and df[d_].nunique() > 1:
                rho, p = stats.spearmanr(df[s], df[d_])
            else:
                rho, p = np.nan, np.nan
            filas.append(dict(sueno=corto[s], desempeno=corto[d_], rho=rho, p=p))
    mat = pd.DataFrame(filas)
    ok = mat["p"].notna()
    mat.loc[ok, "q_BH"] = bh(mat.loc[ok, "p"])
    piv = mat.pivot(index="sueno", columns="desempeno", values="rho").loc[[corto[s] for s in sueno], [corto[d_] for d_ in desem]]
    qv = mat.pivot(index="sueno", columns="desempeno", values="q_BH").loc[piv.index, piv.columns]
    txt = piv.round(2).astype(str).replace("nan", "—") + np.where(qv < .05, "*", "")
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, zmin=-0.8, zmax=0.8, text=txt.values,
                               texttemplate="%{text}", colorscale=[[0, "#B5432F"], [0.5, "#F2F1EC"], [1, NAVY]],
                               colorbar=dict(title="rho")))
    fig.update_layout(title="Spearman sueño × desempeño (alto = mejor; * q < 0,05 Benjamini-Hochberg)",
                      height=460, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("P20 y P22 mencionan el cansancio o el sueño en su redacción, por lo que comparten contenido con las "
               "preguntas de sueño; la relación más limpia es P15 × P24.")

    st.subheader("Explorar cualquier par de preguntas ordinales")
    a, b = st.columns(2)
    vx = a.selectbox("Variable X", ORD, index=ORD.index("FREC_DISPOSITIVOS"), format_func=lambda x: LAB[x])
    vy = b.selectbox("Variable Y", ORD, index=ORD.index("CALIDAD_SUENO"), format_func=lambda x: LAB[x])
    r = spearman_ic(tuple(df[vx]), tuple(df[vy]))
    if vx == vy:
        st.info("Elige dos preguntas distintas.")
    elif r is None:
        st.info("No hay variación suficiente en la selección para calcular la correlación.")
    else:
        a, b = st.columns([1, 2])
        a.metric("rho de Spearman", fmt(r["rho"], 2), f"asociación {fuerza(r['rho'])}", delta_color="off")
        a.caption(f"IC 95 % bootstrap [{fmt(r['lo'], 2)}; {fmt(r['hi'], 2)}] · p = {fmt(r['p'], 3)} · n = {r['n']}. "
                  "Valor p sin ajustar por comparaciones múltiples.")
        a.caption("El signo se lee sobre los códigos originales: en preguntas de frecuencia, un valor alto es «más frecuente».")
        tab = pd.crosstab(df[vy + "_ETQ"], df[vx + "_ETQ"]).reindex(index=list(ET[vy].values()),
                                                                   columns=list(ET[vx].values()), fill_value=0)
        fig = px.imshow(tab, text_auto=True, color_continuous_scale=[[0, "#F2F5FA"], [1, NAVY]], aspect="auto",
                        labels=dict(x=LAB[vx], y=LAB[vy], color="Estudiantes"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        b.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 5 Programas
with tabs[4]:
    cnt = df_all["PROGRAMA"].value_counts().reindex(range(1, 9), fill_value=0)
    tabla = pd.DataFrame({"Programa": [ET["PROGRAMA"][i] for i in range(1, 9)],
                          "En el marco (Nₕ)": [N_H[i] for i in range(1, 9)],
                          "Cuota planeada": [17, 9, 7, 25, 26, 23, 20, 20],
                          "Respuestas": cnt.values})
    tabla["Tasa de respuesta %"] = (tabla["Respuestas"] / tabla["En el marco (Nₕ)"] * 100).round(1)
    tabla["Peso wₕ = Nₕ/nₕ"] = (tabla["En el marco (Nₕ)"] / tabla["Respuestas"].replace(0, np.nan)).round(2)
    a, b = st.columns([1.2, 1])
    a.subheader("Marco y muestra por programa (toda la base)")
    a.dataframe(tabla, hide_index=True, use_container_width=True)
    fig = go.Figure()
    fig.add_bar(y=tabla["Programa"], x=tabla["Tasa de respuesta %"], orientation="h", marker_color=AZUL,
                text=[fmt(v) + " %" for v in tabla["Tasa de respuesta %"]], textposition="outside")
    fig.update_layout(title="Tasa de respuesta por programa", xaxis=dict(range=[0, 100], title="%"),
                      yaxis=dict(autorange="reversed"), height=380, margin=dict(l=10, r=10, t=50, b=10))
    b.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribución de una pregunta por programa")
    v = st.selectbox("Pregunta", [x for x in VARS if x != "PROGRAMA"], index=[x for x in VARS if x != "PROGRAMA"].index("DESEMPENO_ACAD"),
                     format_func=lambda x: LAB[x], key="prog")
    grupos_ok = df["PROGRAMA_ETQ"].value_counts()
    dprog = df[df["PROGRAMA_ETQ"].isin(grupos_ok[grupos_ok >= MIN_CELDA].index)]
    if dprog["PROGRAMA"].nunique() >= 1:
        st.plotly_chart(dist_100(dprog, v, "PROGRAMA", f"{LAB[v]} por programa"), use_container_width=True)
    ocultos = grupos_ok[(grupos_ok > 0) & (grupos_ok < MIN_CELDA)]
    if len(ocultos):
        st.caption("Programas ocultos por tener menos de 5 respuestas en la selección: " + ", ".join(map(str, ocultos.index)))

    if dprog["PROGRAMA"].nunique() >= 2:
        filas = []
        for x in ORD:
            g = [grp[x].values for _, grp in dprog.groupby("PROGRAMA")]
            try:
                H, p = stats.kruskal(*g)
            except ValueError:  # todas las respuestas iguales
                continue
            k, nn = len(g), len(dprog)
            filas.append(dict(Pregunta=LAB[x], H=round(H, 2), gl=k - 1, p=p,
                              **{"ε²": round(max(0, (H - k + 1) / (nn - k)), 3)}))
        kw = pd.DataFrame(filas)
        kw["q (BH)"] = bh(kw["p"])
        kw["p"] = kw["p"].round(4)
        kw["q (BH)"] = kw["q (BH)"].round(4)
        st.subheader("Kruskal-Wallis entre programas")
        st.dataframe(kw.sort_values("q (BH)"), hide_index=True, use_container_width=True)
        st.caption("Con 10 a 12 respuestas por programa la potencia es limitada. Sin comparaciones post hoc salvo que "
                   "alguna prueba quede con q < 0,05 tras el ajuste.")

# ---------------------------------------------------------------- 6 Método y datos
with tabs[5]:
    a, b = st.columns(2)
    a.markdown(f"""
**Diseño.** Cuantitativo, transversal y descriptivo-correlacional.

**Marco.** Grupos de WhatsApp de la inducción 2025-2, fecha de corte 29/10/2026, sin inductores: N = {N_POB}.

**Muestra.** Planeada: {N_PLAN} (95 % de confianza, e = 5 %, p = 0,5) con afijación proporcional. Lograda: {len(df_all)}
respuestas voluntarias ({fmt(len(df_all) / N_POB * 100)} % del marco). La muestra es autoseleccionada y no probabilística.

**Instrumento.** Microsoft Forms, 26 preguntas cerradas obligatorias, revisado por el prof. Julián Obando.
Abierto del 22/09/2026 al 15/11/2026.

**Análisis.** Categorías ordenadas por rangos: mediana y RIC, Spearman con IC bootstrap, Kruskal-Wallis,
chi-cuadrado Monte Carlo, ajuste Benjamini-Hochberg. Ítems negativos invertidos (alto = mejor) solo en la matriz.
""")
    b.markdown("""
**Limitaciones.**
- Participación voluntaria: posible sesgo de no respuesta.
- Marco no oficial, sin contraste con Registro Académico.
- Sueño y desempeño son autopercepciones, no mediciones ni notas.
- Un experto revisó el formulario; no hubo piloto. P23 presupone falta de sueño y las franjas horarias comparten límites.
- Diseño transversal: asociaciones, no causas.

**Privacidad.** La base no tiene nombres, correos ni marcas de tiempo. No se muestran grupos con menos de 5 respuestas.
""")
    st.subheader("Datos de la selección")
    cols = ["ID_ENCUESTA"] if "ID_ENCUESTA" in df.columns else []
    vista = df[cols + [v + "_ETQ" for v in VARS]].rename(columns={v + "_ETQ": LAB[v] for v in VARS})
    st.dataframe(vista, hide_index=True, use_container_width=True, height=320)
    st.download_button("Descargar selección (CSV)", vista.to_csv(index=False).encode("utf-8-sig"),
                       "seleccion_sueno_desempeno.csv", "text/csv")
