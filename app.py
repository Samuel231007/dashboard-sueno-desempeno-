"""
De la noche al día · Dashboard de hallazgos
Hábitos de sueño y desempeño académico percibido — tercer semestre de Ingeniería,
Universidad de La Sabana, 2026-2 (Recolección de Datos).

Archivo único: diccionario, estadística (escalas sumadas, normalidad, ANOVA, correlaciones
no paramétricas), gráficos e interfaz. Solo necesita este archivo, requirements.txt y la base.

Ejecutar:  pip install -r requirements.txt  &&  streamlit run app.py
"""
import html as _html
import io
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402



# ##################################################################################################
# DICCIONARIO, DATOS Y ESTADÍSTICA BÁSICA
# ##################################################################################################
FREC = {1: "Nunca", 2: "Rara vez", 3: "Algunas veces", 4: "Frecuentemente", 5: "Siempre"}
ACOST = {1: "Antes de 9 p. m.", 2: "9–10 p. m.", 3: "10–11 p. m.", 4: "11 p. m.–12 a. m.", 5: "Después de 12 a. m."}
HORAS = {1: "< 4 h", 2: "4 a < 6 h", 3: "6 a < 7 h", 4: "7 a < 8 h", 5: "8 a < 9 h", 6: "≥ 9 h"}

# variable: (pregunta, etiqueta corta, escala, códigos, dirección)  dirección: 1 alto=mejor, -1 alto=peor, 0 n/a
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
    "FREC_DISPOSITIVOS": ("P11", "Pantallas antes de dormir", "ordinal", FREC, -1),
    "FREC_CAFEINA": ("P12", "Cafeína después de 6 p. m.", "ordinal", FREC, -1),
    "FREC_CARGA_ACAD": ("P13", "Carga académica altera el horario", "ordinal", FREC, -1),
    "REGULARIDAD_HORARIO": ("P14", "Horarios regulares", "ordinal", FREC, 1),
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
                               {1: "Carga académica", 2: "Pantallas / dispositivos", 3: "Horarios de clase",
                                4: "Estrés o preocupaciones", 5: "Cafeína / energizantes", 6: "Actividades sociales",
                                7: "Rutinas personales", 8: "Otro"}, 0),
}
VARS = list(CB)
ORD = [v for v in VARS if CB[v][2] == "ordinal"]
ET = {v: CB[v][3] for v in VARS}
LAB = {v: f"{CB[v][0]}. {CB[v][1]}" for v in VARS}

N_H = {1: 27, 2: 15, 3: 12, 4: 40, 5: 41, 6: 36, 7: 32, 8: 32}   # marco al 29/10/2026
CUOTA = {1: 17, 2: 9, 3: 7, 4: 25, 5: 26, 6: 23, 7: 20, 8: 20}  # afijación proporcional de n = 147
N_POB, N_PLAN = 235, 147

# Pares sueño (orientados, alto = mejor) × desempeño (orientados) usados en el informe
SUENO_R = [("HORAS_SUENO_SEM", "P8 Horas de sueño"), ("REGULARIDAD_HORARIO", "P14 Regularidad"),
           ("CALIDAD_SUENO", "P15 Calidad global"), ("FREC_CONCILIAR", "P16 Concilia sin dificultad"),
           ("FREC_DESPERTAR_NOCHE", "P17 Sin despertares"), ("FREC_SUENO_NO_REPARADOR", "P18 Sueño reparador"),
           ("FREC_SOMNOLENCIA_DIA", "P19 Sin somnolencia")]
DESEMP_R = [("FREC_DIFICULTAD_CONCENTRACION", "P20 Concentración"), ("PRODUCTIVIDAD_ACAD", "P21 Productividad"),
            ("FREC_ERRORES_CANSANCIO", "P22 Sin errores"), ("DESEMPENO_ACAD", "P24 Desempeño")]
HABITOS = [("FREC_DISPOSITIVOS", "Pantallas antes de dormir"), ("FREC_CARGA_ACAD", "Carga académica altera el horario"),
           ("FREC_CAFEINA", "Cafeína después de 6 p. m."), ("HR_DORMIR_SEM", "Acostarse más tarde (semana)"),
           ("FREC_SIESTA", "Siestas")]


def fmt(x, k=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.{k}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_p(p):
    return "< 0,001" if p < 0.001 else fmt(p, 3)


def p_txt(p):
    """«< 0,001» o «= 0,012», para escribir «p …» o «q …»."""
    return "< 0,001" if p < 0.001 else "= " + fmt(p, 3)


# ------------------------------------------------------------------ datos
def preparar(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = [str(c).strip().upper() for c in raw.columns]
    faltan = [v for v in VARS if v not in raw.columns]
    if faltan:
        raise ValueError("Faltan columnas del diccionario: " + ", ".join(faltan))
    cols = (["ID_ENCUESTA"] if "ID_ENCUESTA" in raw.columns else []) + VARS
    df = raw[cols].copy()
    for v in VARS:
        x = pd.to_numeric(df[v], errors="coerce")
        df[v] = x.where(x.isin(list(ET[v])))
    df = df.dropna(subset=VARS).copy()
    for v in VARS:
        df[v] = df[v].astype(int)
        df[v + "_ETQ"] = pd.Categorical(df[v].map(ET[v]), categories=list(ET[v].values()),
                                        ordered=CB[v][2] == "ordinal")
        if CB[v][4] == -1:
            df["R_" + v] = (min(ET[v]) + max(ET[v])) - df[v]
        elif CB[v][4] == 1:
            df["R_" + v] = df[v]
    return df.reset_index(drop=True)


def leer_excel(fuente) -> pd.DataFrame:
    if isinstance(fuente, (bytes, bytearray)):
        fuente = io.BytesIO(fuente)
    with pd.ExcelFile(fuente) as xls:
        hoja = "Base_datos" if "Base_datos" in xls.sheet_names else xls.sheet_names[0]
        raw = pd.read_excel(xls, sheet_name=hoja)
    return preparar(raw)


def buscar_base(carpeta: Path):
    for nombre in ["base_procesada.xlsx", "BaseDeDatos.xlsx", "data/base_procesada.xlsx"]:
        p = carpeta / nombre
        if p.exists():
            return p
    return None


# ------------------------------------------------------------------ estadística
def pct(df, v, cods):
    return float(df[v].isin(cods).mean() * 100) if len(df) else float("nan")


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


def spearman_fz(x, y):
    """rho de Spearman con IC 95 % por Fisher z y error de Bonett-Wright (rápido, para muchas parejas)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 6 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return dict(rho=np.nan, p=np.nan, lo=np.nan, hi=np.nan, n=n)
    rho, p = stats.spearmanr(x, y)
    rr = float(np.clip(rho, -0.999, 0.999))
    z, se = np.arctanh(rr), np.sqrt((1 + rr ** 2 / 2) / (n - 3))
    return dict(rho=float(rho), p=float(p), lo=float(np.tanh(z - 1.96 * se)), hi=float(np.tanh(z + 1.96 * se)), n=n)


def spearman_boot(x, y, B=2000, seed=2026):
    """rho de Spearman con IC 95 % bootstrap percentil (como en el notebook) y tau-b de Kendall."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    base = spearman_fz(x, y)
    if np.isnan(base["rho"]):
        return dict(base, tau=np.nan)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if np.unique(x[i]).size > 1 and np.unique(y[i]).size > 1:
            boots.append(stats.spearmanr(x[i], y[i])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    tau = stats.kendalltau(x, y, variant="b")[0]
    return dict(base, lo=float(lo), hi=float(hi), tau=float(tau))


def matriz_sueno_desempeno(df):
    filas = []
    for s, sl in SUENO_R:
        for d, dl in DESEMP_R:
            r = spearman_fz(df["R_" + s], df["R_" + d])
            filas.append(dict(sueno=sl, desempeno=dl, **r))
    m = pd.DataFrame(filas)
    ok = m["p"].notna()
    m["q"] = np.nan
    if ok.any():
        m.loc[ok, "q"] = bh(m.loc[ok, "p"])
    return m


def habitos_tabla(df):
    filas = []
    for v, lab in HABITOS:
        for dest, dl in [("CALIDAD_SUENO", "Calidad del sueño"), ("DESEMPENO_ACAD", "Desempeño percibido")]:
            r = spearman_fz(df[v], df[dest])
            filas.append(dict(habito=lab, resultado=dl, **r))
    t = pd.DataFrame(filas)
    ok = t["p"].notna()
    t["q"] = np.nan
    if ok.any():
        t.loc[ok, "q"] = bh(t.loc[ok, "p"])
    return t


def kruskal_programas(df, min_n=5):
    cnt = df["PROGRAMA"].value_counts()
    d = df[df["PROGRAMA"].isin(cnt[cnt >= min_n].index)]
    k = d["PROGRAMA"].nunique()
    if k < 2:
        return pd.DataFrame()
    filas = []
    for v in ORD:
        g = [grp[v].values for _, grp in d.groupby("PROGRAMA")]
        try:
            H, p = stats.kruskal(*g)
        except ValueError:
            continue
        nn = len(d)
        filas.append(dict(variable=v, pregunta=LAB[v], H=H, gl=k - 1, p=p, eps2=max(0.0, (H - k + 1) / (nn - k))))
    t = pd.DataFrame(filas)
    if len(t):
        t["q"] = bh(t["p"])
    return t


def chi2_3x3(df):
    c3 = pd.cut(df["CALIDAD_SUENO"], [0, 2, 3, 5], labels=["Bajo", "Regular", "Alto"])
    d3 = pd.cut(df["DESEMPENO_ACAD"], [0, 2, 3, 5], labels=["Bajo", "Regular", "Alto"])
    tab = pd.crosstab(c3, d3)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    chi2, p, gl, _ = stats.chi2_contingency(tab, correction=False)
    v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
    return dict(chi2=chi2, p=p, gl=gl, v=v)


def ponderado(df):
    cnt = df["PROGRAMA"].value_counts()
    if any(cnt.get(k, 0) == 0 for k in N_H):
        return None, None
    w = df["PROGRAMA"].map({k: N_H[k] / cnt[k] for k in N_H}).values
    ind = [("Menos de 7 h entre semana", "HORAS_SUENO_SEM", [1, 2, 3]),
           ("Pantallas frecuente/siempre", "FREC_DISPOSITIVOS", [4, 5]),
           ("Carga académica frecuente/siempre", "FREC_CARGA_ACAD", [4, 5]),
           ("Sueño malo o muy malo", "CALIDAD_SUENO", [1, 2]),
           ("Somnolencia frecuente/siempre", "FREC_SOMNOLENCIA_DIA", [4, 5]),
           ("Desempeño alto o muy alto", "DESEMPENO_ACAD", [4, 5]),
           ("Desempeño bajo o muy bajo", "DESEMPENO_ACAD", [1, 2])]
    filas = []
    for lab, v, c in ind:
        x = df[v].isin(c).values
        filas.append(dict(indicador=lab, sin=x.mean() * 100, pond=np.average(x, weights=w) * 100))
    t = pd.DataFrame(filas)
    t["dif"] = t["pond"] - t["sin"]

    def rangos(vals):
        vals = np.asarray(vals)
        out = np.zeros(len(vals))
        for val in np.unique(vals):
            m = vals == val
            out[m] = w[vals < val].sum() + w[m].sum() / 2
        return out
    rx, ry = rangos(df["CALIDAD_SUENO"]), rangos(df["DESEMPENO_ACAD"])
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    rho_w = np.average((rx - mx) * (ry - my), weights=w) / np.sqrt(
        np.average((rx - mx) ** 2, weights=w) * np.average((ry - my) ** 2, weights=w))
    return t, float(rho_w)


def cronbach(X):
    X = X.astype(float)
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))

# ##################################################################################################
# ESCALAS SUMADAS, NORMALIDAD, ANOVA Y CORRELACIONES
# ##################################################################################################
DIMENSIONES = {
    "CALIDAD": dict(nombre="Calidad del sueño", corto="Calidad del sueño",
                    items=["CALIDAD_SUENO", "FREC_CONCILIAR", "FREC_DESPERTAR_NOCHE", "FREC_SUENO_NO_REPARADOR"],
                    preguntas="P15–P18", rango=(4, 20)),
    "FUNCIONAMIENTO": dict(nombre="Funcionamiento académico percibido", corto="Funcionamiento académico",
                           items=["FREC_SOMNOLENCIA_DIA", "FREC_DIFICULTAD_CONCENTRACION", "PRODUCTIVIDAD_ACAD",
                                  "FREC_ERRORES_CANSANCIO", "DESEMPENO_ACAD"],
                           preguntas="P19–P22 y P24", rango=(5, 25)),
}
# Conjunto evaluado y descartado como escala (consistencia insuficiente): se analiza ítem por ítem.
HABITOS_ITEMS = ["FREC_DISPOSITIVOS", "FREC_CAFEINA", "FREC_CARGA_ACAD", "REGULARIDAD_HORARIO"]

# Variables de agrupación para el análisis bivariado (categorías agrupadas para tener grupos de tamaño útil)
GRUPOS = {
    "HORAS": ("Horas de sueño entre semana (P8)", "HORAS_SUENO_SEM",
              {1: "Menos de 6 h", 2: "Menos de 6 h", 3: "6 a < 7 h", 4: "7 h o más", 5: "7 h o más", 6: "7 h o más"},
              ["Menos de 6 h", "6 a < 7 h", "7 h o más"]),
    "PANTALLAS": ("Pantallas antes de dormir (P11)", "FREC_DISPOSITIVOS",
                  {1: "Nunca o rara vez", 2: "Nunca o rara vez", 3: "Algunas veces", 4: "Frecuente o siempre",
                   5: "Frecuente o siempre"}, ["Nunca o rara vez", "Algunas veces", "Frecuente o siempre"]),
    "CARGA": ("Carga académica altera el horario (P13)", "FREC_CARGA_ACAD",
              {1: "Nunca o rara vez", 2: "Nunca o rara vez", 3: "Algunas veces", 4: "Frecuente o siempre",
               5: "Frecuente o siempre"}, ["Nunca o rara vez", "Algunas veces", "Frecuente o siempre"]),
    "CAFEINA": ("Cafeína después de 6 p. m. (P12)", "FREC_CAFEINA",
                {1: "Nunca o rara vez", 2: "Nunca o rara vez", 3: "Algunas veces", 4: "Frecuente o siempre",
                 5: "Frecuente o siempre"}, ["Nunca o rara vez", "Algunas veces", "Frecuente o siempre"]),
    "REGULARIDAD": ("Horarios regulares (P14)", "REGULARIDAD_HORARIO",
                    {1: "Nunca o rara vez", 2: "Nunca o rara vez", 3: "Algunas veces", 4: "Frecuente o siempre",
                     5: "Frecuente o siempre"}, ["Nunca o rara vez", "Algunas veces", "Frecuente o siempre"]),
    "ACOSTARSE": ("Hora de acostarse entre semana (P4)", "HR_DORMIR_SEM",
                  {1: "Antes de 11 p. m.", 2: "Antes de 11 p. m.", 3: "Antes de 11 p. m.", 4: "11 p. m.–12 a. m.",
                   5: "Después de 12 a. m."}, ["Antes de 11 p. m.", "11 p. m.–12 a. m.", "Después de 12 a. m."]),
    "PROGRAMA": ("Programa (P1)", "PROGRAMA", ET["PROGRAMA"], list(ET["PROGRAMA"].values())),
}


def orientado(df, v):
    return df["R_" + v] if "R_" + v in df.columns else df[v]


def cronbach(X):
    X = X.astype(float)
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))


def agregar_puntajes(df):
    df = df.copy()
    for k, d in DIMENSIONES.items():
        df["PUNTAJE_" + k] = sum(orientado(df, v) for v in d["items"])
    for g, (_, var, mapa, orden) in GRUPOS.items():
        df["G_" + g] = pd.Categorical(df[var].map(mapa), categories=orden, ordered=g != "PROGRAMA")
    return df


# ------------------------------------------------------------------ consistencia
def consistencia(df):
    filas, items = [], []
    conjuntos = {k: (d["nombre"], d["items"]) for k, d in DIMENSIONES.items()}
    conjuntos["HABITOS"] = ("Hábitos de sueño (evaluado, no se suma)", HABITOS_ITEMS)
    for k, (nombre, its) in conjuntos.items():
        X = pd.concat([orientado(df, v).rename(v) for v in its], axis=1)
        a = cronbach(X)
        filas.append(dict(clave=k, dimension=nombre, k=len(its), alfa=a,
                          decision="Se suma (α ≥ 0,70)" if a >= .70 else "No se suma: se analiza ítem por ítem"))
        for v in its:
            resto = X.drop(columns=v)
            items.append(dict(clave=k, dimension=nombre, item=f"{CB[v][0]}. {CB[v][1]}",
                              invertido=CB[v][4] == -1,
                              r_item_total=stats.pearsonr(X[v], resto.sum(axis=1))[0],
                              alfa_sin_item=cronbach(resto) if resto.shape[1] > 1 else np.nan))
    return pd.DataFrame(filas), pd.DataFrame(items)


# ------------------------------------------------------------------ descriptivos y normalidad
def descriptivos(df):
    filas = []
    for k, d in DIMENSIONES.items():
        s = df["PUNTAJE_" + k]
        q1, q3 = np.percentile(s, [25, 75])
        sw = stats.shapiro(s)
        filas.append(dict(clave=k, dimension=d["nombre"], preguntas=d["preguntas"],
                          rango_posible=f"{d['rango'][0]}–{d['rango'][1]}", n=len(s), media=s.mean(), de=s.std(ddof=1),
                          mediana=s.median(), q1=q1, q3=q3, minimo=s.min(), maximo=s.max(),
                          asimetria=stats.skew(s, bias=False), curtosis=stats.kurtosis(s, bias=False),
                          shapiro_W=sw.statistic, shapiro_p=sw.pvalue,
                          normal="Sí (p ≥ 0,05)" if sw.pvalue >= .05 else "No (p < 0,05)"))
    return pd.DataFrame(filas)


def normalidad_por_grupo(df, grupo="PROGRAMA"):
    filas = []
    for k, d in DIMENSIONES.items():
        for g, sub in df.groupby("G_" + grupo, observed=True):
            s = sub["PUNTAJE_" + k]
            if len(s) >= 3 and s.nunique() > 1:
                sw = stats.shapiro(s)
                filas.append(dict(dimension=d["corto"], grupo=str(g), n=len(s), W=sw.statistic, p=sw.pvalue))
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ ANOVA y alternativas
def welch_anova(grupos):
    k = len(grupos)
    n = np.array([len(g) for g in grupos], float)
    m = np.array([np.mean(g) for g in grupos])
    v = np.array([np.var(g, ddof=1) for g in grupos])
    w = n / v
    mw = np.sum(w * m) / np.sum(w)
    A = np.sum(w * (m - mw) ** 2) / (k - 1)
    lam = np.sum((1 - w / np.sum(w)) ** 2 / (n - 1))
    B = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    F = A / B
    df2 = (k ** 2 - 1) / (3 * lam)
    return F, k - 1, df2, stats.f.sf(F, k - 1, df2)


def anova(df, k_dim, grupo, min_n=3):
    y = "PUNTAJE_" + k_dim
    d = df[["G_" + grupo, y]].dropna()
    cnt = d["G_" + grupo].value_counts()
    validos = [g for g in d["G_" + grupo].cat.categories if cnt.get(g, 0) >= min_n]
    d = d[d["G_" + grupo].isin(validos)]
    grupos = [d.loc[d["G_" + grupo] == g, y].values for g in validos]
    if len(grupos) < 2:
        return None
    F, p = stats.f_oneway(*grupos)
    N, k = len(d), len(grupos)
    ss_b = sum(len(g) * (g.mean() - d[y].mean()) ** 2 for g in grupos)
    ss_t = ((d[y] - d[y].mean()) ** 2).sum()
    ss_w = ss_t - ss_b
    ms_w = ss_w / (N - k)
    eta2 = ss_b / ss_t
    omega2 = max(0.0, (ss_b - (k - 1) * ms_w) / (ss_t + ms_w))
    lev = stats.levene(*grupos, center="median")
    resid = np.concatenate([g - g.mean() for g in grupos])
    sw_res = stats.shapiro(resid)
    Fw, d1w, d2w, pw = welch_anova(grupos)
    H, pk = stats.kruskal(*grupos)
    medias = {str(g): (float(x.mean()), float(x.std(ddof=1)), len(x)) for g, x in zip(validos, grupos)}
    res = dict(dimension=DIMENSIONES[k_dim]["corto"], grupo=GRUPOS[grupo][0], k=k, N=N,
               F=F, gl1=k - 1, gl2=N - k, p=p, eta2=eta2, omega2=omega2,
               levene_W=lev.statistic, levene_p=lev.pvalue, shapiro_resid_p=sw_res.pvalue,
               welch_F=Fw, welch_gl2=d2w, welch_p=pw, kruskal_H=H, kruskal_p=pk, medias=medias)
    res["prueba_principal"] = "ANOVA clásico" if lev.pvalue >= .05 else "ANOVA de Welch"
    res["p_principal"] = p if lev.pvalue >= .05 else pw
    res["tukey"] = None
    if res["p_principal"] < .05:
        t = stats.tukey_hsd(*grupos)
        filas = []
        for i in range(k):
            for j in range(i + 1, k):
                filas.append(dict(grupo_a=str(validos[i]), grupo_b=str(validos[j]),
                                  diferencia=float(grupos[i].mean() - grupos[j].mean()), p=float(t.pvalue[i, j])))
        res["tukey"] = pd.DataFrame(filas)
    return res


def tabla_anovas(df, grupos=("PROGRAMA", "HORAS", "ACOSTARSE", "PANTALLAS", "CARGA", "CAFEINA", "REGULARIDAD")):
    filas = []
    for g in grupos:
        for k in DIMENSIONES:
            r = anova(df, k, g)
            if r:
                filas.append({kk: vv for kk, vv in r.items() if kk not in ("medias", "tukey")})
    return pd.DataFrame(filas)


def welch_t(df, k_dim, var="SEXO", a=1, b=2):
    y = "PUNTAJE_" + k_dim
    x1, x2 = df.loc[df[var] == a, y], df.loc[df[var] == b, y]
    if len(x1) < 3 or len(x2) < 3:
        return None
    t = stats.ttest_ind(x1, x2, equal_var=False)
    sp = np.sqrt(((len(x1) - 1) * x1.var() + (len(x2) - 1) * x2.var()) / (len(x1) + len(x2) - 2))
    u = stats.mannwhitneyu(x1, x2, alternative="two-sided")
    return dict(dimension=DIMENSIONES[k_dim]["corto"], media_fem=x1.mean(), media_masc=x2.mean(), n_fem=len(x1),
                n_masc=len(x2), t=t.statistic, p=t.pvalue, d_cohen=(x1.mean() - x2.mean()) / sp, mw_p=u.pvalue)


# ------------------------------------------------------------------ correlaciones no paramétricas
def correlaciones(df):
    a, b = df["PUNTAJE_CALIDAD"], df["PUNTAJE_FUNCIONAMIENTO"]
    sp = stats.spearmanr(a, b)
    kt = stats.kendalltau(a, b, variant="b")
    pe = stats.pearsonr(a, b)
    n = len(a)
    z, se = np.arctanh(sp.statistic), np.sqrt((1 + sp.statistic ** 2 / 2) / (n - 3))
    return dict(n=n, spearman=sp.statistic, spearman_p=sp.pvalue, sp_lo=np.tanh(z - 1.96 * se),
                sp_hi=np.tanh(z + 1.96 * se), kendall=kt.statistic, kendall_p=kt.pvalue,
                pearson=pe.statistic, pearson_p=pe.pvalue)


def matriz_no_parametrica(df):
    vars_ = [("PUNTAJE_CALIDAD", "Calidad del sueño (puntaje)"), ("PUNTAJE_FUNCIONAMIENTO", "Funcionamiento académico (puntaje)"),
             ("R_HORAS_SUENO_SEM", "P8 Horas de sueño"), ("REGULARIDAD_HORARIO", "P14 Horarios regulares"),
             ("FREC_DISPOSITIVOS", "P11 Pantallas"), ("FREC_CARGA_ACAD", "P13 Carga académica"),
             ("FREC_CAFEINA", "P12 Cafeína")]
    filas = []
    for i, (v1, l1) in enumerate(vars_):
        for v2, l2 in vars_[i + 1:]:
            s = stats.spearmanr(df[v1], df[v2])
            t = stats.kendalltau(df[v1], df[v2], variant="b")
            filas.append(dict(var1=l1, var2=l2, spearman=s.statistic, p_spearman=s.pvalue, kendall=t.statistic,
                              p_kendall=t.pvalue))
    m = pd.DataFrame(filas)
    m["q_BH"] = bh(m["p_spearman"])
    return m, vars_

# ##################################################################################################
# GRÁFICOS (matplotlib)
# ##################################################################################################
# Paleta «De la noche al día»
NOCHE, INDIGO, LAVANDA, DORADO, ORO_CL = "#141D44", "#2F3F86", "#8E9BE6", "#E9A93B", "#F2C46D"
TINTA, GRIS_T, BORDE, PAPEL, CREMA = "#18233F", "#5B6478", "#E6DECD", "#FFFDF8", "#F1E9D8"
ROJO, ROJO_CL, NEUTRO = "#B5432F", "#EBA393", "#D8D0BF"
LIKERT = [ROJO, ROJO_CL, NEUTRO, LAVANDA, INDIGO]
SEQ = LinearSegmentedColormap.from_list("noche", ["#F6F1E6", "#C7CDEB", LAVANDA, INDIGO, NOCHE])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": BORDE, "axes.labelcolor": GRIS_T,
    "xtick.color": GRIS_T, "ytick.color": GRIS_T, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": TINTA, "axes.titlelocation": "left",
    "axes.titlepad": 26, "figure.facecolor": PAPEL, "axes.facecolor": PAPEL, "savefig.facecolor": PAPEL,
    "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 150,
})


def _fig(w=10, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.16)
    return fig, ax


def _ygrid(ax):
    ax.yaxis.grid(True, color=BORDE, lw=0.8)
    ax.set_axisbelow(True)


def _xgrid(ax):
    ax.xaxis.grid(True, color=BORDE, lw=0.8)
    ax.set_axisbelow(True)


def _sub(ax, txt):
    ax.text(0, 1.015, txt, transform=ax.transAxes, fontsize=9, color=GRIS_T, va="bottom")


# ------------------------------------------------------------------ la noche
def barras_semana_fin(df, v_sem, v_fin, etiquetas, titulo, nota, w=6.4, h=4.4):
    k = len(etiquetas)
    sem = df[v_sem].value_counts(normalize=True).reindex(range(1, k + 1), fill_value=0) * 100
    fin = df[v_fin].value_counts(normalize=True).reindex(range(1, k + 1), fill_value=0) * 100
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.2)
    x = np.arange(k)
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, sem, w, color=INDIGO, label="Entre semana")
    b2 = ax.bar(x + w / 2 + 0.01, fin, w, color=DORADO, label="Fin de semana")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, fmt(b.get_height(), 0),
                    ha="center", va="bottom", fontsize=8, color=TINTA)
    ax.set_xticks(x, list(etiquetas), rotation=20 if k > 5 or w < 8 else 0, ha="right" if w < 8 else "center", fontsize=8.5)
    ax.set_ylabel("% de estudiantes")
    ax.set_ylim(0, max(sem.max(), fin.max()) * 1.2 + 2)
    _ygrid(ax)
    ax.set_title(titulo)
    _sub(ax, nota)
    ax.legend(loc="upper right", ncol=2)
    return fig


def horas(df):
    return barras_semana_fin(df, "HORAS_SUENO_SEM", "HORAS_SUENO_FIN", HORAS.values(),
                             "Horas de sueño por noche", f"P8 y P9 · franjas, no horas exactas · n = {len(df)}")


def acostarse(df):
    return barras_semana_fin(df, "HR_DORMIR_SEM", "HR_DORMIR_FIN", ACOST.values(),
                             "Hora de acostarse", f"P4 y P6 · n = {len(df)}")


def transicion(df):
    t = pd.crosstab(df["HORAS_SUENO_SEM"], df["HORAS_SUENO_FIN"]).reindex(index=range(1, 7), columns=range(1, 7),
                                                                         fill_value=0)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.84, bottom=0.2)
    ax.imshow(t.values, cmap=SEQ, origin="lower", aspect="auto")
    mx = max(t.values.max(), 1)
    for i in range(6):
        for j in range(6):
            v = t.values[i, j]
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=9, color="white" if v > mx * 0.55 else TINTA)
    for k in range(6):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec=DORADO, lw=1.6))
    lab = list(HORAS.values())
    ax.set_xticks(range(6), lab, rotation=30, ha="right")
    ax.set_yticks(range(6), lab)
    ax.set_xlabel("Fin de semana (P9)")
    ax.set_ylabel("Entre semana (P8)")
    for s in ax.spines.values():
        s.set_visible(False)
    arriba = int(np.triu(t.values, 1).sum())
    ax.set_title("¿Se recupera sueño el fin de semana?")
    _sub(ax, f"{arriba} de {int(t.values.sum())} duermen una franja más el fin de semana · borde dorado = igual")
    return fig


def likert(df, variables, titulo, nota=""):
    filas = []
    for v in variables:
        pr = df[v].value_counts(normalize=True).reindex(range(1, 6), fill_value=0).values * 100
        if CB[v][4] != 1:
            pr = pr[::-1]
        filas.append((LAB[v], pr))
    h = 0.75 + 0.48 * len(filas)
    fig, ax = plt.subplots(figsize=(10, h + 0.9))
    fig.subplots_adjust(left=0.30, right=0.98, top=1 - 0.8 / (h + 0.9), bottom=0.95 / (h + 0.9))
    for i, (lab, p) in enumerate(filas):
        izq = -(p[0] + p[1] + p[2] / 2)
        for j in range(5):
            ax.barh(i, p[j], left=izq, color=LIKERT[j], height=0.62, edgecolor=PAPEL, lw=1.2)
            if p[j] >= 8:
                ax.text(izq + p[j] / 2, i, fmt(p[j], 0), ha="center", va="center", fontsize=8.5,
                        color="white" if j in (0, 4) else TINTA)
            izq += p[j]
    ax.set_yticks(range(len(filas)), [f for f, _ in filas], fontsize=9.5, color=TINTA)
    ax.invert_yaxis()
    ax.axvline(0, color=GRIS_T, lw=0.8)
    ax.set_xlim(-100, 100)
    ax.set_xticks([-100, -50, 0, 50, 100], ["100 %", "50 %", "0", "50 %", "100 %"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(titulo)
    _sub(ax, nota or "Izquierda: respuestas desfavorables · derecha: favorables · centrado en la categoría intermedia")
    fig.legend([Patch(color=c) for c in LIKERT],
               ["Muy desfavorable", "Desfavorable", "Intermedio", "Favorable", "Muy favorable"],
               ncol=5, loc="lower center", bbox_to_anchor=(0.64, 0.01), fontsize=8.5)
    return fig


def factor_principal(df):
    t = df["FACTOR_PRINCIPAL_SUENO_ETQ"].value_counts().sort_values()
    t = t[t > 0]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    fig.subplots_adjust(left=0.36, right=0.92, top=0.84, bottom=0.1)
    cols = [DORADO if i >= len(t) - 3 else LAVANDA for i in range(len(t))]
    ax.barh(t.index.astype(str), t.values, color=cols, height=0.62)
    for i, v in enumerate(t.values):
        ax.text(v + 0.3, i, f"{v} · {fmt(v / len(df) * 100, 0)} %", va="center", fontsize=8.5, color=TINTA)
    ax.set_xlim(0, t.max() * 1.35)
    _xgrid(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Principal factor que afecta el sueño")
    _sub(ax, f"P26 · una opción por estudiante · dorado = los tres más citados")
    return fig


def p23(df):
    t = df["AFECTO_EVALUACIONES_ETQ"].value_counts().reindex(list(ET["AFECTO_EVALUACIONES"].values()), fill_value=0)
    pc = t / len(df) * 100
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.12)
    b = ax.bar(pc.index.astype(str), pc.values, color=[INDIGO, NEUTRO, LAVANDA], width=0.55)
    for r in b:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1, fmt(r.get_height(), 1) + " %", ha="center",
                fontsize=9, color=TINTA)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% de estudiantes")
    _ygrid(ax)
    ax.set_title("¿La falta de sueño afectó evaluaciones?")
    _sub(ax, "P23 · la pregunta presupone falta de sueño: leer con cautela")
    return fig


# ------------------------------------------------------------------ asociaciones
def asociacion_principal(df):
    c3 = pd.cut(df["CALIDAD_SUENO"], [0, 2, 3, 5], labels=["Mala o muy mala", "Regular", "Buena o muy buena"])
    d3 = pd.cut(df["DESEMPENO_ACAD"], [0, 2, 3, 5], labels=["Bajo", "Regular", "Alto"])
    t = pd.crosstab(c3, d3, normalize="index").reindex(columns=["Bajo", "Regular", "Alto"], fill_value=0) * 100
    nn = c3.value_counts().reindex(t.index)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.8, bottom=0.27)
    cols = [ROJO_CL, NEUTRO, INDIGO]
    for i, (idx, row) in enumerate(t.iterrows()):
        izq = 0
        for c, col in zip(t.columns, cols):
            ax.barh(i, row[c], left=izq, color=col, height=0.6, edgecolor=PAPEL, lw=1.5)
            if row[c] >= 7:
                ax.text(izq + row[c] / 2, i, fmt(row[c], 0) + " %", ha="center", va="center", fontsize=9,
                        color="white" if col == INDIGO else TINTA)
            izq += row[c]
    ax.set_yticks(range(len(t)), [f"{i}  (n = {nn[i]})" for i in t.index], fontsize=9.5, color=TINTA)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25 %", "50 %", "75 %", "100 %"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Desempeño percibido según la calidad del sueño")
    _sub(ax, "P15 × P24 · % dentro de cada nivel de calidad")
    fig.legend([Patch(color=c) for c in cols], ["Desempeño bajo (1–2)", "Regular (3)", "Alto (4–5)"], ncol=3,
               loc="lower center", bbox_to_anchor=(0.59, 0.0))
    return fig


def forest(m):
    m = m.copy().iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 8.6))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.91, bottom=0.07)
    ylabels, y, pos = [], 0, []
    grupos = list(dict.fromkeys(m["sueno"]))
    for g in grupos:
        sub = m[m["sueno"] == g]
        for _, r in sub.iterrows():
            sig = r["q"] < 0.05
            col = INDIGO if sig else "#A3A9B8"
            ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(r["rho"], y, "o", ms=7, color=DORADO if sig else "white", mec=col, mew=1.8)
            ax.text(0.86, y, fmt(r["rho"], 2) + ("*" if sig else ""), va="center", fontsize=8.5, color=TINTA)
            ylabels.append(f"{r['desempeno']}")
            pos.append(y)
            y += 1
        ax.text(-0.66, y - len(sub) / 2 - 0.5, g, ha="right", va="center", fontsize=9.5, color=TINTA,
                fontweight="bold", transform=ax.transData)
        y += 0.8
    ax.set_yticks(pos, ylabels, fontsize=8.5)
    ax.axvline(0, color=GRIS_T, lw=0.9)
    for xv in (0.1, 0.3, 0.5):
        ax.axvline(xv, color=BORDE, lw=0.8, ls="--", zorder=0)
    ax.set_xlim(-0.4, 0.95)
    ax.set_xlabel("rho de Spearman con IC 95 %  (líneas guía: 0,1 débil · 0,3 moderada · 0,5 fuerte)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Cada aspecto del sueño frente a cada indicador de desempeño")
    _sub(ax, "Variables orientadas: alto = mejor · punto dorado = q < 0,05 (Benjamini-Hochberg, 28 pruebas)")
    return fig


def matriz(m):
    piv = m.pivot(index="sueno", columns="desempeno", values="rho")
    q = m.pivot(index="sueno", columns="desempeno", values="q")
    filas = list(dict.fromkeys(m["sueno"]))
    cols = list(dict.fromkeys(m["desempeno"]))
    piv, q = piv.loc[filas, cols], q.loc[filas, cols]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    fig.subplots_adjust(left=0.33, right=0.97, top=0.8, bottom=0.08)
    ax.imshow(piv.values, cmap=SEQ, vmin=0, vmax=0.75, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            r = piv.values[i, j]
            if np.isnan(r):
                continue
            ax.text(j, i, fmt(r, 2) + ("*" if q.values[i, j] < 0.05 else ""), ha="center", va="center", fontsize=9.5,
                    color="white" if r > 0.42 else TINTA, fontweight="bold" if q.values[i, j] < 0.05 else "normal")
    ax.set_xticks(range(len(cols)), [c.split(" ", 1)[1] for c in cols], rotation=0, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(filas)), filas, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Matriz de Spearman", pad=34)
    return fig


def habitos(t):
    labs = list(dict.fromkeys(t["habito"]))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.subplots_adjust(left=0.3, right=0.97, top=0.84, bottom=0.25)
    for i, h in enumerate(labs):
        for k, (res, col, off) in enumerate([("Calidad del sueño", INDIGO, -0.14), ("Desempeño percibido", DORADO, 0.14)]):
            r = t[(t["habito"] == h) & (t["resultado"] == res)].iloc[0]
            sig = r["q"] < 0.05
            ax.plot([r["lo"], r["hi"]], [i + off] * 2, color=col, lw=2.2, alpha=1 if sig else 0.45)
            ax.plot(r["rho"], i + off, "o", ms=7, color=col if sig else PAPEL, mec=col, mew=1.8)
            ax.text(0.42, i + off, fmt(r["rho"], 2) + ("*" if sig else ""), va="center", fontsize=8.5, color=TINTA)
    ax.set_yticks(range(len(labs)), labs, fontsize=9.5, color=TINTA)
    ax.invert_yaxis()
    ax.axvline(0, color=GRIS_T, lw=0.9)
    ax.set_xlim(-0.7, 0.5)
    ax.set_xlabel("rho de Spearman con IC 95 %  (negativo: más frecuencia del hábito, peor resultado)")
    _xgrid(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Hábitos antes de dormir y su asociación con el sueño y el desempeño")
    _sub(ax, "Relleno = q < 0,05 (Benjamini-Hochberg, 10 pruebas) · hueco = no significativo")
    fig.legend([Patch(color=INDIGO), Patch(color=DORADO)], ["Con calidad del sueño (P15)", "Con desempeño percibido (P24)"],
               ncol=2, loc="lower center", bbox_to_anchor=(0.63, 0.0))
    return fig


def cruce(df, vx, vy):
    t = pd.crosstab(df[vy + "_ETQ"], df[vx + "_ETQ"]).reindex(index=list(ET[vy].values()),
                                                             columns=list(ET[vx].values()), fill_value=0)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.subplots_adjust(left=0.27, right=0.98, top=0.86, bottom=0.26)
    ax.imshow(t.values, cmap=SEQ, origin="lower", aspect="auto")
    mx = max(t.values.max(), 1)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=9, color="white" if v > mx * 0.55 else TINTA)
    ax.set_xticks(range(t.shape[1]), t.columns, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(t.shape[0]), t.index, fontsize=8.5)
    ax.set_xlabel(LAB[vx])
    ax.set_ylabel(LAB[vy])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Tabla cruzada (número de estudiantes)")
    return fig


# ------------------------------------------------------------------ programas
def programas(df_all):
    cnt = df_all["PROGRAMA"].value_counts()
    filas = [(ET["PROGRAMA"][k], N_H[k], CUOTA[k], int(cnt.get(k, 0))) for k in N_H]
    filas.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.84, bottom=0.12)
    for i, (p, N, c, nlog) in enumerate(filas):
        ax.barh(i, N, color=CREMA, height=0.7, edgecolor=BORDE)
        ax.barh(i, nlog, color=INDIGO, height=0.42)
        ax.plot([c, c], [i - 0.36, i + 0.36], color=DORADO, lw=3)
        ax.text(N + 0.6, i, f"{nlog} de {N} · {fmt(nlog / N * 100, 0)} %", va="center", fontsize=8.5, color=TINTA)
    ax.set_yticks(range(len(filas)), [r[0] for r in filas], fontsize=9.5, color=TINTA)
    ax.invert_yaxis()
    ax.set_xlim(0, 52)
    ax.set_xlabel("Estudiantes")
    _xgrid(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Marco, cuota planeada y respuestas por programa")
    _sub(ax, "Barra clara = marco (Nₕ) · barra índigo = respuestas · línea dorada = cuota proporcional (n = 147)")
    return fig


def dist_programa(df, v):
    t = pd.crosstab(df["PROGRAMA_ETQ"], df[v + "_ETQ"], normalize="index") * 100
    t = t[[c for c in ET[v].values() if c in t.columns]]
    t = t.loc[[i for i in t.index if df["PROGRAMA_ETQ"].eq(i).sum() > 0]]
    n = df["PROGRAMA_ETQ"].value_counts()
    k = len(t.columns)
    cols = [SEQ(0.15 + 0.85 * i / max(k - 1, 1)) for i in range(k)]
    fig, ax = plt.subplots(figsize=(10, 0.9 + 0.45 * len(t) + 0.8))
    fig.subplots_adjust(left=0.24, right=0.98, top=0.86, bottom=0.2)
    for i, (idx, row) in enumerate(t.iterrows()):
        izq = 0
        for j, c in enumerate(t.columns):
            ax.barh(i, row[c], left=izq, color=cols[j], height=0.62, edgecolor=PAPEL, lw=1)
            if row[c] >= 9:
                ax.text(izq + row[c] / 2, i, fmt(row[c], 0), ha="center", va="center", fontsize=8,
                        color="white" if j / max(k - 1, 1) > 0.5 else TINTA)
            izq += row[c]
    ax.set_yticks(range(len(t)), [f"{i} (n = {n[i]})" for i in t.index], fontsize=9, color=TINTA)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{LAB[v]} por programa")
    _sub(ax, "% dentro de cada programa · cada respuesta pesa 8 a 10 puntos porcentuales")
    ax.legend([Patch(color=c) for c in cols], list(t.columns), ncol=min(k, 6), loc="upper center",
              bbox_to_anchor=(0.4, -0.12), fontsize=8)
    return fig


# ------------------------------------------------------------------ dimensiones (escalas sumadas)
def normalidad(df, clave, nombre, rango):
    s = df["PUNTAJE_" + clave]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.15, wspace=0.28)
    bins = np.arange(rango[0] - 0.5, rango[1] + 1.5, 1)
    a1.hist(s, bins=bins, color=LAVANDA, edgecolor=PAPEL)
    xs = np.linspace(rango[0], rango[1], 200)
    a1.plot(xs, stats.norm.pdf(xs, s.mean(), s.std(ddof=1)) * len(s), color=INDIGO, lw=2, label="Curva normal teórica")
    a1.axvline(s.mean(), color=DORADO, lw=2, ls="--", label=f"Media = {fmt(s.mean(), 1)}")
    a1.set_xlabel(f"Puntaje ({rango[0]}–{rango[1]})")
    a1.set_ylabel("Estudiantes")
    _ygrid(a1)
    a1.legend(loc="upper left", fontsize=8)
    a1.set_title("Distribución del puntaje")
    (osm, osr), (pend, inter, _) = stats.probplot(s, dist="norm")
    a2.scatter(osm, osr, s=18, color=INDIGO, alpha=0.75)
    a2.plot(osm, pend * np.asarray(osm) + inter, color=DORADO, lw=2)
    a2.set_xlabel("Cuantiles teóricos (normal)")
    a2.set_ylabel("Puntaje observado")
    _ygrid(a2)
    sw = stats.shapiro(s)
    a2.set_title("Gráfico Q-Q")
    _sub(a2, f"Shapiro-Wilk: W = {fmt(sw.statistic, 3)}, p = {fmt(sw.pvalue, 3)}")
    fig.suptitle(nombre, x=0.07, ha="left", fontsize=13, fontweight="bold", color=TINTA, y=0.98)
    return fig


def boxplot(df, grupo_col, clave, nombre, rango, titulo=None, ax=None, horizontal=False):
    cats = [c for c in df[grupo_col].cat.categories if (df[grupo_col] == c).sum() > 0]
    datos = [df.loc[df[grupo_col] == c, "PUNTAJE_" + clave].values for c in cats]
    propio = ax is None
    if propio:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.2)
    rng = np.random.default_rng(7)
    _v = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
    orient = ({"orientation": "horizontal"} if _v >= (3, 10) else {"vert": False}) if horizontal else {}
    bp = ax.boxplot(datos, **orient, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=TINTA, lw=2), whiskerprops=dict(color=GRIS_T),
                    capprops=dict(color=GRIS_T), boxprops=dict(edgecolor=INDIGO, lw=1.2))
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(SEQ(0.18 + 0.27 * i / max(len(cats) - 1, 1)))
        b.set_alpha(0.9)
    for i, d in enumerate(datos):
        j = rng.uniform(-0.13, 0.13, len(d))
        if horizontal:
            ax.scatter(d, np.full(len(d), i + 1) + j, s=12, color=NOCHE, alpha=0.45, zorder=3)
            ax.scatter(d.mean(), i + 1, marker="D", s=48, color=DORADO, edgecolor=NOCHE, zorder=4)
        else:
            ax.scatter(np.full(len(d), i + 1) + j, d, s=12, color=NOCHE, alpha=0.45, zorder=3)
            ax.scatter(i + 1, d.mean(), marker="D", s=48, color=DORADO, edgecolor=NOCHE, zorder=4)
    etiquetas = [f"{c}\n(n = {len(d)})" for c, d in zip(cats, datos)]
    if horizontal:
        ax.set_yticks(range(1, len(cats) + 1), [f"{c} (n = {len(d)})" for c, d in zip(cats, datos)], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(rango[0] - 0.5, rango[1] + 0.5)
        ax.set_xlabel(f"Puntaje de {nombre.lower()} ({rango[0]}–{rango[1]})")
        _xgrid(ax)
    else:
        ax.set_xticks(range(1, len(cats) + 1), etiquetas, fontsize=8.5)
        ax.set_ylim(rango[0] - 0.5, rango[1] + 0.5)
        ax.set_ylabel(f"Puntaje ({rango[0]}–{rango[1]})")
        _ygrid(ax)
    if titulo:
        ax.set_title(titulo)
    return ax.figure if propio else None


def boxplots_dimensiones(df, grupo, titulo_grupo, dims, subtitulos):
    """Dos box plots lado a lado (calidad y funcionamiento) para una variable de agrupación."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.7, bottom=0.18, wspace=0.22)
    for ax, (clave, d), sub in zip(axes, dims.items(), subtitulos):
        boxplot(df, "G_" + grupo, clave, d["nombre"], d["rango"], titulo=d["corto"], ax=ax)
        _sub(ax, sub)
    fig.suptitle(titulo_grupo, x=0.07, ha="left", fontsize=13, fontweight="bold", color=TINTA, y=0.98)
    fig.text(0.07, 0.9, "Caja = cuartiles · línea = mediana · rombo dorado = media · puntos = estudiantes",
             fontsize=9, color=GRIS_T)
    return fig


def boxplot_programas(df, clave, d, sub):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.22, right=0.98, top=0.84, bottom=0.12)
    boxplot(df, "G_PROGRAMA", clave, d["nombre"], d["rango"], titulo=f"{d['nombre']} por programa", ax=ax,
            horizontal=True)
    _sub(ax, sub)
    return fig


def dispersion(df, r):
    rng = np.random.default_rng(3)
    x = df["PUNTAJE_CALIDAD"] + rng.uniform(-0.22, 0.22, len(df))
    y = df["PUNTAJE_FUNCIONAMIENTO"] + rng.uniform(-0.22, 0.22, len(df))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.13)
    ax.scatter(x, y, s=28, color=INDIGO, alpha=0.6, edgecolor=PAPEL, lw=0.6)
    b, a = np.polyfit(df["PUNTAJE_CALIDAD"], df["PUNTAJE_FUNCIONAMIENTO"], 1)
    xs = np.linspace(4, 20, 50)
    ax.plot(xs, a + b * xs, color=DORADO, lw=2.4)
    ax.set_xlim(3.5, 20.5)
    ax.set_ylim(4.5, 25.5)
    ax.set_xlabel("Calidad del sueño (puntaje 4–20)")
    ax.set_ylabel("Funcionamiento académico (5–25)")
    _ygrid(ax)
    _xgrid(ax)
    ax.set_title("Calidad del sueño y funcionamiento académico")
    _sub(ax, f"Spearman ρ = {fmt(r['spearman'], 2)} · Kendall τ-b = {fmt(r['kendall'], 2)} · n = {r['n']}")
    return fig

# ##################################################################################################
# ESTILOS Y COMPONENTES VISUALES
# ##################################################################################################
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500..700&family=Nunito+Sans:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, .stSelectbox, .stMultiSelect { font-family: 'Nunito Sans', 'Segoe UI', Arial, sans-serif; }
.stApp { background: #F8F3E8; }
section[data-testid="stSidebar"] { background: #F1E9D8; border-right: 1px solid #E6DECD; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }
h1, h2, h3, .serif { font-family: 'Fraunces', Georgia, serif !important; color: #18233F; letter-spacing: -0.01em; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #E6DECD; }
.stTabs [data-baseweb="tab"] { font-weight: 700; font-size: 1.02rem; color: #5B6478; padding: 10px 18px; border-radius: 10px 10px 0 0; }
.stTabs [aria-selected="true"] { color: #18233F; background: #FFFDF8; border: 1px solid #E6DECD; border-bottom-color: #FFFDF8; }
div[data-testid="stImage"] img, .stPlot img { border-radius: 14px; border: 1px solid #E6DECD; }
div[data-testid="stDataFrame"] { border: 1px solid #E6DECD; border-radius: 12px; }

.hero { position: relative; overflow: hidden; border-radius: 22px; padding: 34px 40px 30px; margin: 0 0 22px;
  background: linear-gradient(135deg, #0D1530 0%, #1C2752 62%, #3A2F5B 100%); color: #F4EFE3; }
.hero .eyebrow { font-size: .78rem; font-weight: 800; letter-spacing: .18em; text-transform: uppercase; color: #F2C46D; margin: 0 0 10px; }
.hero h1 { font-family: 'Fraunces', Georgia, serif !important; color: #F4EFE3 !important; font-size: 2.6rem; line-height: 1.05; margin: 0 0 10px; padding: 0; }
.hero .sub { font-size: 1.08rem; color: #C3CAE8; max-width: 760px; margin: 0 0 18px; line-height: 1.45; }
.hero .chips { display: flex; flex-wrap: wrap; gap: 8px; }
.hero .chip { background: rgba(244,239,227,.08); border: 1px solid rgba(244,239,227,.18); color: #E7E9F5; padding: 5px 12px; border-radius: 999px; font-size: .84rem; }
.hero svg.moon { position: absolute; right: 46px; top: 22px; opacity: .95; }
.hero svg.stars { position: absolute; right: 0; top: 0; }

.sec { margin: 6px 0 14px; }
.sec .kicker { font-size: .76rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; color: #9A5A0B; margin: 0 0 4px; }
.sec .kicker.night { color: #2F3F86; }
.sec h2 { font-size: 1.75rem; margin: 0 0 6px; padding: 0; }
.sec p { color: #5B6478; margin: 0; font-size: 1rem; max-width: 900px; line-height: 1.5; }

.kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin: 4px 0 22px; }
.kpi { background: #FFFDF8; border: 1px solid #E6DECD; border-radius: 16px; padding: 16px 18px 15px; }
.kpi .lbl { font-size: .8rem; font-weight: 700; color: #5B6478; text-transform: uppercase; letter-spacing: .06em; margin: 0 0 6px; }
.kpi .val { font-family: 'Fraunces', Georgia, serif; font-size: 2.15rem; font-weight: 600; color: #18233F; line-height: 1.05; margin: 0 0 4px; }
.kpi .ctx { font-size: .86rem; color: #5B6478; margin: 0; line-height: 1.35; }
.kpi.accent { background: linear-gradient(160deg, #141D44, #2F3F86); border-color: #141D44; }
.kpi.accent .lbl { color: #C3CAE8; } .kpi.accent .val { color: #F2C46D; } .kpi.accent .ctx { color: #DDE2F4; }

.findings { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 6px 0 22px; }
.find { background: #FFFDF8; border: 1px solid #E6DECD; border-radius: 16px; padding: 18px 20px 16px; display: flex; flex-direction: column; }
.find .top { display: flex; align-items: center; justify-content: space-between; margin: 0 0 10px; }
.find .tag { font-size: .72rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; padding: 3px 9px; border-radius: 999px; }
.find .tag.n { background: #1C2752; color: #F2C46D; } .find .tag.d { background: #F6DDB0; color: #6B3F06; }
.find .num { font-family: 'Fraunces', Georgia, serif; font-size: 1.9rem; font-weight: 600; color: #2F3F86; line-height: 1; }
.find h4 { font-family: 'Fraunces', Georgia, serif; font-size: 1.12rem; color: #18233F; margin: 0 0 8px; line-height: 1.3; }
.find .ev { font-size: .9rem; color: #3C4763; margin: 0 0 10px; line-height: 1.45; flex: 1; }
.find .cv { font-size: .8rem; color: #6E7486; margin: 0; padding-top: 9px; border-top: 1px dashed #E6DECD; line-height: 1.4; }

.callout { border-radius: 14px; padding: 14px 18px; margin: 8px 0 18px; font-size: .95rem; line-height: 1.5; }
.callout.info { background: #EEF0FA; border-left: 5px solid #2F3F86; color: #18233F; }
.callout.warn { background: #FBF0DA; border-left: 5px solid #E9A93B; color: #3D2A06; }
.callout b { color: inherit; }

.stat { background: linear-gradient(160deg, #0D1530, #1C2752); color: #F4EFE3; border-radius: 18px; padding: 22px 24px; }
.stat .lbl { color: #C3CAE8; font-size: .85rem; margin: 0 0 4px; }
.stat .big { font-family: 'Fraunces', Georgia, serif; color: #F2C46D; font-size: 3.6rem; font-weight: 600; line-height: 1; margin: 4px 0 10px; }
.stat .det { color: #DDE2F4; font-size: .9rem; margin: 0 0 4px; line-height: 1.45; }

.mini { background: #FFFDF8; border: 1px solid #E6DECD; border-radius: 14px; padding: 14px 16px; margin: 0 0 12px; }
.mini .v { font-family: 'Fraunces', Georgia, serif; font-size: 1.7rem; color: #18233F; font-weight: 600; line-height: 1.1; }
.mini .t { font-size: .88rem; color: #5B6478; margin-top: 2px; line-height: 1.4; }

.foot { margin-top: 34px; padding-top: 14px; border-top: 1px solid #E6DECD; color: #6E7486; font-size: .82rem; line-height: 1.5; }
@media (max-width: 1100px) { .kpis { grid-template-columns: repeat(2, 1fr); } .findings { grid-template-columns: 1fr; } }
</style>
"""

_STARS = [(30, 30, 1.6), (90, 70, 1.1), (150, 20, 1.3), (210, 95, 1.0), (270, 40, 1.5), (330, 110, 1.1), (380, 25, 1.2),
          (60, 130, 1.0), (240, 140, 1.2), (360, 150, 1.4)]


def _e(t):
    return _html.escape(str(t), quote=False)


def hero(eyebrow, titulo, sub, chips):
    stars = "".join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#F4EFE3" opacity="{0.35 + (i % 3) * 0.2:.2f}"/>'
                    for i, (x, y, r) in enumerate(_STARS))
    moon = ('<svg class="moon" width="120" height="120" viewBox="0 0 120 120" aria-hidden="true">'
            '<circle cx="60" cy="60" r="58" fill="#F2C46D" opacity="0.08"/>'
            '<path d="M66 18 A43 43 0 1 0 101 79 A35 35 0 1 1 66 18 Z" fill="#F2C46D"/></svg>')
    ch = "".join(f'<span class="chip">{_e(c)}</span>' for c in chips)
    return (f'<div class="hero"><svg class="stars" width="420" height="170" viewBox="0 0 420 170" aria-hidden="true">{stars}</svg>'
            f'{moon}<p class="eyebrow">{_e(eyebrow)}</p><h1>{_e(titulo)}</h1><p class="sub">{_e(sub)}</p>'
            f'<div class="chips">{ch}</div></div>')


def seccion(kicker, titulo, texto="", noche=False):
    return (f'<div class="sec"><p class="kicker{" night" if noche else ""}">{_e(kicker)}</p><h2>{_e(titulo)}</h2>'
            + (f'<p>{texto}</p>' if texto else "") + '</div>')


def kpis(items):
    """items: lista de (etiqueta, valor, contexto, acento)"""
    cards = "".join(f'<div class="kpi{" accent" if a else ""}"><p class="lbl">{_e(l)}</p><p class="val">{_e(v)}</p>'
                    f'<p class="ctx">{c}</p></div>' for l, v, c, a in items)
    return f'<div class="kpis">{cards}</div>'


def hallazgos(items):
    """items: lista de dict(tag, tipo 'n'|'d', num, titulo, evidencia, cautela)"""
    cards = "".join(f'<div class="find"><div class="top"><span class="tag {i["tipo"]}">{_e(i["tag"])}</span>'
                    f'<span class="num">{_e(i["num"])}</span></div><h4>{_e(i["titulo"])}</h4>'
                    f'<p class="ev">{i["evidencia"]}</p><p class="cv">{i["cautela"]}</p></div>' for i in items)
    return f'<div class="findings">{cards}</div>'


def callout(texto, tipo="info"):
    return f'<div class="callout {tipo}">{texto}</div>'


def stat(lbl, big, detalles):
    det = "".join(f'<p class="det">{d}</p>' for d in detalles)
    return f'<div class="stat"><p class="lbl">{_e(lbl)}</p><p class="big">{_e(big)}</p>{det}</div>'


def mini(v, t):
    return f'<div class="mini"><div class="v">{_e(v)}</div><div class="t">{t}</div></div>'


def pie(texto):
    return f'<div class="foot">{texto}</div>'

# ##################################################################################################
# INTERFAZ
# ##################################################################################################

# ================================================================== compatibilidad entre versiones de Streamlit
_ST_VER = tuple(int(x) for x in getattr(st, "__version__", "0.0").split(".")[:2] if x.isdigit())
# Desde Streamlit 1.50 «use_container_width» está en desuso y se reemplaza por width="stretch".
ANCHO_KW = {"width": "stretch"} if _ST_VER >= (1, 50) else {"use_container_width": True}


def ancho(fn, *args, **kwargs):
    """Llama a un elemento de Streamlit ocupando todo el ancho, en versiones nuevas y antiguas."""
    try:
        return fn(*args, **ANCHO_KW, **kwargs)
    except TypeError:
        alt = {"use_container_width": True} if "width" in ANCHO_KW else {"width": "stretch"}
        return fn(*args, **alt, **kwargs)

st.set_page_config(page_title="De la noche al día · Sueño y desempeño", page_icon="🌙", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
AQUI = Path(__file__).parent
MIN_N = 5


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def mostrar(fig):
    ancho(st.pyplot, fig)
    plt.close(fig)


@st.cache_data(show_spinner=False)
def cargar_bytes(contenido: bytes):
    return leer_excel(contenido)


@st.cache_data(show_spinner=False)
def cargar_ruta(ruta: str):
    return leer_excel(ruta)


@st.cache_data(show_spinner="Calculando intervalos bootstrap…")
def boot(x: tuple, y: tuple):
    return spearman_boot(np.array(x), np.array(y))


# ================================================================== datos y filtros
with st.sidebar:
    st.markdown("### 🌙 De la noche al día")
    st.caption("Dashboard de hallazgos del proyecto de Recolección de Datos · 2026-2")
    st.divider()
    st.markdown("**Filtros de exploración**")
    st.caption("Afectan las pestañas La noche, El día, Asociaciones y Programas. El Resumen siempre muestra la "
               "muestra completa, igual que el informe.")
    sub = st.file_uploader("Cambiar base de datos (.xlsx)", type=["xlsx"],
                           help="Opcional. Debe tener las 26 columnas del diccionario (hoja Base_datos).")

try:
    if sub is not None:
        df_all = cargar_bytes(sub.getvalue())
        origen = sub.name
    else:
        ruta = buscar_base(AQUI)
        if ruta is None:
            st.error("No se encontró la base de datos. Coloca **base_procesada.xlsx** junto a app.py o súbela desde "
                     "la barra lateral.")
            st.stop()
        df_all = cargar_ruta(str(ruta))
        origen = ruta.name
except ValueError as e:
    st.error(f"La base no tiene el formato esperado: {e}")
    st.stop()

df_all = agregar_puntajes(df_all)

with st.sidebar:
    progs = st.multiselect("Programa", list(ET["PROGRAMA"].values()), default=list(ET["PROGRAMA"].values()))
    sexos = st.multiselect("Sexo", list(ET["SEXO"].values()), default=list(ET["SEXO"].values()))
    edades = st.multiselect("Edad", list(ET["EDAD"].values()), default=list(ET["EDAD"].values()))
    st.divider()
    st.caption(f"Fuente: {origen} · {len(df_all)} respuestas válidas")
    st.caption("Muestra voluntaria y autoseleccionada (no probabilística). Los resultados describen asociaciones, "
               f"no causas. Por privacidad no se muestran grupos con menos de {MIN_N} respuestas.")

df = df_all[df_all["PROGRAMA_ETQ"].isin(progs) & df_all["SEXO_ETQ"].isin(sexos) & df_all["EDAD_ETQ"].isin(edades)]
n, N = len(df), len(df_all)
filtrado = n < N

# ================================================================== encabezado
html(hero("Recolección de Datos · Facultad de Ingeniería · Universidad de La Sabana",
             "De la noche al día",
             "Hábitos de sueño y desempeño académico percibido en estudiantes de tercer semestre de Ingeniería, 2026-2. "
             "Estudio cuantitativo, transversal y descriptivo-correlacional.",
             [f"Marco: {N_POB} estudiantes", f"{N} respuestas válidas", f"Tasa de respuesta {fmt(N / N_POB * 100, 0)} %",
              "8 programas", "Formulario: 22 sep – 15 nov 2026", "Escalas sumadas · ANOVA · correlación no paramétrica"]))

tabs = st.tabs(["Resumen", "🌙 La noche", "☀️ El día", "Dimensiones", "Bivariado y ANOVA", "Programas", "Metodología"])


def aviso_filtro():
    if filtrado:
        html(callout(f"<b>Vista filtrada:</b> {n} de {N} respuestas. Con grupos pequeños los porcentajes y "
                        "correlaciones son inestables.", "warn"))


def suficiente():
    if n < MIN_N:
        html(callout(f"La selección tiene {n} respuestas (menos de {MIN_N}). Amplía los filtros para ver resultados.",
                        "warn"))
        return False
    return True


# ================================================================== 1 · RESUMEN
with tabs[0]:
    A = df_all
    DIMS = DIMENSIONES
    ALF, _ = consistencia(A)
    alfa = ALF.set_index("clave")["alfa"]
    COR = correlaciones(A)
    an_pan_c, an_pan_f = anova(A, "CALIDAD", "PANTALLAS"), anova(A, "FUNCIONAMIENTO", "PANTALLAS")
    an_car_c, an_car_f = anova(A, "CALIDAD", "CARGA"), anova(A, "FUNCIONAMIENTO", "CARGA")
    an_hor_f = anova(A, "FUNCIONAMIENTO", "HORAS")
    an_prg_f = anova(A, "FUNCIONAMIENTO", "PROGRAMA")
    an_prg_c = anova(A, "CALIDAD", "PROGRAMA")
    q_prg = bh([an_prg_c["p_principal"], an_prg_f["p_principal"]])

    html(seccion("Resumen ejecutivo", "Lo que encontramos",
                    "Indicadores y hallazgos calculados sobre la muestra completa, en coherencia con el informe final. "
                    "El análisis principal suma las puntuaciones de cada estudiante por dimensión y compara grupos con ANOVA."))
    html(kpis([
        ("Respuestas válidas", f"{N}", f"de {N_POB} en el marco · {fmt(N / N_POB * 100, 1)} % · meta {N_PLAN}", False),
        ("Duermen < 7 h", f"{fmt(pct(A, 'HORAS_SUENO_SEM', [1, 2, 3]))} %", "entre semana (P8)", False),
        ("Calidad del sueño", f"{fmt(A['PUNTAJE_CALIDAD'].mean(), 1)} / 20", f"puntaje medio · α = {fmt(alfa['CALIDAD'], 2)}", False),
        ("Funcionamiento académico", f"{fmt(A['PUNTAJE_FUNCIONAMIENTO'].mean(), 1)} / 25", f"puntaje medio · α = {fmt(alfa['FUNCIONAMIENTO'], 2)}", False),
        ("Calidad × funcionamiento", f"ρ = {fmt(COR['spearman'], 2)}",
         f"Spearman · IC 95 % [{fmt(COR['sp_lo'], 2)}; {fmt(COR['sp_hi'], 2)}] · τ-b = {fmt(COR['kendall'], 2)}", True),
    ]))
    mp = an_pan_f["medias"]
    html(seccion("Hallazgos principales", "Seis resultados para recordar"))
    html(hallazgos([
        dict(tag="Obj. 1 · Noche", tipo="n", num=f"{fmt(pct(A, 'HORAS_SUENO_SEM', [1, 2, 3]), 0)} %",
             titulo="Dormir poco es la norma entre semana",
             evidencia=f"La mediana entre semana es la franja de 6 a menos de 7 h. El fin de semana el "
                       f"{fmt(pct(A, 'HR_DORMIR_FIN', [5]), 0)} % se acuesta después de la medianoche.",
             cautela="Las horas se preguntaron en franjas: no son horas exactas."),
        dict(tag="Obj. 3–4 · Dimensiones", tipo="n", num=f"α {fmt(alfa['CALIDAD'], 2)}",
             titulo="Dos dimensiones que se pueden sumar",
             evidencia=f"Calidad del sueño (P15–P18, α = {fmt(alfa['CALIDAD'], 2)}) y funcionamiento académico "
                       f"(P19–P22 y P24, α = {fmt(alfa['FUNCIONAMIENTO'], 2)}). Los hábitos (α = {fmt(alfa['HABITOS'], 2)}) "
                       f"no forman una escala y se analizan por separado.",
             cautela="El puntaje de calidad no es normal en la muestra completa (Shapiro-Wilk p = 0,004); se contrasta con Kruskal-Wallis."),
        dict(tag="Obj. 5 · Correlación", tipo="d", num=f"ρ {fmt(COR['spearman'], 2)}",
             titulo="Quien duerme mejor, funciona mejor",
             evidencia=f"Correlación no paramétrica fuerte entre ambos puntajes (Spearman {fmt(COR['spearman'], 2)}; "
                       f"Kendall τ-b {fmt(COR['kendall'], 2)}; p < 0,001).",
             cautela="Asociación, no causa. P19, P20 y P22 mencionan el cansancio y comparten contenido con el sueño."),
        dict(tag="Obj. 2 · ANOVA", tipo="n", num=f"η² {fmt(an_pan_f['eta2'], 2)}",
             titulo="Las pantallas marcan la mayor diferencia",
             evidencia=f"Funcionamiento académico medio de {fmt(mp['Nunca o rara vez'][0], 1)} sin pantallas frente a "
                       f"{fmt(mp['Frecuente o siempre'][0], 1)} con uso frecuente (F(2, 84) = {fmt(an_pan_f['F'], 2)}; "
                       f"p {p_txt(an_pan_f['p'])}). En calidad del sueño: η² = {fmt(an_pan_c['eta2'], 2)}.",
             cautela="Diseño transversal: no indica dirección; quien duerme mal también puede usar más el celular."),
        dict(tag="Obj. 2 · ANOVA", tipo="n", num=f"η² {fmt(an_car_c['eta2'], 2)}",
             titulo="La carga académica pesa; las horas, no tanto",
             evidencia=f"Quien cambia su horario por la carga tiene menor calidad del sueño (F = {fmt(an_car_c['F'], 2)}; "
                       f"p {p_txt(an_car_c['p'])}) y funcionamiento. Las horas de sueño no marcan diferencias "
                       f"significativas (p {p_txt(an_hor_f['p'])}).",
             cautela="Los hábitos se agruparon en tres niveles; se ajustó por comparaciones múltiples (Benjamini-Hochberg)."),
        dict(tag="Obj. 6 · Programas", tipo="d", num=f"p {fmt(an_prg_f['p'], 3)}",
             titulo="Diferencia exploratoria entre programas",
             evidencia=f"El funcionamiento académico difiere entre programas (F(7, 79) = {fmt(an_prg_f['F'], 2)}; "
                       f"η² = {fmt(an_prg_f['eta2'], 2)}); Tukey: Ciencia de Datos por debajo de Industrial. La calidad del "
                       f"sueño no difiere (p = {fmt(an_prg_c['p'], 3)}).",
             cautela=f"No se mantiene al ajustar por comparaciones múltiples (q = {fmt(q_prg[1], 3)}); grupos de 10 a 12 estudiantes."),
    ]))
    html(seccion("Hallazgo central", "Puntajes según el uso de pantallas antes de dormir",
                    "Cada caja resume los puntajes de un grupo; el rombo dorado es la media y cada punto, un estudiante."))
    mostrar(boxplots_dimensiones(A, "PANTALLAS", "Puntajes según uso de pantallas antes de dormir (P11)", DIMS,
                                        [f"ANOVA F(2, 84) = {fmt(an_pan_c['F'], 2)}; p {p_txt(an_pan_c['p'])}; η² = {fmt(an_pan_c['eta2'], 2)}",
                                         f"ANOVA F(2, 84) = {fmt(an_pan_f['F'], 2)}; p {p_txt(an_pan_f['p'])}; η² = {fmt(an_pan_f['eta2'], 2)}"]))
    html(callout("<b>Cómo leer este dashboard.</b> Cada pregunta es una categoría ordenada. Las preguntas que miden lo "
                    "mismo se suman en un puntaje por estudiante (escala tipo Likert) cuando su alfa de Cronbach es ≥ 0,70; "
                    "esos puntajes se comparan con ANOVA (revisando normalidad y varianzas, con Kruskal-Wallis como contraste) "
                    "y se relacionan con coeficientes no paramétricos. La participación fue voluntaria (87 de 235) y una "
                    "asociación no implica causalidad.", "info"))

# ================================================================== 2 · LA NOCHE
with tabs[1]:
    html(seccion("Parte 1 · Objetivos 1 a 3", "La noche: cómo duermen",
                    "Horarios, duración, hábitos antes de dormir y factores que los estudiantes asocian con su sueño.",
                    noche=True))
    aviso_filtro()
    if suficiente():
        c = st.columns(4)
        for col, (v, t) in zip(c, [(f"{fmt(pct(df, 'HORAS_SUENO_SEM', [1, 2, 3]))} %", "duerme menos de 7 h entre semana"),
                                   (f"{fmt(pct(df, 'HR_DORMIR_SEM', [4, 5]))} %", "se acuesta después de las 11 p. m. entre semana"),
                                   (f"{fmt(pct(df, 'FREC_CARGA_ACAD', [4, 5]))} %", "cambia su horario por la carga académica"),
                                   (f"{fmt(pct(df, 'REGULARIDAD_HORARIO', [4, 5]))} %", "mantiene horarios regulares con frecuencia")]):
            with col:
                html(mini(v, t))
        a, b = st.columns(2)
        with a:
            mostrar(horas(df))
        with b:
            mostrar(acostarse(df))
        a, b = st.columns(2)
        with a:
            mostrar(transicion(df))
        with b:
            mostrar(factor_principal(df))
        mostrar(likert(df, ["FREC_DISPOSITIVOS", "FREC_CARGA_ACAD", "FREC_CAFEINA", "REGULARIDAD_HORARIO"],
                              "Hábitos antes de dormir (P11–P14)",
                              "En P11–P13 «Siempre» es desfavorable; en P14 (horarios regulares) «Siempre» es favorable"))
        sies = pct(df, "FREC_SIESTA", [4, 5])
        html(callout(f"<b>Siestas (P10):</b> el {fmt(pct(df, 'FREC_SIESTA', [1, 2]))} % nunca o rara vez las toma y el "
                        f"{fmt(sies)} % las toma frecuentemente o siempre; no se asociaron con la calidad del sueño "
                        "ni con el desempeño.", "info"))

# ================================================================== 3 · EL DÍA
with tabs[2]:
    html(seccion("Parte 2 · Objetivos 3 y 4", "El día: cómo se sienten y cómo rinden",
                    "Calidad del sueño, somnolencia y las cinco preguntas de desempeño académico percibido."))
    aviso_filtro()
    if suficiente():
        c = st.columns(4)
        for col, (v, t) in zip(c, [(f"{fmt(pct(df, 'FREC_SOMNOLENCIA_DIA', [4, 5]))} %", "somnolencia diurna frecuente"),
                                   (f"{fmt(pct(df, 'FREC_DIFICULTAD_CONCENTRACION', [4, 5]))} %", "dificultad frecuente para concentrarse"),
                                   (f"{fmt(pct(df, 'PRODUCTIVIDAD_ACAD', [1, 2]))} %", "productividad baja o muy baja"),
                                   (f"{fmt(pct(df, 'DESEMPENO_ACAD', [4, 5]))} %", "desempeño percibido alto o muy alto")]):
            with col:
                html(mini(v, t))
        mostrar(likert(df, ["CALIDAD_SUENO", "FREC_CONCILIAR", "FREC_DESPERTAR_NOCHE", "FREC_SUENO_NO_REPARADOR",
                                   "FREC_SOMNOLENCIA_DIA"], "Calidad del sueño y somnolencia (P15–P19)"))
        mostrar(likert(df, ["FREC_DIFICULTAD_CONCENTRACION", "PRODUCTIVIDAD_ACAD", "FREC_ERRORES_CANSANCIO",
                                   "DESEMPENO_ACAD"], "Desempeño académico percibido (P20–P24)"))
        a, b = st.columns([1, 1])
        with a:
            mostrar(p23(df))
        with b:
            html(seccion("Percepción de la relación", "¿Creen que el sueño influye?"))
            html(mini(f"{fmt(pct(df, 'INFLUENCIA_SUENO_DESEMPENO', [4, 5]))} %",
                         "cree que la calidad de su sueño influye mucho o demasiado en su desempeño (P25)"))
            html(mini(f"{fmt(pct(df, 'INFLUENCIA_SUENO_DESEMPENO', [1, 2]))} %", "cree que influye poco o nada"))
            html(callout("<b>Sobre P23.</b> La pregunta da por hecho que hubo falta de sueño, lo que puede inflar el "
                            "«Sí». En la muestra completa, la calidad del sueño no difiere entre quienes responden Sí, No "
                            "o Tal vez, por lo que P23 no se usa como evidencia principal.", "warn"))

# ================================================================== 4 · DIMENSIONES
with tabs[3]:
    html(seccion("Objetivos 3 y 4", "Dimensiones: de preguntas a puntajes",
                    "Las preguntas negativas se invierten (6 − x) para que alto signifique «mejor». Luego se suman las "
                    "puntuaciones de cada estudiante en las preguntas de una misma dimensión, solo si la dimensión es "
                    "internamente consistente (α de Cronbach ≥ 0,70)."))
    aviso_filtro()
    if suficiente():
        ALFd, ITMd = consistencia(df)
        v = ALFd.copy()
        v["α de Cronbach"] = v["alfa"].map(lambda x: fmt(x, 3))
        st.dataframe(v[["dimension", "k", "α de Cronbach", "decision"]].rename(columns={"dimension": "Dimensión", "k": "Preguntas", "decision": "Decisión"}),
                     hide_index=True, **ANCHO_KW)
        with st.expander("Correlación ítem-total y alfa si se elimina cada pregunta"):
            w = ITMd.copy()
            w["r ítem-total"] = w["r_item_total"].map(lambda x: fmt(x, 3))
            w["α si se elimina"] = w["alfa_sin_item"].map(lambda x: fmt(x, 3))
            w["Invertida"] = w["invertido"].map({True: "Sí", False: "No"})
            st.dataframe(w[["dimension", "item", "Invertida", "r ítem-total", "α si se elimina"]].rename(
                columns={"dimension": "Dimensión", "item": "Pregunta"}), hide_index=True, **ANCHO_KW)
        html(seccion("Paso 2", "Puntajes y prueba de normalidad"))
        des = descriptivos(df)
        c = st.columns(2)
        for col, (_, r) in zip(c, des.iterrows()):
            with col:
                html(mini(f"{fmt(r['media'], 2)} (DE {fmt(r['de'], 2)})",
                             f"<b>{r['dimension']}</b> · rango {r['rango_posible']} · mediana {fmt(r['mediana'], 1)} · "
                             f"asimetría {fmt(r['asimetria'], 2)} · curtosis {fmt(r['curtosis'], 2)} · "
                             f"Shapiro-Wilk W = {fmt(r['shapiro_W'], 3)}, p {p_txt(r['shapiro_p'])} → {r['normal']}"))
        for k, d in DIMENSIONES.items():
            mostrar(normalidad(df, k, d["nombre"], d["rango"]))
        with st.expander("Normalidad dentro de cada programa (supuesto del ANOVA por grupos)"):
            npg = normalidad_por_grupo(df)
            if len(npg):
                npg["W"] = npg["W"].map(lambda x: fmt(x, 3))
                npg["p"] = npg["p"].map(fmt_p)
                ancho(st.dataframe, npg.rename(columns={"dimension": "Dimensión", "grupo": "Programa"}), hide_index=True)
        html(callout("<b>Lectura.</b> Con varias preguntas sumadas el puntaje tiene más niveles y se comporta de forma "
                        "aproximadamente continua. El de calidad del sueño es más plano que una normal (curtosis negativa), "
                        "pero dentro de los programas y en los residuos del ANOVA la normalidad es razonable; por eso el ANOVA "
                        "se acompaña siempre de Kruskal-Wallis.", "info"))

# ================================================================== 5 · BIVARIADO Y ANOVA
with tabs[4]:
    html(seccion("Objetivos 2 y 5", "Análisis bivariado: puntajes según los hábitos de sueño",
                    "Cada hábito se agrupa en tres niveles y se compara cada puntaje entre niveles con ANOVA de un factor. "
                    "Se verifica homogeneidad (Levene), se informa Welch y Kruskal-Wallis, y Tukey cuando el ANOVA es significativo."))
    aviso_filtro()
    if suficiente():
        opciones = {"PANTALLAS": "Pantallas antes de dormir (P11)", "CARGA": "Carga académica altera el horario (P13)",
                    "CAFEINA": "Cafeína después de 6 p. m. (P12)", "HORAS": "Horas de sueño entre semana (P8)",
                    "ACOSTARSE": "Hora de acostarse entre semana (P4)", "REGULARIDAD": "Horarios regulares (P14)"}
        g = st.selectbox("Hábito para comparar", list(opciones), format_func=lambda x: opciones[x])
        res = {k: anova(df, k, g) for k in DIMENSIONES}
        if any(r is None for r in res.values()):
            html(callout("No hay al menos dos grupos con 3 o más respuestas en la selección.", "warn"))
        else:
            subs = [f"F({r['gl1']}, {r['gl2']}) = {fmt(r['F'], 2)}; p {p_txt(r['p'])}; η² = {fmt(r['eta2'], 2)}" for r in res.values()]
            mostrar(boxplots_dimensiones(df, g, f"Puntajes según {opciones[g][0].lower() + opciones[g][1:]}", DIMENSIONES, subs))
            c = st.columns(2)
            for col, (k, r) in zip(c, res.items()):
                with col:
                    det = [f"Prueba principal: {r['prueba_principal']} · p {p_txt(r['p_principal'])}",
                           f"η² = {fmt(r['eta2'], 3)} · ω² = {fmt(r['omega2'], 3)}",
                           f"Levene p = {fmt(r['levene_p'], 3)} · Shapiro residuos p = {fmt(r['shapiro_resid_p'], 3)}",
                           f"Welch p {p_txt(r['welch_p'])} · Kruskal-Wallis p {p_txt(r['kruskal_p'])}"]
                    html(stat(DIMENSIONES[k]["nombre"], f"F = {fmt(r['F'], 2)}", det))
                    medias = pd.DataFrame([(gr, n_, fmt(m, 2), fmt(s, 2)) for gr, (m, s, n_) in r["medias"].items()],
                                          columns=["Grupo", "n", "Media", "DE"])
                    ancho(st.dataframe, medias, hide_index=True)
                    if r["tukey"] is not None:
                        tk = r["tukey"].copy()
                        tk["diferencia"] = tk["diferencia"].map(lambda x: fmt(x, 2))
                        tk["p"] = tk["p"].map(fmt_p)
                        st.caption("Comparaciones de Tukey HSD")
                        st.dataframe(tk.rename(columns={"grupo_a": "Grupo A", "grupo_b": "Grupo B", "diferencia": "Diferencia de medias", "p": "p (Tukey)"}),
                                     hide_index=True, **ANCHO_KW)
        html(seccion("Resumen", "Las 12 pruebas con ajuste de Benjamini-Hochberg"))
        tb = tabla_anovas(df, grupos=("HORAS", "ACOSTARSE", "PANTALLAS", "CARGA", "CAFEINA", "REGULARIDAD"))
        if len(tb):
            tb["q"] = bh(tb["p_principal"])
            vista = pd.DataFrame({"Hábito": tb["grupo"], "Dimensión": tb["dimension"],
                                  "F (gl)": [f"{fmt(a, 2)} ({b}, {c_})" for a, b, c_ in zip(tb["F"], tb["gl1"], tb["gl2"])],
                                  "p": tb["p"].map(fmt_p), "η²": tb["eta2"].map(lambda x: fmt(x, 3)),
                                  "Levene p": tb["levene_p"].map(lambda x: fmt(x, 3)), "Kruskal-Wallis p": tb["kruskal_p"].map(fmt_p),
                                  "q (BH)": tb["q"].map(fmt_p)})
            ancho(st.dataframe, vista, hide_index=True)

        html(seccion("Coeficientes no paramétricos", "Correlación entre las dimensiones y con los hábitos"))
        cr = correlaciones(df)
        a, b = st.columns([1, 1.35])
        with a:
            html(stat("Calidad del sueño × funcionamiento académico", f"ρ = {fmt(cr['spearman'], 2)}",
                         [f"Spearman, asociación {fuerza(cr['spearman'])}",
                          f"IC 95 % [{fmt(cr['sp_lo'], 2)}; {fmt(cr['sp_hi'], 2)}] · p {p_txt(cr['spearman_p'])}",
                          f"τ-b de Kendall = {fmt(cr['kendall'], 2)} · p {p_txt(cr['kendall_p'])}",
                          f"Pearson (referencia) = {fmt(cr['pearson'], 2)} · n = {cr['n']}"]))
            html(callout("P19, P20 y P22 mencionan el cansancio o el sueño, así que parte de la correlación puede deberse a "
                            "contenido compartido entre dimensiones.", "warn"))
        with b:
            mostrar(dispersion(df, cr))
        mnp_, _ = matriz_no_parametrica(df)
        vista = pd.DataFrame({"Variable 1": mnp_["var1"], "Variable 2": mnp_["var2"],
                              "Spearman ρ": mnp_["spearman"].map(lambda x: fmt(x, 3)), "Kendall τ-b": mnp_["kendall"].map(lambda x: fmt(x, 3)),
                              "p (Spearman)": mnp_["p_spearman"].map(fmt_p), "q (BH)": mnp_["q_BH"].map(fmt_p)})
        with st.expander("Matriz completa de correlaciones no paramétricas (puntajes y hábitos)"):
            ancho(st.dataframe, vista, hide_index=True)

# ================================================================== 4b · COMPLEMENTO
with tabs[4]:
    html(seccion("Complemento", "Análisis pregunta a pregunta",
                    "Como complemento de los puntajes sumados: correlaciones de Spearman entre preguntas individuales, "
                    "orientadas para que un valor alto signifique «mejor»."))
    if suficiente():
        r = boot(tuple(df["CALIDAD_SUENO"]), tuple(df["DESEMPENO_ACAD"]))
        a, b = st.columns([1, 2.2])
        with a:
            if np.isnan(r["rho"]):
                html(callout("No hay variación suficiente en la selección para calcular la correlación.", "warn"))
            else:
                html(stat("Calidad del sueño (P15) × desempeño (P24)", f"ρ = {fmt(r['rho'], 2)}",
                             [f"Asociación {fuerza(r['rho'])}",
                              f"IC 95 % bootstrap [{fmt(r['lo'], 2)}; {fmt(r['hi'], 2)}]",
                              f"p {p_txt(r['p'])} · τ-b = {fmt(r['tau'], 2)}",
                              f"n = {r['n']}"]))
        with b:
            mostrar(asociacion_principal(df))
        m = matriz_sueno_desempeno(df)
        mostrar(forest(m))
        a, b = st.columns([1.15, 1])
        with a:
            mostrar(matriz(m))
        with b:
            top = m.sort_values("rho", ascending=False).head(3)
            html(seccion("Lectura", "Qué aspecto del sueño pesa más"))
            for _, t in top.iterrows():
                html(mini(f"ρ = {fmt(t['rho'], 2)}", f"{t['sueno']} × {t['desempeno']} · q {p_txt(t['q'])}"))
            html(callout("Las horas de sueño (P8) y la regularidad (P14) se asocian poco con el desempeño; la calidad "
                            "percibida y la somnolencia, mucho más. Los ítems P20 y P22 mencionan el cansancio, así que "
                            "la relación más limpia es P15 × P24.", "info"))
        mostrar(habitos(habitos_tabla(df)))

        html(seccion("Explorador", "Compara cualquier par de preguntas"))
        a, b = st.columns(2)
        vx = a.selectbox("Pregunta X", ORD, index=ORD.index("FREC_DISPOSITIVOS"), format_func=lambda x: LAB[x])
        vy = b.selectbox("Pregunta Y", ORD, index=ORD.index("CALIDAD_SUENO"), format_func=lambda x: LAB[x])
        if vx == vy:
            html(callout("Elige dos preguntas distintas.", "warn"))
        else:
            rr = spearman_fz(df[vx], df[vy])
            a, b = st.columns([1, 2.2])
            with a:
                if np.isnan(rr["rho"]):
                    html(callout("No hay variación suficiente para calcular la correlación.", "warn"))
                else:
                    html(stat("rho de Spearman (códigos originales)", f"ρ = {fmt(rr['rho'], 2)}",
                                 [f"Asociación {fuerza(rr['rho'])}",
                                  f"IC 95 % [{fmt(rr['lo'], 2)}; {fmt(rr['hi'], 2)}] (Fisher z)",
                                  f"p {p_txt(rr['p'])} · n = {rr['n']}",
                                  "Valor p sin ajuste por comparaciones múltiples."]))
            with b:
                mostrar(cruce(df, vx, vy))

# ================================================================== 6 · PROGRAMAS
with tabs[5]:
    html(seccion("Objetivo 6", "Comparación entre programas",
                    "ANOVA de un factor de los puntajes por programa, con Levene, Welch, Kruskal-Wallis y Tukey. También "
                    "se muestra el marco, la cuota y las respuestas por programa."))
    mostrar(programas(df_all))
    cnt = df_all["PROGRAMA"].value_counts()
    tabla = pd.DataFrame({"Programa": [ET["PROGRAMA"][k] for k in N_H], "En el marco (Nₕ)": list(N_H.values()),
                          "Cuota (nₕ)": list(CUOTA.values()), "Respuestas": [int(cnt.get(k, 0)) for k in N_H]})
    tabla["Tasa de respuesta"] = (tabla["Respuestas"] / tabla["En el marco (Nₕ)"] * 100).map(lambda x: fmt(x, 1) + " %")
    tabla["Peso wₕ"] = (tabla["En el marco (Nₕ)"] / tabla["Respuestas"].replace(0, np.nan)).map(lambda x: fmt(x, 2))
    ancho(st.dataframe, tabla, hide_index=True)

    aviso_filtro()
    if suficiente():
        cuenta = df["PROGRAMA_ETQ"].value_counts()
        visibles = cuenta[cuenta >= MIN_N].index
        dprog = df[df["PROGRAMA_ETQ"].isin(visibles)].copy()
        dprog["G_PROGRAMA"] = dprog["G_PROGRAMA"].cat.remove_unused_categories()
        if len(visibles) < 2:
            html(callout("Se necesitan al menos dos programas con 5 o más respuestas para comparar.", "warn"))
        else:
            res = {k: anova(dprog, k, "PROGRAMA") for k in DIMENSIONES}
            qs = bh([r["p_principal"] for r in res.values()])
            for (k, r), q in zip(res.items(), qs):
                d = DIMENSIONES[k]
                a, b = st.columns([1, 2.2])
                with a:
                    html(stat(d["nombre"], f"F = {fmt(r['F'], 2)}",
                                 [f"gl = ({r['gl1']}, {r['gl2']}) · p {p_txt(r['p'])} · q (BH) {p_txt(q)}",
                                  f"η² = {fmt(r['eta2'], 3)} · ω² = {fmt(r['omega2'], 3)}",
                                  f"Levene p = {fmt(r['levene_p'], 3)} · Welch p {p_txt(r['welch_p'])}",
                                  f"Kruskal-Wallis p {p_txt(r['kruskal_p'])}"]))
                    if r["tukey"] is not None:
                        sig = r["tukey"][r["tukey"]["p"] < .05]
                        txt = "; ".join(f"{x.grupo_a} vs. {x.grupo_b} (dif. {fmt(x.diferencia, 2)}, p = {fmt(x.p, 3)})" for x in sig.itertuples()) or "ningún par significativo"
                        html(callout(f"<b>Tukey HSD:</b> {txt}.", "info"))
                with b:
                    mostrar(boxplot_programas(dprog, k, d, f"ANOVA F({r['gl1']}, {r['gl2']}) = {fmt(r['F'], 2)}; p {p_txt(r['p'])}; η² = {fmt(r['eta2'], 2)}"))
            html(callout("Una diferencia entre programas que no se mantiene al ajustar por comparaciones múltiples (q ≥ 0,05) "
                            "se considera exploratoria: con 10 a 12 respuestas por programa conviene confirmarla con más datos.", "warn"))
        ocultos = cuenta[(cuenta > 0) & (cuenta < MIN_N)]
        if len(ocultos):
            st.caption("Ocultos por privacidad (menos de 5 respuestas): " + ", ".join(map(str, ocultos.index)))
        with st.expander("Distribución de una pregunta por programa"):
            opciones = [v for v in VARS if v != "PROGRAMA"]
            v = st.selectbox("Pregunta", opciones, index=opciones.index("DESEMPENO_ACAD"), format_func=lambda x: LAB[x])
            if len(visibles):
                mostrar(dist_programa(df[df["PROGRAMA_ETQ"].isin(visibles)], v))

    html(seccion("Sensibilidad", "¿Cambian los resultados al ponderar por programa?",
                    "Cada respuesta se pondera por wₕ = Nₕ / nₕ para devolver a cada programa su peso real en el marco."))
    tp, rho_w = ponderado(df_all)
    if tp is not None:
        v = tp.copy()
        v["Sin ponderar"] = v["sin"].map(lambda x: fmt(x) + " %")
        v["Ponderado"] = v["pond"].map(lambda x: fmt(x) + " %")
        v["Diferencia (pp)"] = v["dif"].map(lambda x: ("+" if x > 0 else "") + fmt(x))
        st.dataframe(v[["indicador", "Sin ponderar", "Ponderado", "Diferencia (pp)"]].rename(columns={"indicador": "Indicador"}),
                     hide_index=True, **ANCHO_KW)
        st.caption("La ponderación corrige la desproporción entre programas, pero no la autoselección dentro de cada programa.")

# ================================================================== 7 · METODOLOGÍA
with tabs[6]:
    html(seccion("Metodología", "Cómo se hizo el estudio"))
    a, b = st.columns(2)
    with a:
        ficha = pd.DataFrame([
            ("Pregunta", "¿Qué relación hay entre los hábitos y la calidad del sueño y el desempeño académico percibido?"),
            ("Diseño", "Cuantitativo, no experimental, transversal, descriptivo-correlacional"),
            ("Población", "Estudiantes de tercer semestre de 8 programas de Ingeniería, U. de La Sabana, 2026-2"),
            ("Marco", f"Grupos de WhatsApp de la inducción 2025-2, corte 29/10/2026, sin inductores (N = {N_POB})"),
            ("Muestra planeada", f"n = {N_PLAN} (95 % de confianza, e = 5 %, p = 0,5) con afijación proporcional"),
            ("Muestra lograda", f"{N} respuestas voluntarias ({fmt(N / N_POB * 100)} % del marco); no probabilística"),
            ("Instrumento", "Microsoft Forms: consentimiento + 26 preguntas cerradas; revisado por el prof. Julián Obando"),
            ("Recolección", "22 de septiembre al 15 de noviembre de 2026"),
        ], columns=["Elemento", "Descripción"])
        ancho(st.dataframe, ficha, hide_index=True)
    with b:
        plan = pd.DataFrame([
            ("1–4", "Descripción", "Frecuencias, %, mediana y RIC por categoría"),
            ("3–4", "Dimensiones", "Ítems invertidos; α de Cronbach (calidad 0,82; funcionamiento 0,76; hábitos 0,13, no se suma); suma de puntuaciones"),
            ("3–4", "Supuestos", "Media, DE, asimetría, curtosis; Shapiro-Wilk, histograma y Q-Q; Levene"),
            ("2 y 5", "Bivariado", "ANOVA de cada puntaje por hábito (3 niveles) + box plots; Welch, Kruskal-Wallis, Tukey; BH"),
            ("5", "Correlación", "Spearman y τ-b de Kendall entre puntajes y con hábitos (no paramétricos)"),
            ("6", "Programas", "ANOVA de un factor + Tukey; χ² Monte Carlo para nominales"),
            ("—", "Sensibilidad", "Ponderación por posestratificación wₕ = Nₕ / nₕ"),
        ], columns=["Objetivo", "Análisis", "Procedimiento"])
        ancho(st.dataframe, plan, hide_index=True)
    html(seccion("Alcance", "Limitaciones"))
    html(callout("<b>Muestra voluntaria:</b> respondió el 37 % del marco; puede haber sesgo de no respuesta.<br>"
                    "<b>Marco no oficial:</b> no se contrastó con Registro Académico.<br>"
                    "<b>Autoinforme:</b> sueño y desempeño son percepciones, no mediciones ni notas.<br>"
                    "<b>Instrumento:</b> un experto y sin piloto; franjas horarias con límites repetidos y P23 que "
                    "presupone falta de sueño.<br><b>Diseño transversal:</b> asociaciones, no causas ni dirección.", "warn"))
    html(seccion("Datos", "Descargas"))
    a, b = st.columns(2)
    datos = df_all[(["ID_ENCUESTA"] if "ID_ENCUESTA" in df_all.columns else []) + VARS]
    a.download_button("⬇️ Base codificada (CSV, 87 × 27)", datos.to_csv(index=False).encode("utf-8-sig"),
                      "base_sueno_desempeno.csv", "text/csv", **ANCHO_KW)
    dic = pd.DataFrame([(CB[v][0], v, CB[v][1], CB[v][2],
                         "; ".join(f"{k} {e}" for k, e in ET[v].items())) for v in VARS],
                       columns=["Pregunta", "Variable", "Descripción", "Escala", "Códigos"])
    b.download_button("⬇️ Diccionario de variables (CSV)", dic.to_csv(index=False).encode("utf-8-sig"),
                      "diccionario_sueno_desempeno.csv", "text/csv", **ANCHO_KW)
    with st.expander("Ver diccionario de variables"):
        ancho(st.dataframe, dic, hide_index=True)

html(pie("Rusinque Covaleda, S., González Lozano, E. y Cárdenas Zambrano, S. (2026). <i>Hábitos de sueño y "
            "desempeño académico percibido en estudiantes de tercer semestre de Ingeniería de la Universidad de La "
            "Sabana, 2026-2</i>. Proyecto de Recolección de Datos, Facultad de Ingeniería, Universidad de La Sabana. "
            "La base no contiene nombres, correos ni marcas de tiempo."))
