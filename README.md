# Dashboard — Sueño y desempeño académico percibido

Dashboard en Streamlit del proyecto de Recolección de Datos (tercer semestre de Ingeniería, Universidad de La Sabana, 2026-2).

## Cómo ejecutarlo en el computador

1. Instalar Python 3.10 o superior.
2. En esta carpeta, abrir una terminal y ejecutar:

```
pip install -r requirements.txt
streamlit run app.py
```

3. Se abre en el navegador (http://localhost:8501).

## Cómo publicarlo en línea (gratis)

1. Subir esta carpeta a un repositorio de GitHub (incluyendo `base_procesada.xlsx`).
2. Entrar a https://share.streamlit.io, elegir el repositorio y el archivo `app.py`.

## Datos

- Por defecto lee `base_procesada.xlsx` de esta carpeta. También acepta la base original (`BaseDeDatos.xlsx`, hoja `Base_datos`) o un Excel subido desde la barra lateral.
- Solo usa las 26 respuestas codificadas; las etiquetas, los ítems invertidos y las correlaciones se recalculan en el app.
- La base no contiene nombres, correos ni marcas de tiempo. Por privacidad, no se muestran grupos con menos de 5 respuestas.

## Pestañas

| Pestaña | Contenido |
|---|---|
| Resumen | Indicadores clave y relación principal P15 × P24 (Spearman con IC bootstrap) |
| Hábitos | Horas de sueño y hora de acostarse (semana vs. fin de semana); factores asociados |
| Calidad y desempeño | Barras divergentes P15–P24, factor principal, P23 y tablas de frecuencias |
| Asociaciones | Matriz de Spearman con ajuste Benjamini-Hochberg y explorador de cualquier par de preguntas |
| Programas | Marco, cuota y tasa de respuesta; distribución por programa; Kruskal-Wallis |
| Método y datos | Diseño, limitaciones y descarga de la selección filtrada |

Los filtros de la barra lateral (programa, sexo, edad) afectan todas las pestañas excepto la tabla del marco.

## Criterios de interpretación

- Las respuestas son categorías ordenadas: se analizan por rangos (no promedios).
- La muestra es voluntaria y autoseleccionada (87 de 235): describe a quienes respondieron.
- Asociación no implica causalidad.
