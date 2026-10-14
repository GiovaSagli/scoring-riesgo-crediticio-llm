# Sistema de Scoring de Riesgo Crediticio con Explicabilidad mediante LLM

Trabajo Fin de Máster — Máster en Data Science, Big Data & Business Analytics, Universidad Complutense de Madrid.
**Autor:** Giovanni Saglimbeni

## Qué resuelve

Un sistema de scoring de riesgo crediticio para banca que no se queda en la predicción: añade interpretabilidad (SHAP), explicabilidad en lenguaje natural mediante un LLM, y se despliega como un servicio real (API REST + base de datos cloud), simulando cómo este tipo de modelo se integraría en una entidad financiera.

## Arquitectura

```
Datos del cliente → Modelo LightGBM → Predicción + valores SHAP → LLM (Gemini) → Explicación en lenguaje natural
                                                      ↓
                                          API REST (FastAPI) + Snowflake (cloud DB)
```

## Resultados clave

- Dataset: [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default-of-credit-card-clients) (UCI ML Repository, Yeh 2009), 30.000 clientes de tarjetas de crédito.
- Comparativa de 4 modelos (Regresión Logística, Random Forest, XGBoost, LightGBM); el modelo final, LightGBM optimizado con Optuna, alcanza un **ROC-AUC de 0,78** frente al 0,66 del modelo base, detectando el **61% de los clientes que efectivamente incurren en impago**.
- La variable de ingeniería de features `ratio_uso_credito` se posiciona como la **2ª variable más importante** del modelo según SHAP, validando el proceso de feature engineering.
- Análisis de equidad del modelo por género, edad y nivel educativo (ROC-AUC estable entre grupos, con una limitación identificada en el segmento de menor tamaño muestral).
- Explicaciones en lenguaje natural generadas automáticamente para perfiles no técnicos (directores de riesgo, gestores comerciales).

## Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · Optuna · SHAP · FastAPI · Snowflake · Google Gemini API

## Estructura del repositorio

```
notebooks/   → análisis exploratorio, preparación de datos, modelado e interpretabilidad
src/         → api.py: servicio FastAPI que expone el modelo
docs/        → memoria completa del proyecto
```

## Cómo ejecutar la API

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="tu_clave_aqui"
uvicorn api:app --reload
```

La API expone un único endpoint `/predecir` que recibe los datos de un cliente en JSON y devuelve el nivel de riesgo, la probabilidad de impago y una explicación en lenguaje natural.

## Licencia de los datos

El dataset utilizado tiene licencia Creative Commons Attribution 4.0 (CC BY 4.0): Yeh, I.C. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
