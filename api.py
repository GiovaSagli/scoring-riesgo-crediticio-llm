from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from google import genai

modelo = joblib.load('modelo_lgbm_opt.pkl')
columnas = joblib.load('columnas.pkl')



GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
cliente_genai = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="API Scoring Crediticio")

class Cliente(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.post("/predecir")
def predecir(cliente: Cliente):
    datos = pd.DataFrame([cliente.dict()])
    datos['ratio_uso_credito'] = datos['BILL_AMT1'] / (datos['LIMIT_BAL'] + 1)
    datos['total_deuda'] = datos[[f'BILL_AMT{i}' for i in range(1,7)]].sum(axis=1)
    datos['total_pagado'] = datos[[f'PAY_AMT{i}' for i in range(1,7)]].sum(axis=1)
    datos['ratio_pago_deuda'] = datos['total_pagado'] / (datos['total_deuda'] + 1)
    datos['tendencia_pago'] = datos['PAY_0'] - datos['PAY_6']
    datos = datos[columnas]
    prediccion = int(modelo.predict(datos)[0])
    probabilidad = float(modelo.predict_proba(datos)[0][1])
    riesgo = 'ALTO' if prediccion == 1 else 'BAJO'
    prompt = f'Eres analista de riesgo crediticio. Explica en 3-4 frases en español para un director de banco. Riesgo: {riesgo}. Probabilidad de impago: {probabilidad:.1%}. La explicacion debe ser profesional.'
    respuesta = cliente_genai.models.generate_content(model='gemini-3.6-flash', contents=prompt)
    return {"riesgo": riesgo, "probabilidad_impago": f"{probabilidad:.1%}", "explicacion": respuesta.text}