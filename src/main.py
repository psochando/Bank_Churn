import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO

data = pd.read_csv('https://raw.githubusercontent.com/psochando/bank_CHURN/main/DATA/processed/dataset_completo_predictoras.csv')

# URL del modelo en GitHub
url_modelo = 'https://github.com/psochando/bank_CHURN/raw/main/MODELS/SVC1'
# Descargar el modelo desde la URL de GitHub
response = requests.get(url_modelo)
# Cargar el modelo directamente desde la respuesta HTTP
SVC1 = joblib.load(BytesIO(response.content))

def main():
    
    # Cargamos los datos y el modelo
    X = data
    model = SVC1
    
    probs = model.predict_proba(X)
    preds = model.predict(X)
    
    # Se crea un archivo con las predicciones del modelo y las respectivas probabilidades
    results = pd.DataFrame({'Prediction': preds, 'Probability_cancels': probs[:, 1]})
    results.to_csv('output.csv', index = False)
    
    
if __name__ == '__main__':
    main()