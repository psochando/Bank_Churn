import pandas as pd
import joblib
import requests
import sys
from io import BytesIO
from utils import intro
from info_paquete import __version__, __description__
from utils import consola as console
import errores

ERROR_ENTRADA = 1
ERROR_SALIDA = 2
ERROR_GENERAL = 99

data = pd.read_csv('https://raw.githubusercontent.com/psochando/bank_CHURN/main/DATA/processed/dataset_completo_predictoras.csv')

# URL del modelo en GitHub
url_modelo = 'https://github.com/psochando/bank_CHURN/raw/main/MODELS/SVC1'
# Descargar el modelo desde la URL de GitHub
response = requests.get(url_modelo)
# Cargar el modelo directamente desde la respuesta HTTP
SVC1 = joblib.load(BytesIO(response.content))

# Carpeta de salida
fichero_salida = r'output/output.csv'

def main():
    """
    Método principal
    """
    # Introducción
    intro(__description__, __version__)

    try:
        with console.status('[magenta]Trabajando...') as status:

            # Cargamos los datos y el modelo
            str_status = f'Cargando datos...'
            console.log(str_status)
            status.update(status=str_status)

            X = data

            str_status = f'Cargando modelo...'
            console.log(str_status)
            status.update(status=str_status)

            model = SVC1

            str_status = f'Aplicando modelo...'
            console.log(str_status)
            status.update(status=str_status)

            probs = model.predict_proba(X)
            preds = model.predict(X)

            # Se crea un archivo con las predicciones del modelo y las respectivas probabilidades
            str_status = f'Generando salida...'
            console.log(str_status)
            status.update(status=str_status)

            results = pd.DataFrame({'Prediction': preds, 'Probability_cancels': probs[:, 1]})
            results.to_csv(fichero_salida, index = False)

            str_status = f'Fichero de salida generado en {fichero_salida}'
            console.log(str_status)
            status.update(status=str_status)

        console.log('[green]Proceso finalizado')

    except errores.InputError as err:
        console.log(f'[traceback.text]Se ha producido un error de entrada:[\]\n'
                    f'[traceback.text]{err}[traceback.text]')

        sys.exit(ERROR_ENTRADA)

    except errores.OutputError as err:
        console.log(f'[traceback.text]Se ha producido un error de salida:[\]\n'
                    f'[traceback.text]{err}[traceback.text]')

        sys.exit(ERROR_SALIDA)

if __name__ == '__main__':
    main()