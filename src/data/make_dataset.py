import sys
# Agregamos la ruta al directorio principal 'banck_CHURN' al PYTHONPATH para luego poder acceder a los modulos con rutas absolutas.
sys.path.append(r'C:\Users\Pablo\Documents\GitHub\bank_CHURN')
import pandas as pd
from src.visualization_analysis.analysis import dumm_var, num_var, asymmetry, outliers_detector
data = pd.read_csv('https://raw.githubusercontent.com/psochando/bank_CHURN/main/DATA/raw/Churn_Modelling.csv')

data

# Automatiza el tratamiento de outliers para características numéricas, imputando por la mediana cuando la asimetria de la distribucion es "grande" y por la media cuando no
# Devuelve el dataframe actualizado
def outliers_dealer(df = data, asym = 0.5):
    a = asymmetry(df)
    for col in num_var(df.drop(columns = 'Exited')):
        outliers_indices = outliers_detector(df)[col].index
        if abs(a[col]) > asym:
            df[col].iloc[outliers_indices] = df[col].median()
        else:
            df[col].iloc[outliers_indices] = df[col].mean()
    return df
            

# Función principal en el procesamiento de datasets
def process_data(drop_cols, target, df = data, split_X_y = False):

    df = df.drop(columns = drop_cols)
    # df = df.drop_duplicates() # se pierden casi 3000 registros si hacemos drop_duplicates despues de eliminar las columnas consideradas. Se ha decidido mantener estos registros repetidos, pues no se repiten por error sino que realmente dichos clientes presentan estos valores en las variables finalmente seleccionadas.
    
    df = outliers_dealer(df) # tratamos los outliers en las variables numéricas del dataframe
    
    cols_to_dummies = dumm_var(df)
    df_dumm = pd.get_dummies(df[cols_to_dummies], drop_first = True) # codificamos las columnas categóricas
    df_not_dumm = df.drop(columns = cols_to_dummies)
    df = pd.concat([df_not_dumm, df_dumm], axis = 1)
    
    if split_X_y == True:  # posibilidad de separar en datos en variables predictoras y target
        X = df.drop(columns = target)
        X.to_csv(r'C:\Users\Pablo\Documents\GitHub\bank_CHURN\DATA\processed\dataset_completo_predictoras.csv', index=False)
        y = df[target]
        y.to_csv(r'C:\Users\Pablo\Documents\GitHub\bank_CHURN\DATA\processed\dataset_completo_target.csv', index=False)
        
    return df


drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'HasCrCard', 'EstimatedSalary', 'Tenure']
process_data(drop_cols, 'Exited', split_X_y = False)