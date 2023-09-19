import sys
# Agregamos la ruta al directorio principal 'banck_CHURN' al PYTHONPATH para luego poder acceder a los modulos con rutas absolutas que partan de ahi.
sys.path.append(r'C:\Users\Pablo\Documents\GitHub\bank_CHURN')
import pandas as pd
from src.visualization_analysis.analysis import dumm_var
data = pd.read_csv('https://raw.githubusercontent.com/psochando/bank_CHURN/main/DATA/raw/Churn_Modelling.csv')


def outliers_dealer(df):

    indices_outliers = df[df['Age'] >= 63].index.values
    df['Age'].iloc[indices_outliers] = df['Age'].median()

    df['NumOfProducts'] = df['NumOfProducts'].replace(4, df['NumOfProducts'].median())
    
    return df


def process_data(drop_cols, target, df = data, split_X_y = False):

    df.drop(columns = drop_cols, inplace = True)
    # df = df.drop_duplicates() # se pierden casi 3000 registros si hacemos drop_duplicates despues de eliminar las columnas consideradas. Se ha decidido mantener estos registros repetidos, pues no se repiten por error sino que realmente dichos clientes presentan estos valores en las variables finalmente seleccionadas.
    
    cols_to_dummies = dumm_var(df)
    df_dumm = pd.get_dummies(df[cols_to_dummies], drop_first = True)
    df_not_dumm = df.drop(columns = cols_to_dummies)
    df = pd.concat([df_not_dumm, df_dumm], axis = 1)
    
    df = outliers_dealer(df)
    
    if split_X_y == True:        
        X = df.drop(columns = target)
        X.to_csv(r'C:\Users\Pablo\Documents\GitHub\bank_CHURN\DATA\processed\dataset_completo_predictoras.csv', index=False)
        y = df[target]
        y.to_csv(r'C:\Users\Pablo\Documents\GitHub\bank_CHURN\DATA\processed\dataset_completo_target.csv', index=False)
        
    return df


drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'HasCrCard', 'EstimatedSalary', 'Tenure']
process_data(drop_cols, 'Exited', split_X_y = True)