import pandas as pd

def outliers_dealer(df):

    indices_outliers = df[df['Age'] >= 63].index.values
    df['Age'].iloc[indices_outliers] = df['Age'].median()

    df['NumOfProducts'] = df['NumOfProducts'].replace(4, df['NumOfProducts'].median())
    
    return df


def dummies_or_not(df):
    dum_list = []
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            dum_list.append(col)
    return dum_list
            

def process_data(path, drop_cols, target):
    
    df = pd.read_csv(path)

    df.drop(columns = drop_cols, inplace = True)
    # df = df.drop_duplicates() # se pierden casi 3000 registros si hacemos drop_duplicates despues de eliminar las columnas consideradas. Se ha decidido mantener estos registros repetidos, pues no se repiten por error sino que realmente dichos clientes presentan estos valores en las variables finalmente seleccionadas.
    
    cols_to_dummies = dummies_or_not(df)
    df_dum = pd.get_dummies(df[cols_to_dummies], drop_first = True)
    df_not_dum = df.drop(columns = cols_to_dummies)
    df = pd.concat([df_not_dum, df_dum], axis = 1)
    
    df = outliers_dealer(df)
        
    X = df.drop(columns = target)
    X.to_csv(r'C:\Users\Pablo\Documents\GitHub\bank_CHURN\DATA\processed\dataset_completo_predictoras.csv', index=False)
    y = df[target]
    y.to_csv(r'C:\Users\Pablo\Documents\GitHub\bank_CHURN\DATA\processed\dataset_completo_target.csv', index=False)


path = 'https://raw.githubusercontent.com/psochando/bank_CHURN/main/DATA/raw/Churn_Modelling.csv'
drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'HasCrCard', 'EstimatedSalary', 'Tenure']
process_data(path, drop_cols, 'Exited')