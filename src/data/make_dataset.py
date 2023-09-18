import pandas as pd

def outliers_dealer(df):

    indices_outliers = df[df.CreditScore <= 382].index.values
    df.CreditScore.iloc[indices_outliers] = df.CreditScore.mean()

    indices_outliers = df[df.Age >= 63].index.values
    df.Age.iloc[indices_outliers] = df.Age.median()

    df.NumOfProducts = df.NumOfProducts.replace(4, df.NumOfProducts.median())
    
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
    df = df.drop_duplicates()
    
    cols_to_dummies = dummies_or_not(df)
    df_dum = pd.get_dummies(df[cols_to_dummies], drop_first = True)
    df_not_dum = df.drop(columns = cols_to_dummies)
    df = pd.concat([df_not_dum, df_dum], axis = 1)
    
    df = outliers_dealer(df)
    
    X = df.drop(columns = target)
    X.to_csv(r'C:\Users\Pablo\Documents\GitHub\bank_CHURN\DATA\processed\dataset_completo_predictoras.csv', index=False)
    y = df[target]
    y.to_csv(r'C:\Users\Pablo\Documents\GitHub\bank_CHURN\DATA\processed\dataset_completo_target.csv', index=False)


# path = 'https://raw.githubusercontent.com/psochando/bank_churn_/main/data_/raw/Churn_Modelling.csv'
drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'HasCrCard', 'EstimatedSalary', 'Tenure']
process_data(path, drop_cols, 'Exited')