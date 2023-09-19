import pandas as pd
import seaborn as sns
from scipy.stats import skew
data = pd.read_csv('https://raw.githubusercontent.com/psochando/bank_CHURN/main/DATA/raw/Churn_Modelling.csv')


def num_var(df = data):
    r = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            r.append(col)
    return r

def dumm_var(df = data):
    r = []
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            r.append(col)
    return r
    

def outliers_detector(df = data[num_var()]):
    
    for col in df.columns:
        
        # Coeficiente de asimetría de Fisher
        asymmetry = skew(df[col], bias=False)
        print(f'Coef. de asimetría en {col}: {asymmetry}')
        
        # Outliers
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        print(f'Outliers en {col}: {outliers.sort_values(ascending=True).values}')
        print('Total:', len(outliers), '\n')
        