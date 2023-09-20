import pandas as pd
import seaborn as sns
from scipy.stats import skew
data = pd.read_csv('https://raw.githubusercontent.com/psochando/bank_CHURN/main/DATA/raw/Churn_Modelling.csv')

# Selecciona las variables numéricas del dataset
def num_var(df = data):
    r = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            r.append(col)
    return r

# Selecciona las variables que nos interesan para aplicar one-hot encoding
def dumm_var(df = data):
    r = []
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            r.append(col)
    return r
    

# Outliers de variables numéricas aplicando rangos intercuartílicos
# Devuelve un diccionario con los outliers para cada variable
def outliers_detector(df):
    df_num = df[num_var(df)]
    r = {}
    for col in df_num.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        r[col] = outliers
    return r
        
        
# Coeficientes de asimetría de Fisher
# Devuelve un diccionario con los coeficientes de cada distribución    
def asymmetry(df):
    df_num = df[num_var(df)]
    r = {}
    for col in df_num.columns:
        asymmetry = skew(df[col], bias=False)
        r[col] = asymmetry
    return r