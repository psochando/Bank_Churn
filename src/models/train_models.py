import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


X = pd.read_csv('https://raw.githubusercontent.com/psochando/bank_CHURN/main/DATA/processed/dataset_completo_predictoras.csv')
y = pd.read_csv('https://raw.githubusercontent.com/psochando/bank_CHURN/main/DATA/processed/dataset_completo_target.csv')

svc = SVC(C = 3, class_weight = 'balanced', degree = 3, kernel = 'poly', coef0 = 0.5, probability = True)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', svc)
])

pipeline.fit(X, y)

joblib.dump(pipeline, r'C:\Users\Pablo\Documents\GitHub\bank_CHURN\MODELS\SVC')