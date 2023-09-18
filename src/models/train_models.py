import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
X = pd.read_csv('https://raw.githubusercontent.com/psochando/bank_CHURN/main/DATA/processed/dataset_completo_predictoras.csv')
y = pd.read_csv('https://raw.githubusercontent.com/psochando/bank_CHURN/main/DATA/processed/dataset_completo_target.csv')

svc = SVC(C = 3, class_weight = 'balanced', degree = 3, kernel = 'poly', coef0 = 0.5, probability = True)

def final_training(model, X = X, y = y, save = False):

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ]).fit(X, y)
    
    auc_cross = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    recall_cross = cross_val_score(pipeline, X, y, cv=5, scoring='recall')
    
    if save == True:
        joblib.dump(pipeline, r'C:\Users\Pablo\Documents\GitHub\bank_CHURN\MODELS\SVC')

    return (recall_cross.mean(), auc_cross.mean())

final_training(svc, save = True)