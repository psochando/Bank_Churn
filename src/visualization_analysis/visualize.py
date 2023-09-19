import sys
# Agregamos la ruta al directorio principal 'banck_CHURN' al PYTHONPATH para luego poder acceder a los modulos con rutas absolutas.
sys.path.append(r'C:\Users\Pablo\Documents\GitHub\bank_CHURN')
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
from src.visualization_analysis.analysis import num_var
data = pd.read_csv('https://raw.githubusercontent.com/psochando/bank_CHURN/main/DATA/raw/Churn_Modelling.csv')


def numvar_dist(df = data):
    
    numvar_list = num_var()
    fig, axes = plt.subplots(len(numvar_list), 2, figsize=(10, 5 * len(numvar_list)))

    for i, col in enumerate(numvar_list):
        # Histograma
        sns.histplot(df[col], ax=axes[i, 0])
        axes[i, 0].set_title(f'{col}')
        
        # Diagrama de caja (boxplot)
        sns.boxplot(df[col], ax=axes[i, 1])
        axes[i, 1].set_title(f'{col}')

    plt.show()



def plot_model(model, y_test, y_pred, fpr, tpr, cm = True, roc = True):
    
    if cm == True:
        cm = confusion_matrix(y_test, y_pred, labels = model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)  
        disp.plot()
        plt.show()
        
    if roc == True:
        plt.plot(fpr, tpr)
        plt.plot(np.linspace(0,1), np.linspace(0,1), 'r')
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.show()