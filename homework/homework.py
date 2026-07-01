# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import gzip
import pickle
import pandas as pd
import json
import os

# ===========================
# Paso 1: Limpieza de datos
# ===========================

def procesar_datos(df):
    df = df.copy()
    df.drop(columns='ID', inplace=True)
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.dropna(inplace=True)
    df = df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    return df

# ===================
# Paso 3: Pipeline
# ===================

def construir_pipeline():
    categorias = ['SEX', 'EDUCATION', 'MARRIAGE']
    continuas = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    transformador = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorias),
            ('numerical', MinMaxScaler(), continuas)
        ],
        remainder='passthrough'
    )

    seleccionador = SelectKBest(score_func=f_classif)

    pipe = Pipeline(steps=[
        ('preprocessing', transformador),
        ('feature_selection', seleccionador),
        ('logistic', LogisticRegression(max_iter=1000, solver='saga', random_state=42))
    ])
    return pipe

# ========================================
# Paso 4: Optimización de hiperparámetros
# =========================================

def ajustar_modelo(pipeline, cv_folds, X_train, y_train, metric):
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={
            'feature_selection__k': range(1, 11),
            'logistic__penalty': ['l1', 'l2'],
            'logistic__C': [0.001, 0.01, 0.1, 1, 10, 100],
        },
        cv=cv_folds,
        scoring=metric,
        verbose=0
    )
    grid.fit(X_train, y_train)
    return grid


# ============================
# Paso 5: Cálculo de Métricas
# ============================

def obtener_metricas(modelo, X_train, y_train, X_test, y_test):
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, y_pred_train),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_pred_train),
        'recall': recall_score(y_train, y_pred_train),
        'f1_score': f1_score(y_train, y_pred_train)
    }

    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, y_pred_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'f1_score': f1_score(y_test, y_pred_test)
    }

    return metrics_train, metrics_test


# ============================
# Paso 6: Matriz de confusión
# ============================

def generar_confusiones(modelo, X_train, y_train, X_test, y_test):
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
    tn_test, fp_test, fn_test, tp_test = cm_test.ravel()

    cm_dict_train = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {'predicted_0': int(tn_train), 'predicted_1': int(fp_train)},
        'true_1': {'predicted_0': int(fn_train), 'predicted_1': int(tp_train)}
    }

    cm_dict_test = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {'predicted_0': int(tn_test), 'predicted_1': int(fp_test)},
        'true_1': {'predicted_0': int(fn_test), 'predicted_1': int(tp_test)}
    }

    return cm_dict_train, cm_dict_test

# =======================
# Paso 7: Guardar modelo
# =======================

def guardar_modelo(modelo, path):
    with gzip.open(path, 'wb') as f:
        pickle.dump(modelo, f)


# ====================
# EJECUCIÓN DEL LAB
# ====================

if __name__ == "__main__":

    # Cargar datasets
    df_train = pd.read_csv("files/input/train_data.csv.zip")
    df_test = pd.read_csv("files/input/test_data.csv.zip")

    # Limpiar datos
    df_train = procesar_datos(df_train)
    df_test = procesar_datos(df_test)

    # Separar características y objetivo
    X_train = df_train.drop(columns=['default'])
    y_train = df_train['default']
    X_test = df_test.drop(columns=['default'])
    y_test = df_test['default']

    # Construir pipeline
    modelo_pipeline = construir_pipeline()

    # Ajustar modelo
    modelo_ajustado = ajustar_modelo(modelo_pipeline, 10, X_train, y_train, 'balanced_accuracy')

    # Guardar modelo
    os.makedirs("files/models", exist_ok=True)
    guardar_modelo(modelo_ajustado, "files/models/model.pkl.gz")
    # Calcular métricas y matrices
    resultados = []

    metricas_train, metricas_test = obtener_metricas(modelo_ajustado, X_train, y_train, X_test, y_test)
    resultados.append(metricas_train)
    resultados.append(metricas_test)

    cm_train, cm_test = generar_confusiones(modelo_ajustado, X_train, y_train, X_test, y_test)
    resultados.append(cm_train)
    resultados.append(cm_test)

    # Guardar resultados
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", 'w') as f_out:
        for item in resultados:
            f_out.write(json.dumps(item) + '\n')