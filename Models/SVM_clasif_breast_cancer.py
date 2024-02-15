# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:46:07 2023

@author: Usuario

Se pretende generar un SVM (Support Vector Machine) para clasificar datos de 
cancer de mama.

Se parte de un conjunto de datos de cancer de mama que se encuentra en la librería
sklearn. N=569 valores con 30 vbles numéricas y un campo target 
(0-Benigno, 1-Maligno).

Se aplica un modelo SVM de kernel lineal, se entrena y se realiza la previsión.

Se genera una matriz de confusión y un resumen de algunos scores.

Tambien se prueba SVM con el kernel Gaussian Kernel (RBF) que indica que da 
buenos resultados, previamente buscamos el valor óptimo de los parámetros (C).
Este kernel tiene dos ventajas: que solo tiene dos hiperparámetros que optimizar 
( γ y la penalización  C común a todos los SVM) y que su flexibilidad puede ir 
desde un clasificador lineal a uno muy complejo.

Accuracy 
    - lineal: 0.9649, mejor por poco
    - RBF:    0.9496, quiza se podria optimizar el valor del hp gamma
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm #Import svm model
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

#Load dataset
cancer = datasets.load_breast_cancer()

#<class 'sklearn.utils._bunch.Bunch'>
# print(cancer.feature_names)
# print(cancer.target_names)
# print(cancer.data)
# print(cancer.target)


X_train, X_test, y_train, y_test = train_test_split(cancer.data, 
                                                    cancer.target, 
                                                    test_size=0.3, 
                                                    random_state=42)

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #0.9649

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print(cm)
print(cr)

# REAL_0 [[ 59       4]
# REAL_1  [  2     106]]
#        PRED_0  PRED_1


#               precision    recall  f1-score   support

#            0       0.97      0.94      0.95        63
#            1       0.96      0.98      0.97       108

#     accuracy                           0.96       171
#    macro avg       0.97      0.96      0.96       171
# weighted avg       0.96      0.96      0.96       171

print("prueba RBF")
# prueba con kernel Gaussian Kernel (RBF) que indica que da buenos resultados, 
# previamente buscamos el valor óptimo de los parámetros
# Grid de hiperparámetros
# ==============================================================================
param_grid = {'C': np.logspace(-5, 7, 20)}

# Búsqueda por validación cruzada
# ==============================================================================
grid = GridSearchCV(
        estimator  = SVC(kernel= "rbf", gamma='scale'),
        param_grid = param_grid,
        scoring    = 'accuracy',
        n_jobs     = -1,
        cv         = 3, 
        verbose    = 2,
        return_train_score = True,
        refit = True
      )

# Se asigna el resultado a _ para que no se imprima por pantalla
_ = grid.fit(X = X_train, y = y_train)

# Resultados del grid
# ==============================================================================
resultados = pd.DataFrame(grid.cv_results_)
resultados.filter(regex = '(param.*|mean_t|std_t)')\
    .drop(columns = 'params')\
    .sort_values('mean_test_score', ascending = False) \
    .head(5)

print(resultados.head(5))
# Mejores hiperparámetros por validación cruzada
# ==============================================================================
print("----------------------------------------")
print("Mejores hiperparámetros encontrados (cv)")
print("----------------------------------------")
print(grid.best_params_, ":", grid.best_score_, grid.scoring)

# {'C': 1623.7767391887176} : 0.9496 accuracy

# con refit = True, el modelo entrenado con los valores óptimos se encuentra en
# grid.best_estimator_
modelo = grid.best_estimator_
predicciones = modelo.predict(X = X_test)

cm = confusion_matrix(y_test, predicciones)
cr = classification_report(y_test, predicciones)
print(cm)
print(cr)

# [[ 58   5]
#  [  1 107]]
#               precision    recall  f1-score   support

#            0       0.98      0.92      0.95        63
#            1       0.96      0.99      0.97       108

#     accuracy                           0.96       171
#    macro avg       0.97      0.96      0.96       171
# weighted avg       0.97      0.96      0.96       171
