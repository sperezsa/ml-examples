# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:46:34 2023

@author: Usuario

Se pretende emplear un bosque aleatorio para mejorar las predicciones de un 
único árbol de decisión. El objetivo es predecir operaciones fraudulentas
dentro de un fichero de operaciones bancarias.

Previamente ejecutamos una LogisticRegression como estudio base para ver si 
RandomForestClassifier puede mejorar el resultado.

El entreno del modelo RandomForestClassifier es más pesado que LogisticRegression.
 
El juego de datos es un fichero de Kaggle con transacciones fraudulentas.
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 
Esta compuesto de 29 campos, la mayoría de ellos son desconocidos, el importante
es el campo Class que indica si la operación es marcada como fraudulenta o no.
0-normal, 1-fraude.
Este campo Class se encuentra desbalanceado, ya que tiene un total de 492 
operaciones fraudulentas de un total de 285.000 operaciones.

La precisión de los modelo es: 
    - LogisticRegression: 0.9754 
    - RandomForestClassifier: 0.9996
Los dos son buenos modelos. Aunque lleva más tiempo de cómputo, mejores 
resultados arroja RF. Viendo la CM, la tasa de FP y FN es menor. Lo importante 
en este estudio es tener bajos los FN, para que no se nos "cuele" ninguna 
operación fraudulenta.

"""
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path #para el control de rutas en el so
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


# rutas del fichero de entrada con los datos 
ruta_fichero = Path(".spyder-py3/Data", "Data_creditcard.csv")
print(ruta_fichero)
home = Path.home()
ruta_completa = Path(home, ruta_fichero)


# fichero de entrada
data = pd.read_csv(ruta_completa)

print(data.shape) #(284807, 31)
# print(data.head(5))
print(pd.value_counts(data['Class'], sort = True)) 
#class comparison 0=Normal 1=Fraud
# datos desbalanceados!
# 0    284315
# 1       492

LABELS = ["Normal","Fraud"]

y = data['Class']
X = data.drop('Class', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

#función para dibujar la matriz de confusión
def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print (classification_report(y_test, pred_y))

# LogisticRegression
# solver = Algoritmo a utilizar en el problema de la optimización.
model_logistic = LogisticRegression(C=1.0,penalty='l2',random_state=1,
                                    solver="newton-cg",class_weight="balanced")

model_logistic.fit(X_train, y_train)

# Con el parámetro class_weight="balanced" hacemos que el target esté balanceado.
# Con esto aumentamos los FP para reducir los FN. Es más interesante arrojar 
# mejores datos en los FN para evitar que se nos escape una operación fraudulenta.
pred_y = model_logistic.predict(X_test)
precision = metrics.accuracy_score(y_test, pred_y)
print("La precisión del modelo de LogisticRegression es:", precision) # 0.9754 
mostrar_resultados(y_test, pred_y)

# (83207 2093
#     6 137)
#    6: pocos falsos negativos (predecimos Normal cuando es Fraudulenta). 
# 2093: muchos falsos positivos, 2093. 
# En este caso de fraude, mejor tener pocos FN

#               precision    recall  f1-score   support

#            0       1.00      0.98      0.99     85300
#            1       0.06      0.96      0.12       143

#     accuracy                           0.98     85443
#    macro avg       0.53      0.97      0.55     85443
# weighted avg       1.00      0.98      0.99     85443




# RandomForestClassifier. Crear el modelo con 100 arboles
# boostrap: para utilizar diversos tamaños de muestras para entrenar. Si se pone 
# en falso, utilizará siempre el dataset completo.
# max_features: la manera de seleccionar la cantidad máxima de features para cada 
# árbol: max_features=sqrt(n_features).
model = RandomForestClassifier(n_estimators=100, 
                                bootstrap = True, verbose=2,
                                max_features = 'sqrt')
model.fit(X_train, y_train)
pred_y = model.predict(X_test)
precision = metrics.accuracy_score(y_test, pred_y)
print("La precisión del modelo RandomForestClassifier es:", precision) # 0.9996722
mostrar_resultados(y_test, pred_y)

# (85293 7
#     21 122)
# 21: pocos falsos negativos (consideramos Normal cuando es Fraudulenta). 
#  7: pocos falsos positivos. Mejor tener pocos FN


#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00     85300
#            1       0.95      0.85      0.90       143

#     accuracy                           1.00     85443
#    macro avg       0.97      0.93      0.95     85443
# weighted avg       1.00      1.00      1.00     85443



