# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:20:07 2023

@author: Usuario

Se parte del dataset de Iris (contiene datos de 3 tipos de flores iris, 'setosa', 
'versicolor', 'virginica') para predecir el tipo de flor empleando para ello el 
método KNeighborsClassifier que es un clasificador de aprendizaje supervisado 
no paramétrico, que utiliza la proximidad para hacer clasificaciones o 
predicciones sobre la agrupación de un punto de datos individual.
La “K” significa la cantidad de “puntos vecinos” que tenemos en cuenta en 
las cercanías para clasificar los “n” grupos que ya se conocen de antemano, 
pues es un algoritmo supervisado. En nuestro caso los grupos son 3: 


iris es un objeto del tipo <class 'sklearn.utils._bunch.Bunch'> , que tiene 
almacenado los datos de 150 flores con 4 campos (sepal length, sepal width, 
petal length, petal width) incluyendo la etiqueta 0,1,2 que indica el tipo de 
flor que representa: 0-setosa, 1-versicolor, 2-virginica

Revisando los datos de train/test vemos que el reparto de tipos de 
flores (0,1,2) está balanceado.

Estandarización de datos. A los datos se les aplica un escalado stándar.
Existen diferentes tipos y estos son más/menos sensibles a los outliers.
 - Original data
 - StandardScaler
 - MinMaxScaler
 - MaxAbsScaler
 - RobustScaler
 - PowerTransformer
 - QuantileTransformer (uniform output / Gaussian output)


Para determinar el valor del parámetro n_neighbors se emplea un gráfico que 
llama al modelo con diferentes valores de k y se selecciona el primer valor 
que el da un mayor valor de accuracy.

CONCLUSIÓN: 
    Como pros tiene sobre todo que es sencillo de aprender e implementar, se 
    utiliza en la resolución de multitud de problemas, como en sistemas de 
    recomendación, búsqueda semántica y detección de anomalías.
    Tiene como contras que utiliza todo el dataset para entrenar “cada punto” y 
    por eso requiere de uso de mucha memoria y recursos de procesamiento (CPU). 
    Por estas razones kNN tiende a funcionar mejor en datasets pequeños y sin 
    una cantidad enorme de features (columnas).

"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pandas as pd

# Carga el conjunto de datos Iris
iris = load_iris()

#print(iris.keys())
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

#print(iris.values())

# Separa las variables independientes de la variable dependiente
X = iris.data
y = iris.target

# prueba para ver el reparto de datos
df = pd.DataFrame(X, columns = ['Column_A', 'Column_B', 'Column_C', 'Column_D'])
print(df.describe())

#print("X", X)
#print("y", y)

# Crea el escalador estándar y ajusta los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Divide el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=(40))

#print("X_train", X_train)
#print("X_test", X_test)
#print("y_train", y_train)
#print("y_test", y_test)

# Datos de train/test de y balanceados:
# train
# 0	42
# 1	38
# 2	40
#  120
# test 
# 0	 8
# 1	12
# 2	10
#   30

# Crea el modelo de clasificación utilizando k-NN
# el valor a definir para n_neighbors está explicado más abajo 
clf = KNeighborsClassifier(n_neighbors=3) 
clf.fit(X_train, y_train)

# Realiza una predicción utilizando el conjunto de prueba
y_pred = clf.predict(X_test)
 
# Como son pocos, podemos mostrar los valores predichos 
# y los valores reales y compararlos visualmente
print("y_pred", y_pred)
print("y_test", y_test)


# Muestra la precisión del modelo 1.0 sobre test y 0.95 sobre train
print("precisión del modelo sobre test", clf.score(X_test, y_test))    #1.0
print("precisión del modelo sobre train", clf.score(X_train, y_train)) #0.95

# para obtener el numero "k" de vecinos se prueba el modelo con diferentes 
# valores de k y se muestra en un gráfico la precisión del modelo
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])
