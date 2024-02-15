# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:20:07 2023

@author: Usuario

Se pretende generar un árbol de decisión con los datos de iris.

Se aplica el metodo GridSearchCV para poder seleccionar el número óptimo del 
valor del parámetro para el árbol de decisión.

Se parten de los datos de iris para generar el árbol de decisión
target: 0-setosa, 1-versicolor, 2-virginica

Se crean dos gráficos para ver como se distribuyen los datos, uno a nivel de 
sépalos y otro a nivel de pétalos. En ambos gráficos se observa que hay un grupo 
de flores, 0-setosa que aparece en color naranja, que se diferencia muy bien 
de las otras dos.

    
"""
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Cargar el conjunto de datos de iris
iris = load_iris()

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

#X_train está compuesto de ndarray de (112, 4) 
#X_test está compuesto de ndarray de (38, 4) 

# traslado la info a un df para ver cómo se distribuyen los datos y graficarlos
df1 = pd.DataFrame(iris.data, columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df2 = pd.DataFrame(iris.target, columns = ['target'])
#convertir el target que viene en float a int
df2 = df2.convert_dtypes(infer_objects=True, convert_integer=True)
#df -> union de df1 y df2
df = pd.concat([df1,df2], axis=1)
# lo mismo que el concat pero con merge df = pd.merge(df1, df2, left_index=True, right_index=True)


#Útiles para componer los gráficos de dispersión
colores = ['orange', 'blue', 'yellow'] 
labl = ['0-setosa', '1-versicolor', '2-virginica']

info_leyenda = [(color, txt) 
                for i, color in enumerate(colores) 
                for j, txt in enumerate(labl) if i==j]


plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


# gráfico-1 para comparar los SEPALOS de las flores
f1 = df['sepal_length'].values
f2 = df['sepal_width'].values
 
asignar=[]
asignar2=[]
for index, row in df.iterrows(): 
    asignar.append(colores[row['target']])
    asignar2.append(labl[row['target']])

plt.subplot(3,2,1)
plt.title("Gráfico de dispersión - SÉPALOS")
plt.scatter(f1, f2, c=asignar, label=asignar2, s=20)
plt.axis([4,8,1,5]) # valores para 'sepal_length', 'sepal_width'
plt.xlabel('sepal_length (cm)')
plt.ylabel('sepal_width (cm)')
# no logro pintar la leyenda con esta sentencia, habría que hacer una leyenda adhoc
#plt.legend(loc="upper left", asignar2)

                        
# gráfico-2 para comparar los PETALOS de las flores
f3 = df['petal_length'].values
f4 = df['petal_width'].values
 
asignar3=[]
asignar4=[]
for index, row in df.iterrows(): 
    asignar3.append(colores[row['target']])
    asignar4.append(labl[row['target']])

plt.subplot(3,2,2)
plt.title("Gráfico de dispersión - PÉTALOS")
plt.scatter(f3, f4, c=asignar3, label=asignar4, s=20)
plt.axis([1,7,0,3])  # valores para 'petal_length', 'petal_width'
plt.xlabel('petal_length (cm)')
plt.ylabel('petal_width (cm)')

plt.show()

# GridSearchCV
#Definir el espacio de búsqueda de hiperparámetros
param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

# Crear un modelo de árbol de decisión
tree_z = DecisionTreeClassifier(random_state=0)

# Realizar la búsqueda de la cuadrícula para encontrar los mejores hiperparámetros
# cv es un entero e indica el parametro de validacion cruzada, 5 es el valor por defecto si no indicas nada
grid = GridSearchCV(tree_z, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

# Imprimir los mejores hiperparámetros y la puntuación de validación cruzada
print("Mejores hiperparámetros: ", grid.best_params_) #3
print("Puntuación de validación cruzada: ", grid.best_score_) # 0.9644

# Evaluar el modelo en los datos de prueba
print("Precisión del modelo en los datos de prueba: ", grid.score(X_test, y_test)) #0.9736


# Decision Tree
# Crear los arrays de entrenamiento y las etiquetas que indican los tres tipos 
#de flores 
y_train_tree = df['target']
x_train_tree = df.drop(['target'], axis=1).values 
 
# Crear Arbol de decision con profundidad = 2
decision_tree = tree.DecisionTreeClassifier(criterion='gini',
                                            min_samples_split=20,
                                            min_samples_leaf=15,
                                            max_depth = 2, random_state=42)
decision_tree.fit(x_train_tree, y_train_tree)

# acc_decision_tree = round(decision_tree.score(x_train_tree, y_train_tree) * 100, 2)
# print(acc_decision_tree)

#ValueError: Classification metrics can't handle a mix of unknown and 
# multiclass targets
#parece que no puede calcular la prob pq el target puede tomar 3 valores posibles (0,1,2)

# pintamos los valores que toman los datos en un punto donde cambia el tipo de flor 
# del 47 al 49 son 0 y del 50 al 52 son 1
# resulta en [0 0 0 1 1 1]
print( decision_tree.predict(x_train_tree[47:53]))#iris.data[47:53]) )

# si queremos saber las probabilidades podemos usar el método predict_proba
print( decision_tree.predict_proba(x_train_tree[47:53]))  #iris.data[47:53]) )

# para visualizar el árbol
fig = plt.figure(figsize=(14,6))
_ = plot_tree(decision_tree, 
              feature_names=list(df.drop(['target'], axis=1)),
              filled=True,
              class_names=labl,
              fontsize=10)

