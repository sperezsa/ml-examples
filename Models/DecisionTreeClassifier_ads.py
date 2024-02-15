# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:28:32 2023

@author: Usuario

Se pretende generar un árbol de decisión con los datos de anuncios en rrss.

Se parte de un fichero con datos sobre anuncios de redes sociales que muestran 
si los usuarios han comprado un producto haciendo clic en los anuncios que se 
les muestran. N=400 valores

Ya que los árboles de decisión tienden a sobreajustar, se generan 2 árboles, el 
primero sin configurar, este tiende a memorizar las soluciones y generar 
sobreajuste, y el segundo optimizado, definiendo el máximo nivel de profundidad 
de 3. 
Se observa que el primero seguramente se encuentre overfitting a los datos de 
entreno. 
El segundo es más sencillo de visualizar las predicciones y mejora la precisión

En la matriz de confusión se identifican 8/100 observaciones catalogadas como 
false.

Se genera un gráfico de dispersión para ver la relación entre las vbles 
independientes y la vble dependiente. Se observa que la vble Age a partir de 
45 se compra independiente del nivel salarial, y con menos de 45 se compra solo
con niveles altos de salario.
  
"""
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from pathlib import Path #para el control de rutas en el so
from sklearn.metrics import confusion_matrix

# rutas del fichero de entrada con los datos 
ruta_fichero = Path(".spyder-py3/Data", "Data_Social_Network_Ads.csv")
print(ruta_fichero)
home = Path.home()
ruta_completa = Path(home, ruta_fichero)

# fichero de entrada
data = pd.read_csv(ruta_completa)


print("Observaciones y variables: ", data.shape) # (400, 5)
print("Columnas y tipo de dato", data.columns) 
#(['User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased'], dtype='object')
print(data[['User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased']].describe())

# nos quedamos con las vbles Age,EstimatedSalary, ya que las otras características
# son irrelevantes y no aportan en la elección de compra de la persona
feature_cols = ['Age','EstimatedSalary']
X = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# ARBOL_1. Se crea el árbol de decisión sin parámetros. Es un árbol muy grande, 
# de 14 niveles, y seguramente esté overfitting. Vamos a optimizarlo, creando 
# un nuevo árbol con menos niveles
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

precision = metrics.accuracy_score(y_test, y_pred)
print("La precisión del modelo es:", precision) # 0.85

plt.figure(figsize=(10,8))
tree.plot_tree(dtree, feature_names=feature_cols,filled=True)

plt.show()

# ARBOL_2.
# nuevo arbol pero configurado con parámetros, max profundidad 3
dtree2 = DecisionTreeClassifier(criterion='entropy',
                                max_depth = 3, random_state=42)
    
dtree2.fit(X_train, y_train)

y_pred = dtree2.predict(X_test)

precision = metrics.accuracy_score(y_test, y_pred)
print("La precisión del modelo es:", precision) # 0.92

plt.figure(figsize=(10,8))
tree.plot_tree(dtree2, feature_names=feature_cols,filled=True)

plt.show()

# matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Output:
# array([[57,  6],
#        [ 2, 35]])


# gráfico para comparar las vbles
colores = ['orange', 'blue'] 
labl = ['0-no compra', '1-sí compra']

f1 = data['Age']
f2 = data['EstimatedSalary']

asignar=[]
asignar2=[]
for index, row in data.iterrows(): 
    asignar.append(colores[row['Purchased']])
    asignar2.append(labl[row['Purchased']])

plt.subplot(3,2,1)
plt.title("Gráfico de dispersión - Age/EstimatedSalary")
plt.scatter(f1, f2, c=asignar, label=asignar2, s=20)
plt.axis([16, 65, 14000, 160000]) # valores para 
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.show()



