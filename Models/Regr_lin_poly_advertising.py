# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:20:07 2023

@author: Usuario

Se plantean 2 modelos:
    - modelo de regr lineal múltiple
    - modelo de regr lineal polinómica (grado 2)
Para ver la relación existente entre las ventas con respecto a la inversión en 
marketing sobre diferentes medios de comunicación, TV, Radio, Newspaper, y Web.

Se realiza un gráfico temprano para ver la relación de cada una de las vbles 
independientes vs vble dependiente.

CONCLUSIÓN: 
    Aplicar modelo de reg lineal múltiple.
    Revisando los RMSE de ambos modelos, el modelo de regr polinomial (0,50) 
    ajusta casi 3 veces mejor que el modelo de regr lineal (1,51), pero parece 
    que se produce overfitting viendo el gráfico de regr polinomial. 
    Tb se realiza una prueba para ampliar a grado 3, r2 pasa de 0,98 a 0,99, por 
    lo que se entiende que overfitting es aun mayor.

"""

import pandas as pd
import numpy as np
import seaborn as sns  # Gráficos
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures # Polinomial 

from pathlib import Path #para el control de rutas en el so


# rutas del fichero de entrada con los datos 
ruta_fichero = Path(".spyder-py3/Data", "Data_Advertising.csv")
print(ruta_fichero)
home = Path.home()
ruta_completa = Path(home, ruta_fichero)

# fichero de entrada
# Datos de importes de ventas por inversiones realizadas en marketing sobre  
# diferentes medios de comunicación. N=200 valores 
datos = pd.read_csv(ruta_completa)



print("Observaciones y variables: ", datos.shape) # (200, 7)
print("Columnas y tipo de dato", datos.columns) 
#Index(['Unnamed: 0', 'X', 'TV', 'Radio', 'Newspaper', 'Web', 'Sales'], dtype='object')
print(datos[['TV','Radio', 'Newspaper', 'Web', 'Sales', ]].describe())

# Se grafican los datos de cada una de las vbls indep vs vble dependiente
# Existe relación lineal (+) para TV, Radio y Newspaper pero para esta última
# con poco impacto en las ventas por la alta dispersión de los datos.
# Para Web no existe relación lineal, los datos están muy dispersos
sns.pairplot(datos, x_vars=['TV','Radio','Newspaper', 'Web'], y_vars='Sales', 
             height=7, aspect=0.8,kind = 'reg')

#plt.savefig("pairplot.jpg")
plt.show()

X_independientes = datos.iloc[:,2:6]
print("Variables independientes \n", X_independientes)

Y_dependiente = datos.iloc[:, 6:7]
#print("Variable dependiente \n", Y_dependiente)

X_entrena,X_valida,Y_entrena,Y_valida = train_test_split(X_independientes, 
                                                         Y_dependiente,
                                                         train_size=.70,
                                                         random_state=1280)

#print("Estructura de datos de entrenamiento... ", X_entrena.shape) # (140, 4)

# Se define el modelo de reg lineal y se ajusta con los datos de entrenamiento
modelo_rm = LinearRegression()
modelo_rm.fit(X_entrena,Y_entrena)

b0 = modelo_rm.intercept_
print ("Intercepción o b0", b0)

b1 = modelo_rm.coef_[0, 0:1]
b2 = modelo_rm.coef_[0, 1:2]
b3 = modelo_rm.coef_[0, 2:3]
b4 = modelo_rm.coef_[0, 3:4]
print ("Coeficientes: b1, b2, b3 y b4", b1, b2, b3, b4)

# R2. El coeficiente de determinación de la predicción vale 0'887, por encima 
# del 80% por lo que podemos considerarlo como bueno. Teniendo en cuenta que 
# con solo las 2 primeras vbles (TV y Radio) el R2 vale 0.8834, son las vbles 
# que aportan por lo que el gráfico de 3D se realizará sólo teniendo en cuenta 
# estas vbles
print("En modelo de Reg Lineal r2 vale:", modelo_rm.score(X_entrena, Y_entrena))

predicciones = modelo_rm.predict(X_valida)

# Se imprime una muestra del listado de los datos de validación 
# incluyendo la predicción
comparaciones = pd.DataFrame(X_valida)
comparaciones = comparaciones.assign(Sales_Real = Y_valida)
comparaciones = comparaciones.assign(Predicho = predicciones.flatten().tolist())
print(comparaciones.head(15))

MSE_lin = metrics.mean_squared_error(Y_valida, predicciones)
RMSE_lin = np.sqrt(MSE_lin)

print('Mean Squared Error: MSE', MSE_lin)
print('Root Mean Squared Error RMSE:', RMSE_lin)
#  MSE: 2.3025937240512038 / para TV y Radio baja un poco más: 2.2428
# RMSE: 1.5174299733599583 / para TV y Radio baja un poco más: 1.4976
# MSE y RMSE ambos son bajos por lo que es bueno el modelo

# Gráfico para mostrar los resultados en 3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

fig = plt.figure()
ax = Axes3D(fig)

# Con solo las 2 vbles dependientes (TV y Radio) que son las vbles que aportan
# Creamos una malla, sobre la cual graficaremos el plano
# en este primer caso de linspace, genera una array de 10 números entre 0, y 300.
xx, yy = np.meshgrid(np.linspace(0, 300, num=10), np.linspace(0, 50, num=10))

# calculamos los valores del plano para los puntos x e y
nuevoX = (b1 * xx)
nuevoY = (b2 * yy) 

# calculamos los correspondientes valores para z. Debemos sumar el punto de intercepción
z = (nuevoX + nuevoY + modelo_rm.intercept_)

# Graficamos el plano
ax.plot_surface(xx, yy, z, alpha=0.2, cmap='hot')
 
# Graficamos en azul los puntos en 3D del entreno
ax.scatter(X_entrena['TV'], X_entrena['Radio'], Y_entrena, c='blue',s=30)
 
# Graficamos en rojo los puntos en 3D del test 
ax.scatter(X_valida['TV'], X_valida['Radio'], predicciones, c='red',s=40)
 
# con esto situamos la "camara" con la que visualizamos
ax.view_init(elev=10., azim=70)

ax.set_xlabel('TV')
ax.set_ylabel('Radio')
ax.set_zlabel('Ventas')
ax.set_title('Regresión Lineal con Múltiples Variables')
plt.show()

print("REGR LINEAL POLINÓMICA (2º grado) SOLO PARA TV Y RADIO")
X, y = datos[["TV", "Radio"]], datos["Sales"]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=1280)

# Se define el modelo de reg lineal (polinómica) 
# y se ajusta con los datos de entrenamiento
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)

# R2. El coeficiente de determinación de la predicción vale 0.98265
# por encima del 80% por lo que podemos considerarlo como bueno
print("En modelo de Reg Lineal POLINÓMICA r2 vale:", 
      poly_reg_model.score(X_train, y_train))

poly_reg_y_predicted = poly_reg_model.predict(X_test)

MSE_poly = metrics.mean_squared_error(y_test, poly_reg_y_predicted)
RMSE_poly = np.sqrt(MSE_poly)
print('Mean Squared Error: MSE', MSE_poly)
print('Root Mean Squared Error RMSE:', RMSE_poly)
# MSE: 0.2254
#RMSE: 0.4747


#Gráficos de TV y Radio con respecto a Sales
font1 = {'family': 'serif', 'color':'blue', 'size': 20}
font2 = {'family': 'monospace', 'color':'red', 'size': 10}

plt.subplot(3,2,1)

plt.plot(X_test, poly_reg_y_predicted, marker='*', linestyle='dashed', c='green', linewidth='5.5')
plt.plot(X["TV"], y, marker='o', c='red', linewidth='2.5')
plt.xlabel("Eje X, TV", fontdict=font1)
plt.ylabel("Eje Y, Sales", fontdict=font1)
plt.title("Regresión polinomial (gr=2) TV", size=16)
plt.grid(axis='x', color = 'grey', linestyle = 'dotted', linewidth = 2.5)

plt.subplot(3,2,2)

plt.plot(X_test, poly_reg_y_predicted, marker='*', linestyle='dashed', c='green', linewidth='5.5')
plt.plot(X["Radio"], y, marker='o', c='red', linewidth='2.5')
plt.xlabel("Eje X, Radio", fontdict=font2)
plt.ylabel("Eje Y, Sales", fontdict=font2)
plt.title("Regresión polinomial (gr=2) Radio", size=16)
plt.grid(axis='x', color = 'grey', linestyle = 'dotted', linewidth = 2.5)

plt.show()

