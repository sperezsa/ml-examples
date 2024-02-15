# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:46:44 2023

@author: Usuario

Se pretende generar un modelo de LinearDiscriminantAnalysis con los datos de iris.

Se aplica el procedimiento RepeatedStratifiedKFold, que nos va a servir para 
estimar el rendimiento del algoritmo utilizado.

Se parten de los datos de iris para generar el modelo.
target: 0-setosa, 1-versicolor, 2-virginica

Se pinta un gráfico para observar cómo se distribuyen los datos. Se observa que 
hay un grupo de flores, 0-setosa, que aparece en color rojo, que se diferencia 
muy bien de las otras dos.

Partiendo de los datos de una nueva observación, se realiza una predicción.

"""
# Importing required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn import datasets
import matplotlib.pyplot as plt

# Load iris dataset
iris=datasets.load_iris()

# Convert dataset into a pandas dataframe
# This dataframe contains new column with the name of the specie
df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                 columns = iris['feature_names'] + ['target'])

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.columns = ['s_length', 's_width', 'p_length', 'p_width', 'target', 'species']
print(df.head())

# Fitting the model LDA
X = df[['s_length', 's_width', 'p_length', 'p_width']] #DF shape (150,4)
y = df['species']

model = LinearDiscriminantAnalysis()
model.fit(X, y)

# Evaluating the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("np.mean(scores)", np.mean(scores)) #0.9800000000000001


# Visualize the results
X = iris.data #shape (150, 4)
y = iris.target # shape y (150,)
model = LinearDiscriminantAnalysis()

# para poderlo graficar en 2-D se realiza una transformación de los datos 
# originales de X pasando de 4 valores a 2 valores
data_plot = model.fit(X, y).transform(X) # ndarray(150,2)
target_names = iris.target_names

plt.figure()
colors = ['red', 'green', 'blue']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(data_plot[y == i, 0], data_plot[y == i, 1], alpha=.8, color=color,
                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

# Prediction for the new observation
new = [5, 2, 1, .4]
#other_obs = [5, 3, 4, 14]
pred = model.predict([new]) 
print("Predicción para un nuevo valor es:", pred) #array([0]) setosa

