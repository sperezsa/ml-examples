# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:00:00 2024

@author: Usuario

Se pretende emplear un modelo no supervisado como KMeans para identificar 
patrones de agrupación en los datos 

Se parte de un conjunto de datos de préstamos, nos quedamos con un 
par de vbles y se estandarizan los datos.

Basándose en el valor de inercia, calculamos el óptimo número de 
clusters por medio de la regla del codo.

Una vez definido k, ajustamos el modelo y calculamos la predicción
En el gráfico de dispersión podemos ver por colores cómo se agrupan 
los datos de los clusters.

"""

# KMEANS 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# standardizing the data
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Data/clustering.csv')
print(data.head())

# selection 2 vars
X = data[["LoanAmount","ApplicantIncome"]]

#innitial number of clusters
K=3

#Select random observation as centroids
Centroids = (X.sample(n=K))
#Visualize data points
plt.scatter(X["ApplicantIncome"],X["LoanAmount"],s=50, c='black')
plt.scatter(Centroids["ApplicantIncome"],Centroids["LoanAmount"],c='red')
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (In Thousands)')
plt.show()

#Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

"""
# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,15):
    kmeans = KMeans(n_clusters = cluster, init='k-means++', n_init=10)
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,15), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
"""

# k-means using 5 clusters and k-means++ initialization
k=5
kmeans = KMeans(k, init='k-means++', n_init=10)
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
frame['cluster'].value_counts()
print("cluster and value counts:",frame['cluster'].value_counts())

arr = frame.to_numpy()

colors =['blue', 'red', 'brown', 'green', 'orange']

for i in range(k):
    plt.scatter(arr[pred == i, 1], arr[pred == i, 0], s=50, c=colors[i], label=f'Cluster {i+1}')
    plt.xlabel('AnnualIncome')
    plt.ylabel('Loan Amount (In Thousands)') 
# Add centroids (optional)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 1], centroids[:, 0], s=250, marker='*', c='yellow', label='Centroids')
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()


