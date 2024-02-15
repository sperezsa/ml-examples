# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:28:55 2023

@author: Usuario

Se realiza un PCA Principal Component Analysis para revisar cuáles de las vbles 
aportan más valor a los datos.

Se parte del fichero de marketing que contiene las ventas relacionadas con la 
inversión de publicidad realizada en diferentes medios de comunicación TV, 
Radio, Newspaper, y Web.

Se realiza una normalización de datos con StandardScaler
 
Se realiza 2 gráficas:
    - Acumulado de var explicada en función de las componentes. 
    Vemos que se tienen que tener en cuenta todas las vbles.
    - Teniendo en cuenta las 2 primeras componentes, ver cómo se distribuyen 
    las ventas (por debajo y por encima de la media).
    A nivel general, vemos que a inversiones bajas tanto de TV como Radio, 
    obtenemos sales por debajo de la media, al contrario ocurre con inversiones
    altas. Remarcar que para inversiones altas en Radio y bajas en TV se observan 
    sales por encima de la media, por lo que parece que sale más rentable las 
    inversiones altas en Radio que en TV. 

CONCLUSIÓN: 
    Se concluye que partiendo de 4 vbles, el 85% de la var explicada se llega 
    teniendo en cuenta las 4 vbles, por lo que todas las vbles aportan. 
    Puede ser debido a que el fichero de datos original ya ha sido trabajado 
    previamente y ya se habían descartado otras vbles.
    
    

"""

#importamos librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pathlib import Path #para el control de rutas en el so

# rutas del fichero de entrada con los datos 
ruta_fichero = Path(".spyder-py3/Data", "Data_Advertising.csv")
print(ruta_fichero)
home = Path.home()
ruta_completa = Path(home, ruta_fichero)

# fichero de entrada
# Datos de importes de ventas por inversiones realizadas en marketing sobre  
# diferentes medios de comunicación. N=200 valores 
dataframe = pd.read_csv(ruta_completa)
print(dataframe.tail(10))

#normalizamos los datos
scaler=StandardScaler()
df = dataframe.drop(['Sales'], axis=1) # quito la variable dependiente "Y"
scaler.fit(df) # calculo la media para poder hacer la transformacion
X_scaled=scaler.transform(df)# Ahora sí, escalo los datos y los normalizo


#Instanciamos objeto PCA y aplicamos
pca=PCA(n_components=4) # Asigno 4 pq ya he comprobado que explica > 85% de la var
pca.fit(X_scaled) # obtener los componentes principales
X_pca=pca.transform(X_scaled) # convertimos nuestros datos con las nuevas dimensiones de PCA
 
print("shape of X_pca", X_pca.shape)
expl = pca.explained_variance_ratio_
print(expl)
print('suma:',sum(expl[0:4]))
#Vemos que con todos los componentes tenemos algo mas del 85% de varianza explicada


#graficamos el acumulado de varianza explicada en las nuevas dimensiones
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
 

#graficamos en 2 Dimensiones, tomando los 2 primeros componentes principales
Xax=X_pca[:,0] # inversión en TV
Yax=X_pca[:,1] # inversión en Radio
# Me creo una vble categórica en función de la media de Sales para ver cómo se
# distribuyen los datos
dataframe['categorical_median'] = np.where(dataframe['Sales'] >= 14.022500, 1, 0)

labels=dataframe['categorical_median'].values
cdict={0:'red',1:'green'}
labl={0:'Por_debajo_media',1:'Por_encima_media'}
marker={0:'*',1:'o'}
alpha={0:.3, 1:.5} # indica la opacidad, cuan trasparente se pintan los datos
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
    ix=np.where(labels==l)
    ax.scatter(Xax[ix],Yax[ix],c=cdict[l],label=labl[l],s=40,marker=marker[l],alpha=alpha[l])
 
plt.xlabel("First Principal Component (TV)",fontsize=14)
plt.ylabel("Second Principal Component (Radio)",fontsize=14)
plt.legend(loc="lower left")
plt.show()
