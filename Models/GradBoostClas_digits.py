# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:47:36 2023

@author: Usuario

Se pretende emplear GradientBoostingClassifier para clasificar datos de dígitos
escritos a mano.

Se parte de un conjunto de imágenes de 8x8 píxels de dígitos hechos a mano, por 
lo que tenemos un array de 64 valores y un target, campo adicional que te indica 
el label (el número entre 0-9). N=1797 observaciones 

Se aplica un modelo GradientBoostingClassifier, se entrena y se realiza la 
previsión. 98% de precisión!

"""

# GradientBoostingClassifier
##########################################################################
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
 
# Setting SEED for reproducibility
SEED = 23
 
# Importing the dataset, returns (data, target)
X, y = load_digits(return_X_y=True)

# Another alternative: Importing the dataset, returns tuple (DataFrame, Series)
tp = load_digits(return_X_y=True,as_frame=True)

print(tp[0])
print(tp[1])

#X (1797, 64) 
#y = (1797, )
 
# Splitting dataset
train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size = 0.25,
                                                    random_state = SEED)
 
# Instantiate Gradient Boosting Regressor
gbc = GradientBoostingClassifier(n_estimators=300,
                                  learning_rate=0.05,
                                  random_state=100,
                                  max_features=5 )
# Fit to training set
gbc.fit(train_X, train_y)
 
# Predict on test set
pred_y = gbc.predict(test_X)
 
# accuracy
acc = accuracy_score(test_y, pred_y)
print("Gradient Boosting Classifier accuracy is : {:.2f}".format(acc)) #0.98

