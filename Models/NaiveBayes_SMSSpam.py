# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:55:14 2023

@author: Usuario

Se propone aplicar el modelo de Naive Bayes para poder diferenciar los SMS como 
SPAM o no SPAM.

Se parte de un fichero que se encuentra en internet con SMS de UK. Este fichero 
contiene dos campos, 'type', 'message'. El primero indica la categoria (spam / 
ham) y el segundo es el cuerpo del mensaje de texto.
Contiene un total de 4,827 SMS OK (86.6%) y 747 SPAM (13.4%).

Se aplican diferentes tratamientos al mensaje: se tokeniza, se eliminan las 
palabras vacías (stopwords), se aplican stemming, que es la eliminación de las 
terminaciones de las palabras para quedarnos únicamente con la raíz y finalmente 
se vuelve a pasar a string para que el siguiente paso pueda tratarlo.

El paso siguiente es vectorizar, que es el paso que crea la matriz TF. Con ella 
se entrena y se trasforman los datos que se le pasan al modelo NB para que 
entrene con estos datos y por ultimo se hace la predicción y se muestran los 
scores y la matriz de confusión.
 
"""

import requests #para la petición https de descarga del fichero de datos
import zipfile #para descomprimir el fichero
import pandas as pd
from sklearn.feature_extraction import text # Sklearn
import nltk # NLTK
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
data_file = 'SMSSpamCollection'

# Make request
resp = requests.get(url)

# Get filename --> smsspamcollection.zip
filename = url.split('/')[-1]

# Download zipfile
with open(filename, 'wb') as f:
  f.write(resp.content)

# Extract Zip
with zipfile.ZipFile(filename, 'r') as zip:
  zip.extractall('')

# Read Dataset
data = pd.read_table(data_file, 
                     header = 0,
                     names = ['type', 'message']
                     )

# Show dataset
print(data.head())

text.ENGLISH_STOP_WORDS

# Install everything necessary
# stopwords listado de palabras vacías, como yo, mi, yo mismo, nuestro, usted...
# existe en diferentes idiomas
nltk.download('stopwords')
# PunktTokenizer: tokeniza en los signos de puntuación, pero los mantiene junto 
# a la palabra.
nltk.download('punkt')


stop = stopwords.words('english')

# Tokenize
data['tokens'] = data.apply(lambda x: nltk.word_tokenize(x['message']), axis = 1)

# Remove stop words
data['tokens'] = data['tokens'].apply(lambda x: [item for item in x if item not in stop])

# Apply Porter stemming
# el stemming consiste en la eliminación de las terminaciones de las palabras 
# para quedarnos únicamente con la raíz.
stemmer = PorterStemmer()
data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])


# Unify the strings once again
data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))


# Make split
x_train, x_test, y_train, y_test = train_test_split(
    data['tokens'], 
    data['type'], 
    test_size= 0.2
    )

# Create vectorizer
# crear la matriz TF. Si indicamos parámetro binary = True, creará una matriz 
# de apariciones. En el caso de SMS es muy posible que arroje los mismos 
# resultados ambas matrices
vectorizer = CountVectorizer(
    strip_accents = 'ascii', 
    lowercase = True
    )

# Fit vectorizer & transform it
vectorizer_fit = vectorizer.fit(x_train)
x_train_transformed = vectorizer_fit.transform(x_train)
x_test_transformed = vectorizer_fit.transform(x_test)

# Build and train the model
naive_bayes = MultinomialNB()
naive_bayes_fit = naive_bayes.fit(x_train_transformed, y_train)


# Make predictions
train_predict = naive_bayes_fit.predict(x_train_transformed)
test_predict = naive_bayes_fit.predict(x_test_transformed)

def get_scores(y_real, predict):
  ba_train = balanced_accuracy_score(y_real, predict)
  cm_train = confusion_matrix(y_real, predict)

  return ba_train, cm_train 

def print_scores(scores):
  return f"Balanced Accuracy: {scores[0]}\nConfussion Matrix:\n {scores[1]}"

train_scores = get_scores(y_train, train_predict)
test_scores = get_scores(y_test, test_predict)


print("## Train Scores")
print(print_scores(train_scores))
print("\n\n## Test Scores")
print(print_scores(test_scores))

# ## Train Scores
# Balanced Accuracy: 0.9826166231747266
# Confussion Matrix:
#  [[3867    8]
#  [  19  562]]


# ## Test Scores
# Balanced Accuracy: 0.9622748105170947
# Confussion Matrix:
#  [[946   3]
#  [ 12 154]]