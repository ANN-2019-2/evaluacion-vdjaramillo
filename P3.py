import re 
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as confusion
from sklearn.metrics import accuracy_score as precision
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold as KF
from tensorflow import keras
import tensorflow as tf

# 1 para spam 0 para ham

datos = pd.DataFrame(columns=['etiqueta','contenido'])
"""
def limpieza(txt):
    return re.sub("\s{1,}"," ",re.sub('[0-9]*', '', re.sub('https?:\/\/.*[\n]*','', txt))).replace('&gt;','').replace('&lt;','').translate(str.maketrans('', '', string.punctuation))

with open('datos/SMSSpamCollection.txt', encoding='utf-8') as abierto:
    lista = str(abierto.read()).split("\n")
    for row in lista:
        patron = "ham\s"
        remplazo = "1,,. "
        row = re.sub(patron, remplazo, row)
        patron = "spam\s"
        remplazo = "0,,. "
        row = re.sub(patron, remplazo, row)
        row = row.split(",,.")
        if(len(row)>1):
            datos = datos.append({
                'etiqueta': row[0],
                'contenido': limpieza(row[1]) 
                }, ignore_index=True)   
        

datos.to_csv('datos.csv')
"""
datos = pd.read_csv('datos.csv')

X = datos['contenido']
Y = datos['etiqueta'].astype('int')

stem = nltk.stem.SnowballStemmer('english')

## palabras propias del lenguaje sin aporte al análiss
propias = stopwords.words("english")
def aplicarStem(txt):   
   
    words = [stem.stem(word) for word in txt.split() if word.lower() not in propias]
    return " ".join(words)

ss = []
for i in X:
    ss.append(aplicarStem(i))

# Bolsa de palabras

vectorizador = CountVectorizer(ss)
sX = vectorizador.fit_transform(ss)

sX = sX.toarray()
X_train, X_test, Y_train, Y_test = tts(sX, Y , stratify = Y, test_size = 0.2, random_state=101)


regresion = LogisticRegression()
regresion.fit(X_train, Y_train)
prediccion = regresion.predict(X_test)

print(
      confusion(Y_test, prediccion),
      precision(Y_test, prediccion)
)

###red neuronal keras 
# Se elige la catidad de neuronas
nresults = [0,0]
for i in range(1,21):
    ## Crea un modelo vacio    
    model = keras.Sequential()
    ## Adiciona las capas
    model.add(keras.layers.Dense(units = i, activation=tf.nn.relu, input_dim = X_train.shape[1]))
    model.add(keras.layers.Dense(units = 1, activation=tf.nn.sigmoid))
    
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc']
                  )
    ## Estructura del modelo creado
    model.fit(X_train,Y_train,epochs=50,batch_size=30)
    r = model.evaluate(X_test, Y_test)
    if(r[1]>nresults[1]):
        nresults = [i,r[1]]
    model = 0
    
## Se elige la cantidad óptima de capas
    
cresults = [0,0]
for i in range(1,10):
    ## Crea un modelo vacio    
    model = keras.Sequential()
    ## Adiciona las capas
    for j in range(0,i):
        model.add(keras.layers.Dense(units =nresults[0], activation=tf.nn.relu, input_dim = X_train.shape[1]))
    model.add(keras.layers.Dense(units = 1, activation=tf.nn.sigmoid))
    
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc']
                  )
    ## Estructura del modelo creado
    model.fit(X_train,Y_train,epochs=50,batch_size=30)
    r = model.evaluate(X_test, Y_test)
    if(r[1]>cresults[1]):
        cresults = [i,r[1]]
    model = 0


print("Neuronas ---- " ,nresults)
print("Capas ---- " ,nresults)

## Crea el modelo definitivo    
model = keras.Sequential()
## Adiciona las capas
for j in range(0,cresults[0]):
    model.add(keras.layers.Dense(units =nresults[0], activation=tf.nn.relu, input_dim = X_train.shape[1]))
model.add(keras.layers.Dense(units = 1, activation=tf.nn.sigmoid))
    
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc','mse']
              )
## Estructura del modelo creado
model.fit(X_train,Y_train,epochs=100,batch_size=30)
r = model.evaluate(X_test, Y_test)


print("Desempeño del modelo definitivo", r)




##falta la validación cruzada. 





def validacion(indice_entren, indice_test, X, Y):
    
    X_train = X[indice_entren]
    X_test = X[indice_test]
    Y_train = Y.iloc[indice_entren]
    Y_test = Y.iloc[indice_test]
        
    model = keras.Sequential()
    ## Adiciona las capas
    for j in range(0,cresults[0]):
        model.add(keras.layers.Dense(units =nresults[0], activation=tf.nn.relu, input_dim = X_train.shape[1]))
    model.add(keras.layers.Dense(units = 1, activation=tf.nn.sigmoid))
    
    model.summary()
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc','mse']
              )
## Estructura del modelo creado
    model.fit(X_train,Y_train,epochs=100,batch_size=30)
    r = model.evaluate(X_test, Y_test)

            
    return r[2]
    
kf = KF(n_splits=4)

crossval = []
for indice_entren, indice_test in kf.split(X_train):
    crossval.append(validacion(indice_entren,indice_test, X_train, Y_train))

print("")
print("MSE Validación cruzada: ", sum(crossval)/len(crossval))    
