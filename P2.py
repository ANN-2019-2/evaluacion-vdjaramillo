from sklearn.neural_network import MLPRegressor
import pandas as pd
import statsmodels.api as sm
import copy
import matplotlib.pyplot as plt
import sklearn.metrics as skm


def encadenar(i,lista):
    nlista = copy.deepcopy(lista)
    nlista.append(i)
    return nlista
    
def forward(keys, skeys, X, mseh, itera):
    itera = itera +1
    d = X.loc[:,'Y']
    ikey=0
    l = len(mseh)-1
    if(len(skeys)==0):  
        for i in keys:
            model = sm.OLS(d, X.loc[:,i])
            results = model.fit()
            results.summary()
            mse = results.mse_total
            if(mse<mseh[0][1]):
                mseh[0] = [i,mse,ikey]
            ikey = ikey +1
    else:   
        for i in keys: 
            k = encadenar(i,skeys)
            model = sm.OLS(d, X.loc[:,k])
            results = model.fit()
            results.summary()
            mse = results.mse_total
            if(mse<mseh[l][1] and mse<mseh[l-1][1]):
                
                mseh[l] = [i,mse,ikey]
            ikey = ikey +1
    
           
    skeys.append(keys.pop(mseh[l][2]))
    if(len(keys)==0):
        dkeys = []
        for j in mseh:
            if(not(j[0]=='')):
                dkeys.append(j[0])
        return dkeys
    
    mseh.append(['',100000,0])
    return forward(keys,skeys,X,mseh, itera)



X = pd.read_csv('./datos/diabetes.csv')
key = []
for i in X.keys():
    key.append(i)
    
ind = 0
key.pop(10)
    
dkeys = forward(key,[], X, [['',100000,0]], 0)
# Regresión Lineal del mejor modelo
print("Vars elegidas: ",dkeys)
dmodel = sm.OLS(X['Y'], X.loc[:,dkeys]) 
dresults = dmodel.fit()
X_train = X.loc[:, dkeys][:380]
X_test = X.loc[:, dkeys][380:420]
X_pred = X.loc[:,dkeys][420:]
d = X['Y']
Y_train = d[:380]
Y_test = d[380:420]
Y_pred = d[420:]
#
for j in range(1,10):
        
    E_train_hist = []
    E_test_hist = []

    for i in range(1,20):
        m = MLPRegressor(
            hidden_layer_sizes = (i,j),  # Una capa oculta con una neurona
            activation = 'logistic',    #  {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
            solver = 'sgd',             #  {‘lbfgs’, ‘sgd’, ‘adam’}
            alpha = 0.0,                #
            learning_rate_init = 0.1,   # Valor de la tasa de aprendizaje
            learning_rate = 'constant', # La tasa no se adapta automáticamente
            verbose = False,            # Reporte del proceso de optimización
            shuffle = True,             #
            tol = 1e-8,                 #
            max_iter = 25000,           # Número máximo de iteraciones
            momentum = 0.0,             #
            nesterovs_momentum = False) 
       
        m.fit(X_train, Y_train)
        E_train_hist.append(skm.mean_squared_error(m.predict(X_train), Y_train))
        E_test_hist.append(skm.mean_squared_error(m.predict(X_test), Y_test))
    
    plt.figure()
    plt.title("Errores ("+str(j)+")")
    plt.plot(range(1,20) ,E_train_hist,  '-', color="red");
    plt.plot( range(1,20) ,E_test_hist, '-', color="black");
    
    
#3 neuronas, 8 capas 
    
m = MLPRegressor(
    hidden_layer_sizes = (3,8),  # Una capa oculta con una neurona
    activation = 'logistic',    #  {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
    solver = 'sgd',             #  {‘lbfgs’, ‘sgd’, ‘adam’}
    alpha = 0.0,                #
    learning_rate_init = 0.1,   # Valor de la tasa de aprendizaje
    learning_rate = 'constant', # La tasa no se adapta automáticamente
    verbose = False,            # Reporte del proceso de optimización
    shuffle = True,             #
    tol = 1e-8,                 #
    max_iter = 25000,           # Número máximo de iteraciones
    momentum = 0.0,             #
    nesterovs_momentum = False) 
m.fit(X_train, Y_train)
