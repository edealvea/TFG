#%%
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras as kr 
from tensorflow_addons.metrics import RSquare as r2 
import tensorflow as tf 
import numpy as np
import pandas as pd
from sklearn import linear_model
import seaborn as sb
import matplotlib.pyplot as plt
import graphviz
import pydot
#%%
######Preparación de los datos#########
data = pd.read_table(r"C:\Users\Administrador\Documents\cosas_universidad\TFG\Viento\Datos_viento_Madrid.txt" , header=0 , sep = "\t")
data = data.dropna() #Quitamos los Nan

scaler=Normalizer() #Vamos a noramlizar los datos de los predictores 
X = data.drop(['fecha','obsU','obsV','obsM'], axis=1)
Xscaler = scaler.fit_transform(X)
Xnorm = pd.DataFrame(Xscaler,  columns=['predU10m','predV10m','predM10m','predU150m','predV150m','predM150m','predMes','predHora','predMesU','predMesV','predHoraU','predHoraV','predDia','predDiaU','predDiaV'])

Y=data.drop(['fecha','predU10m','predV10m','predM10m','predU150m','predV150m','predM150m','predMes','predHora','predMesU','predMesV','predHoraU','predHoraV','predDia','predDiaU','predDiaV'], axis=1)


xtrain_valid, xtest, ytrain_valid, ytest = train_test_split(Xnorm, Y, test_size= .15, random_state = 12) #Hacemos una separación entre el set de validación y de entrenamiento
xtrain, xvalid, ytrain, yvalid =  train_test_split(xtrain_valid, ytrain_valid, test_size= .2, random_state = 142)#Ahora en el set de entrenamiento separamos por set de trainy de validación
#%%

xtrain_valid2, xtest2, ytrain_valid2, ytest2 = train_test_split(X, Y, test_size= .15, random_state = 12) #Hacemos una separación entre el set de validación y de entrenamiento
xtrain2, xvalid2, ytrain2, yvalid2 =  train_test_split(xtrain_valid2, ytrain_valid2, test_size= .2, random_state = 142)
#%%Funcion para visualizar resultados
def graficas_result(modelo, x, y, N= 100):
    a = modelo.predict(x.iloc[0:N])#Veamos con 100 predicciones, cambiar por el modelo que se quiera ver
    plt.scatter(a[:,0],y.iloc[0:100,0])
    x0=[i for i in range(int(np.floor(min(a[:,0]))),int(np.ceil(max(a[:,0]))))]
    plt.plot(x0,x0,"r-", label="y=x")
    regr=linear_model.LinearRegression()
    regr.fit(a[:,0].reshape(-1,1), y.iloc[0:100,0])
    plt.plot(a[:,0],regr.predict(a[:,0].reshape(-1,1)), "g-",label="recta regresion")
    plt.title("Scatter plot 1a coord")
    plt.legend()
    plt.show()

    plt.scatter(a[:,1],y.iloc[0:100,1])
    x0=[i for i in range(int(np.floor(min(a[:,1]))),int(np.ceil(max(a[:,1]))))]
    plt.plot(x0,x0,"r-", label="y=x")
    regr=linear_model.LinearRegression()
    regr.fit(a[:,1].reshape(-1,1), y.iloc[0:100,1])
    plt.plot(a[:,1],regr.predict(a[:,1].reshape(-1,1)),"g-",label="recta regresion")
    plt.title("Scatter plot 2a coord")
    plt.legend()
    plt.show()

    plt.scatter(a[:,2],y.iloc[0:100,2])
    x0=[i for i in range(int(np.floor(min(a[:,2])))-7,int(np.ceil(max(a[:,2])))+7)]
    plt.plot(x0,x0,"r-", label="y=x")
    regr=linear_model.LinearRegression()
    regr.fit(a[:,2].reshape(-1,1), y.iloc[0:100,2])
    plt.plot(a[:,2],regr.predict(a[:,2].reshape(-1,1)),"g-",label="recta regresion")
    plt.title("Scatter plot 3a coord")
    plt.legend()
    plt.show()
#%%

##################Perceptrón######################

input_percep = kr.Input(shape=(15,))
x_perc = kr.layers.Dense(units = 15, activation= "relu")(input_percep)
out_percep = kr.layers.Dense(units = 3, activation = "linear")(x_perc)

perceptron = kr.models.Model(inputs=[input_percep], outputs=[out_percep])
perceptron.compile(loss="mean_squared_error", optimizer=kr.optimizers.Adam(learning_rate=.001), metrics = r2())
callback = kr.callbacks.EarlyStopping(monitor='val_loss', min_delta=.1, mode="min", patience=3, restore_best_weights=True)

history_perc = perceptron.fit(xtrain, ytrain, batch_size = 64, epochs = 200, callbacks= callback, validation_data =(xvalid, yvalid), verbose = 1)
dfhist = pd.DataFrame(history_perc.history).drop(['r_square', 'val_r_square'], axis = 1)
dfhist.plot()
plt.grid(True)
plt.gca() # set the vertical range to [0-1]
plt.title("Tasa de aprendizaje perceptron")
plt.show()

#Scoring del modelo
scores_train = perceptron.evaluate(xtrain, ytrain, verbose=0)
scores_valid = perceptron.evaluate(xvalid, yvalid, verbose=0)
scores_test = perceptron.evaluate(xtest, ytest, verbose=0)
print(f"Evaluación del modelo para MSE y R2:\nConjunto entrenamiento: {scores_train}\nConjunto validacion: {scores_valid}\nConjunto test: {scores_test}")

graficas_result(perceptron, xtest, ytest)

#%%
#Multicapa
input_MLv1 = kr.Input(shape=(15,))
x_MLv11 = kr.layers.Dense(units = 2750, activation= "relu")(input_MLv1)
x_MLv12 = kr.layers.Dense(units = 2000, activation= "relu")(x_MLv11)
x_MLv13 = kr.layers.Dense(units = 1750, activation= "relu")(x_MLv12)
x_MLv14 = kr.layers.Dense(units = 1500, activation= "relu")(x_MLv13)
x_MLv15 = kr.layers.Dense(units = 1250, activation= "relu")(x_MLv14)
x_MLv16 = kr.layers.Dense(units = 1000, activation= "relu")(x_MLv15)
x_MLv17 = kr.layers.Dense(units = 750, activation= "relu")(x_MLv16)
out_MLv1 = kr.layers.Dense(units = 3, activation = "linear")(x_MLv17)

MLv1 = kr.models.Model(inputs=[input_MLv1], outputs=[out_MLv1])
MLv1.compile(loss="mean_squared_error", optimizer=kr.optimizers.Adam(learning_rate=.005), metrics = r2())
callback = kr.callbacks.EarlyStopping(monitor='val_loss', min_delta=.001, mode="min", patience = 5, restore_best_weights=True)

history_MLv1 = MLv1.fit(xtrain2, ytrain2, batch_size = 32, epochs = 200, validation_data =(xvalid2, yvalid2), callbacks = [callback] , verbose = 1)
dfhist = pd.DataFrame(history_MLv1.history).drop(['r_square', 'val_r_square'], axis = 1)
dfhist.plot()
plt.grid(True)
plt.ylim([0,80])
plt.gca() # set the vertical range to [0-1]
plt.title("Tasa de aprendizaje Red Multicapa v1")
plt.show()

#Scoring del modelo
scores_train = MLv1.evaluate(xtrain2, ytrain2, verbose=0)
scores_valid = MLv1.evaluate(xvalid2, yvalid2, verbose=0)
scores_test = MLv1.evaluate(xtest2, ytest2, verbose=0)
print(f"Evaluación del modelo para MSE y R2:\nConjunto entrenamiento: {scores_train}\nConjunto validacion: {scores_valid}\nConjunto test: {scores_test}")

graficas_result(MLv1, xtest2, ytest2)

#%%
#3 modelos 
#Multicapa
input_MLv2 = kr.Input(shape=(15,))
# x_MLv21_1coord = kr.layers.Dense(units = 1200, activation= "relu")(input_MLv2)
# x_MLv22_1coord = kr.layers.Dense(units = 1100, activation= "relu")(x_MLv21_1coord)
# x_MLv23_1coord = kr.layers.Dense(units = 900, activation= "relu")(x_MLv22_1coord)
# x_MLv24_1coord = kr.layers.Dense(units = 600, activation= "relu")(x_MLv23_1coord)
# x_MLv25_1coord = kr.layers.Dense(units = 500, activation= "relu")(x_MLv24_1coord)
# out_MLv2_1cord = kr.layers.Dense(units = 1, activation = "linear")(x_MLv24_1coord)

x_MLv21_2coord = kr.layers.Dense(units = 1200, activation= "relu")(input_MLv2)
x_MLv22_2coord = kr.layers.Dense(units = 1000, activation= "relu")(x_MLv21_2coord)
x_MLv23_2coord = kr.layers.Dense(units = 900, activation= "relu")(x_MLv22_2coord)
out_MLv2_2cord = kr.layers.Dense(units = 1, activation = "linear")(x_MLv23_2coord)

# x_MLv21_3coord = kr.layers.Dense(units = 800, activation= "relu")(input_MLv2)
# x_MLv22_3coord = kr.layers.Dense(units = 750, activation= "relu")(x_MLv21_3coord)
# x_MLv23_3coord = kr.layers.Dense(units = 700, activation= "relu")(x_MLv22_3coord)
# x_MLv24_3coord = kr.layers.Dense(units = 650, activation= "relu")(x_MLv23_3coord)
# x_MLv25_3coord = kr.layers.Dense(units = 550, activation= "relu")(x_MLv24_3coord)
# x_MLv26_3coord = kr.layers.Dense(units = 400, activation= "relu")(x_MLv25_3coord)
# out_MLv2_3cord = kr.layers.Dense(units = 1, activation = "linear")(x_MLv26_3coord)

#out_MLv2 = kr.layers.concatenate([out_MLv2_1cord, out_MLv2_2cord, out_MLv2_3cord])

MLv2 = kr.models.Model(inputs=[input_MLv2], outputs=[out_MLv2_2cord])
MLv2.compile(loss="mean_squared_error", optimizer=kr.optimizers.Adam(learning_rate=.005), metrics = r2())
callback = kr.callbacks.EarlyStopping(monitor='val_loss', min_delta=.001, mode="min", patience = 5, restore_best_weights=True)

history_MLv2 = MLv2.fit(xtrain2, ytrain2["obsU"], batch_size = 32, epochs = 200, validation_data =(xvalid, yvalid["obsM"]), callbacks = [callback] , verbose = 1)
dfhist = pd.DataFrame(history_MLv2.history).drop(['r_square', 'val_r_square'], axis = 1)
dfhist.plot()
plt.grid(True)
plt.gca() # set the vertical range to [0-1]
plt.title("Tasa de aprendizaje Red Multicapa v1")
plt.show()

#Scoring del modelo
scores_train = MLv2.evaluate(xtrain2, ytrain2["obsU"], verbose=0)
scores_valid = MLv2.evaluate(xvalid2, yvalid2["obsU"], verbose=0)
scores_test = MLv2.evaluate(xtest2, ytest2["obsU"], verbose=0)
print(f"Evaluación del modelo para MSE y R2:\nConjunto entrenamiento: {scores_train}\nConjunto validacion: {scores_valid}\nConjunto test: {scores_test}")

graficas_result(MLv2, xtest2, ytest2)
#%%
def graficas_result(modelo, x, y, N= 100):
    a = modelo.predict(x.iloc[0:N])#Veamos con 100 predicciones, cambiar por el modelo que se quiera ver

    plt.scatter(a,y.iloc[0:100,0])
    x0=[i for i in range(int(np.floor(min(a)))-7,int(np.ceil(max(a)))+7)]
    plt.plot(x0,x0,"r-", label="y=x")
    regr=linear_model.LinearRegression()
    regr.fit(a.reshape(-1,1), y.iloc[0:100,0])
    plt.plot(a,regr.predict(a.reshape(-1,1)),"g-",label="recta regresion")
    plt.title("Scatter plot 3a coord")
    plt.legend()
    plt.show()
# %%
