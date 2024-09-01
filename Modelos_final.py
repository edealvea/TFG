#%%
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#%%
######Preparación de los datos#########
data = pd.read_table(r"C:\Users\Administrador\Documents\cosas_universidad\TFG\Viento\Datos_viento_Madrid.txt" , header=0 , sep = "\t")
data = data.dropna() #Quitamos los Nan

scaler=Normalizer() #Vamos a noramlizar los datos de los predictores 
X = data.drop(['fecha','obsU','obsV','obsM'], axis=1)
Xscaler = scaler.fit_transform(X)
Xnorm = pd.DataFrame(Xscaler,  columns=['predU10m','predV10m','predM10m','predU150m','predV150m','predM150m','predMes','predHora','predMesU','predMesV','predHoraU','predHoraV','predDia','predDiaU','predDiaV'])

Y=data.drop(['fecha','predU10m','predV10m','predM10m','predU150m','predV150m','predM150m','predMes','predHora','predMesU','predMesV','predHoraU','predHoraV','predDia','predDiaU','predDiaV'], axis=1)

#%%Funcion para visualizar resultados
def graficas_result(modelo, x, y, N= 100):
    a = modelo.predict(x.iloc[0:N], verbose =0)#Veamos con 100 predicciones, cambiar por el modelo que se quiera ver
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
#Vamos a ver primero el modelo del perceptrón
fold_no=1
compare_scores= []
mejor=[0,999999999] #Voy poner un valor arbitrario que va a superar seguro 

kf= KFold(n_splits =5, shuffle= True ,random_state=8)
for train, valid in kf.split(Xnorm, Y):

    inputdataperceptron = keras.layers.Input(shape=(15,)) 
    capaperceptron = keras.layers.Dense(50, activation = "relu")(inputdataperceptron)
    outputdataperceptron = keras.layers.Dense(3, activation = "linear")(capaperceptron)


    modelopred = keras.models.Model(inputs=[inputdataperceptron], outputs=[outputdataperceptron]) 

    modelopred.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=.001))
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.05, mode="min" ,patience=5, restore_best_weights=True)

    history= modelopred.fit(Xnorm[train[0]:(train[-1]+1)], Y[train[0]:(train[-1]+1)], epochs=2000,
    validation_data=(Xnorm[valid[0]:(valid[-1]+1)], Y[valid[0]:(valid[-1]+1)]), callbacks=[callback], verbose=1)

    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca() # set the vertical range to [0-1]
    plt.title("Tasa de aprendizaje")
    plt.show()
    scores = [modelopred.evaluate(Xnorm[train[0]:(train[-1]+1)], Y[train[0]:(train[-1]+1)], verbose=0), modelopred.evaluate(Xnorm[valid[0]:(valid[-1]+1)], Y[valid[0]:(valid[-1]+1)])]
    if scores[1] < mejor[1]:
        modeloperceptron = modelopred
        mejor = scores
        validf = valid 
    print(f'Score for fold {fold_no}: loss of {scores}')
    fold_no+=1

graficas_result(modeloperceptron, Xnorm[validf[0]:(validf[-1]+1)], Y[validf[0]:(validf[-1]+1)])
print(f"Scoring del modelo con respecto al MSE: {mejor}")
modeloperceptron.save(r'C:\Users\Administrador\Documents\cosas_universidad\TFG\modelos\modelo_2a_coord_general.h5')

# %%
#Veamos un modelo multicapa

fold_no=1
compare_scores= []
mejor=[0,999999999] #Voy poner un valor arbitrario que va a superar seguro 

kf= KFold(n_splits =5, shuffle= True ,random_state=8)
for train, valid in kf.split(Xnorm, Y):

    inputdataMLv1 = keras.layers.Input(shape=(15,)) 
    capa1MLv1 = keras.layers.Dense(250, activation = "relu")(inputdataMLv1)
    capa2MLv1 = keras.layers.Dense(200, activation = "relu")(capa1MLv1)
    capa3MLv1 = keras.layers.Dense(150, activation = "relu")(capa2MLv1)
    capa4MLv1 = keras.layers.Dense(100, activation = "relu")(capa3MLv1)
    capa5MLv1 = keras.layers.Dense(50, activation = "relu")(capa4MLv1)
    outputdataMLv1 = keras.layers.Dense(3, activation = "linear")(capa5MLv1)


    modelopred = keras.models.Model(inputs=[inputdataMLv1], outputs=[outputdataMLv1]) 

    modelopred.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=.001))
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.05, mode="min" ,patience=5, restore_best_weights=True)

    history= modelopred.fit(Xnorm[train[0]:(train[-1]+1)], Y[train[0]:(train[-1]+1)], epochs=2000,
    validation_data=(Xnorm[valid[0]:(valid[-1]+1)], Y[valid[0]:(valid[-1]+1)]), callbacks=[callback], verbose=1)

    history= modelopred.fit(Xnorm[train[0]:(train[-1]+1)], Y[train[0]:(train[-1]+1)], epochs=2000,
    validation_data=(Xnorm[valid[0]:(valid[-1]+1)], Y[valid[0]:(valid[-1]+1)]), callbacks=[callback], verbose=1)


    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca() # set the vertical range to [0-1]
    plt.title("Tasa de aprendizaje")
    plt.show()
    scores = [modelopred.evaluate(Xnorm[train[0]:(train[-1]+1)], Y[train[0]:(train[-1]+1)], verbose=0), modelopred.evaluate(Xnorm[valid[0]:(valid[-1]+1)], Y[valid[0]:(valid[-1]+1)])]
    if scores[1] < mejor[1]:
        modeloMLV1 = modelopred
        mejor = scores
        validf = valid
    print(f'Score for fold {fold_no}: loss of {scores}')
    fold_no+=1
graficas_result(modeloMLV1, Xnorm[validf[0]:(validf[-1]+1)], Y[validf[0]:(validf[-1]+1)])
print(f"Scoring del modelo con respecto al MSE: {mejor}")

modeloMLV1.save(r'C:\Users\Administrador\Documents\cosas_universidad\TFG\modelos\modeloMLv1.h5')

# %%
#Veamos un segundo modelo multicapa con más capas y neuronas

fold_no=1
compare_scores= []
mejor=[0,999999999] #Voy poner un valor arbitrario que va a superar seguro 

kf= KFold(n_splits =5, shuffle= True ,random_state=8)
for train, valid in kf.split(Xnorm, Y):

    inputdataMLv2 = keras.layers.Input(shape=(15,)) 
    capa1MLv1 = keras.layers.Dense(1000, activation = "relu")(inputdataMLv2)
    capa2MLv1 = keras.layers.Dense(950, activation = "relu")(capa1MLv1)
    capa3MLv1 = keras.layers.Dense(900, activation = "relu")(capa2MLv1)
    capa4MLv1 = keras.layers.Dense(850, activation = "relu")(capa3MLv1)
    capa5MLv1 = keras.layers.Dense(800, activation = "relu")(capa4MLv1)
    capa6MLv1 = keras.layers.Dense(750, activation = "relu")(capa5MLv1)
    capa7MLv1 = keras.layers.Dense(700, activation = "relu")(capa6MLv1)
    capa8MLv1 = keras.layers.Dense(650, activation = "relu")(capa7MLv1)
    outputdataMLv2 = keras.layers.Dense(3, activation = "linear")(capa8MLv1)


    modelopred = keras.models.Model(inputs=[inputdataMLv2], outputs=[outputdataMLv2]) 

    modelopred.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=.001))
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.05, mode="min" ,patience=5, restore_best_weights=True)

    history= modelopred.fit(Xnorm[train[0]:(train[-1]+1)], Y[train[0]:(train[-1]+1)], epochs=2000,
    validation_data=(Xnorm[valid[0]:(valid[-1]+1)], Y[valid[0]:(valid[-1]+1)]), callbacks=[callback], verbose=1)


    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca() # set the vertical range to [0-1]
    plt.title("Tasa de aprendizaje")
    plt.show()
    scores = [modelopred.evaluate(Xnorm[train[0]:(train[-1]+1)], Y[train[0]:(train[-1]+1)], verbose=0), modelopred.evaluate(Xnorm[valid[0]:(valid[-1]+1)], Y[valid[0]:(valid[-1]+1)])]
    if scores[1] < mejor[1]:
        modeloMLv2 = modelopred
        mejor = scores
        validf = valid
    print(f'Score for fold {fold_no}: loss of {scores}')
    fold_no+=1

graficas_result(modeloMLv2, Xnorm[validf[0]:(validf[-1]+1)], Y[validf[0]:(validf[-1]+1)])
print(f"Scoring del modelo con respecto al MSE: {mejor}")

modeloMLv2.save(r'C:\Users\Administrador\Documents\cosas_universidad\TFG\modelos\modeloMLv2.h5')

# %%
#Este ya es un modelo bastante bueno, pero solo lo hemos probado para los datos de Madrid veamos su rendimiento para los datos de todas las ciudades
data1 = pd.read_table(r"C:\Users\Administrador\Documents\cosas_universidad\TFG\Viento\Datos_viento_Madrid.txt" , header=0 , sep = "\t")
data1 = data1.dropna() #Quitamos los Nan
data2 = pd.read_table(r"C:\Users\Administrador\Documents\cosas_universidad\TFG\Viento\Datos_viento_Coruña.txt" , header=0 , sep = "\t")
data2 = data2.dropna()
data3 = pd.read_table(r"C:\Users\Administrador\Documents\cosas_universidad\TFG\Viento\Datos_viento_Tarifa.txt" , header=0 , sep = "\t")
data3 = data3.dropna()
data4 = pd.read_table(r"C:\Users\Administrador\Documents\cosas_universidad\TFG\Viento\Datos_viento_Barcelona.txt" , header=0 , sep = "\t")
data4 = data4.dropna()

datag=pd.concat([data1,data2,data3,data4], ignore_index=True)
scaler=Normalizer() #Vamos a noramlizar los datos de los predictores 
scaler.fit(data1.drop(['fecha','obsU','obsV','obsM'], axis=1))
Xg = datag.drop(['fecha','obsU','obsV','obsM'], axis=1)
Xscaler = scaler.transform(Xg)
Xnormg = pd.DataFrame(Xscaler,  columns=['predU10m','predV10m','predM10m','predU150m','predV150m','predM150m','predMes','predHora','predMesU','predMesV','predHoraU','predHoraV','predDia','predDiaU','predDiaV'])
Yg=datag.drop(['fecha','predU10m','predV10m','predM10m','predU150m','predV150m','predM150m','predMes','predHora','predMesU','predMesV','predHoraU','predHoraV','predDia','predDiaU','predDiaV'], axis=1)

xtestg, xtraing, ytestg, ytraing = train_test_split(Xnormg, Yg, test_size= .15, random_state = 100)
#%%
print(f"Scoring del modelo para todos los datos respecto al MSE: {modeloMLv2.evaluate(xtraing.iloc[0:100],ytraing.iloc[0:100] , verbose=0)}")
graficas_result(modeloMLv2, xtestg, ytestg)
#%%
#Modelo multicapa para todos los datos disponibles
#Primero tratemos los datos 
scaler=Normalizer() #Vamos a noramlizar los datos de los predictores 
Xg = datag.drop(['fecha','obsU','obsV','obsM'], axis=1)
Xscaler = scaler.fit_transform(Xg)
Xnormg = pd.DataFrame(Xscaler,  columns=['predU10m','predV10m','predM10m','predU150m','predV150m','predM150m','predMes','predHora','predMesU','predMesV','predHoraU','predHoraV','predDia','predDiaU','predDiaV'])
Yg=datag.drop(['fecha','predU10m','predV10m','predM10m','predU150m','predV150m','predM150m','predMes','predHora','predMesU','predMesV','predHoraU','predHoraV','predDia','predDiaU','predDiaV'], axis=1)

#%%
fold_no=1
compare_scores= []
mejor=[0,999999999] #Voy poner un valor arbitrario que va a superar seguro 

kf= KFold(n_splits =5, shuffle= True ,random_state=8)
for train, valid in kf.split(Xnormg, Yg):

    inputdataMLv3 = keras.layers.Input(shape=(15,)) 
    capa1MLv3 = keras.layers.Dense(1000, activation = "relu")(inputdataMLv3)
    capa2MLv3 = keras.layers.Dense(950, activation = "relu")(capa1MLv3)
    capa3MLv3 = keras.layers.Dense(900, activation = "relu")(capa2MLv3)
    capa4MLv3 = keras.layers.Dense(850, activation = "relu")(capa3MLv3)
    capa5MLv3 = keras.layers.Dense(800, activation = "relu")(capa4MLv3)
    capa6MLv3 = keras.layers.Dense(750, activation = "relu")(capa5MLv3)
    capa7MLv3 = keras.layers.Dense(700, activation = "relu")(capa6MLv3)
    capa8MLv3 = keras.layers.Dense(650, activation = "relu")(capa7MLv3)
    outputdataMLv3 = keras.layers.Dense(3, activation = "linear")(capa8MLv3)


    modelopred = keras.models.Model(inputs=[inputdataMLv3], outputs=[outputdataMLv3]) 

    modelopred.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=.001))
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.05, mode="min" ,patience=5, restore_best_weights=True)

    history= modelopred.fit(Xnormg[train[0]:(train[-1]+1)], Yg[train[0]:(train[-1]+1)], epochs=2000,
    validation_data=(Xnormg[valid[0]:(valid[-1]+1)], Yg[valid[0]:(valid[-1]+1)]), callbacks=[callback], verbose=1)


    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca() # set the vertical range to [0-1]
    plt.title("Tasa de aprendizaje")
    plt.show()
    scores = [modelopred.evaluate(Xnormg[train[0]:(train[-1]+1)], Yg[train[0]:(train[-1]+1)], verbose=0), modelopred.evaluate(Xnormg[valid[0]:(valid[-1]+1)], Yg[valid[0]:(valid[-1]+1)])]
    if scores[1] < mejor[1]:
        modeloMLv3 = modelopred
        mejor = scores
        validf= valid
    print(f'Score for fold {fold_no}: loss of {scores}')
    fold_no+=1

graficas_result(modeloMLv3, Xnormg[validf[0]:(validf[-1]+1)], Yg[validf[0]:(validf[-1]+1)])
print(f"Scoring del modelo con respecto al MSE: {mejor}")

modeloMLv3.save(r'C:\Users\Administrador\Documents\cosas_universidad\TFG\modelos\modeloMLv3.h5')

# %%
#Como podemos ver no conseguimos los mismos resultados que para el mismo modelo pero de solamente Madrid, por lo que podemos deducir que es una tarea más compleja
#Vamos a hacer ahora un modelo más especializado, ya que nuestra solución son 3 coordenadas vamos a dividir el problema en 3. Creando una red para cada coordenada de la solución
#Ya que dividimos la solución vamos a intentar reducir la dimensión de cada una de las redes para que el coste computacional no sea tan grande
fold_no=1
compare_scores= []
mejor=[0,999999999] #Voy poner un valor arbitrario que va a superar seguro 

kf= KFold(n_splits =5, shuffle= True ,random_state=8)
for train, valid in kf.split(Xnormg, Yg):

    inputdataMLv4 = keras.layers.Input(shape=(15,)) 


    capa1MLv41 = keras.layers.Dense(1000, activation = "relu")(inputdataMLv4)
    capa2MLv41 = keras.layers.Dense(950, activation = "relu")(capa1MLv41)
    capa3MLv41 = keras.layers.Dense(900, activation = "relu")(capa2MLv41)
    capa4MLv41 = keras.layers.Dense(850, activation = "relu")(capa3MLv41)
    capa5MLv41 = keras.layers.Dense(800, activation = "relu")(capa4MLv41)
    outMLv41 = keras.layers.Dense(1, activation = "linear")(capa5MLv41)

    capa1MLv42 = keras.layers.Dense(1000, activation = "relu")(inputdataMLv4)
    capa2MLv42 = keras.layers.Dense(950, activation = "relu")(capa1MLv42)
    capa3MLv42 = keras.layers.Dense(900, activation = "relu")(capa2MLv42)
    capa4MLv42 = keras.layers.Dense(850, activation = "relu")(capa3MLv42)
    capa5MLv42 = keras.layers.Dense(800, activation = "relu")(capa4MLv42)
    outMLv42 = keras.layers.Dense(1, activation = "linear")(capa5MLv42)

    capa1MLv43 = keras.layers.Dense(1000, activation = "relu")(inputdataMLv4)
    capa2MLv43 = keras.layers.Dense(950, activation = "relu")(capa1MLv43)
    capa3MLv43 = keras.layers.Dense(900, activation = "relu")(capa2MLv43)
    capa4MLv43 = keras.layers.Dense(850, activation = "relu")(capa3MLv43)
    outMLv43 = keras.layers.Dense(1, activation = "linear")(capa4MLv43)

    outputdataMLv4 = keras.layers.concatenate([outMLv41, outMLv42, outMLv43])


    modelopred = keras.models.Model(inputs=[inputdataMLv4], outputs=[outputdataMLv4]) 

    modelopred.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=.001))
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.05, mode="min" ,patience=5, restore_best_weights=True)

    history= modelopred.fit(Xnormg[train[0]:(train[-1]+1)], Yg[train[0]:(train[-1]+1)], epochs=2000,
    validation_data=(Xnormg[valid[0]:(valid[-1]+1)], Yg[valid[0]:(valid[-1]+1)]), callbacks=[callback], verbose=1)


    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca() # set the vertical range to [0-1]
    plt.title("Tasa de aprendizaje")
    plt.show()
    scores = [modelopred.evaluate(Xnormg[train[0]:(train[-1]+1)], Yg[train[0]:(train[-1]+1)], verbose=0), modelopred.evaluate(Xnormg[valid[0]:(valid[-1]+1)], Yg[valid[0]:(valid[-1]+1)])]
    if scores[1] < mejor[1]:
        modeloMLv4 = modelopred
        mejor = scores
        validf= valid
    print(f'Score for fold {fold_no}: loss of {scores}')
    fold_no+=1

graficas_result(modeloMLv4, Xnormg[validf[0]:(validf[-1]+1)], Yg[validf[0]:(validf[-1]+1)])
print(f"Scoring del modelo con respecto al MSE: {mejor}")

modeloMLv4.save(r'C:\Users\Administrador\Documents\cosas_universidad\TFG\modelos\modeloMLv4.h5')

# %%
#Con esto hemos conseguido un Modelo bastante bueno, pero vamos a intentar hacer otro intento 
#cambiando un poco la geometría del modelo, para ello vamos a juntar las ideas de los modelos anteriores
#creando un modelo que haga una predicción y usaremos los datos de esa predicción para hacer una predicción
#real, como si fuese un método implícito

#Para ello usaré como modelo predictivo inicial uno de los modelos creados anteriormente, por ejemplo el modelo 3

modelo3 = keras.models.load_model(r'C:\Users\Administrador\Documents\cosas_universidad\TFG\modelos\modeloMLv4.h5')
modelo_pred_inic=keras.models.clone_model(modelo3)
modelo_pred_inic.set_weights(modelo3.get_weights())
modelo_pred_inic.trainable=False
#He puesto que no sea entrenable, ya que el modelo ya está entrenado para este conjunto de datos, y solo 
#Queremos una aproximación a la predicción real

#ahora nos haremos el modelo de verdad, usando la estructura del modelo 4, reduciendo un poco el modelo
fold_no=1
compare_scores= []
mejor=[0,999999999] #Voy poner un valor arbitrario que va a superar seguro 

kf= KFold(n_splits =5, shuffle= True ,random_state=8)
for train, valid in kf.split(Xnormg, Yg):

    inputdataMLv5 = keras.layers.Input(shape=(15,)) 

    capa_pred_inic = modelo3(inputdataMLv5)

    input_MLv5 = keras.layers.concatenate([inputdataMLv5, capa_pred_inic])

    capa1MLv51 = keras.layers.Dense(1000, activation = "relu")(input_MLv5)
    capa2MLv51 = keras.layers.Dense(950, activation = "relu")(capa1MLv51)
    capa3MLv51 = keras.layers.Dense(900, activation = "relu")(capa2MLv51)
    capa4MLv51 = keras.layers.Dense(850, activation = "relu")(capa3MLv51)
    outMLv51 = keras.layers.Dense(1, activation = "linear")(capa4MLv51)

    capa1MLv52 = keras.layers.Dense(1000, activation = "relu")(input_MLv5)
    capa2MLv52 = keras.layers.Dense(950, activation = "relu")(capa1MLv52)
    capa3MLv52 = keras.layers.Dense(900, activation = "relu")(capa2MLv52)
    capa4MLv52 = keras.layers.Dense(850, activation = "relu")(capa3MLv52)
    outMLv52 = keras.layers.Dense(1, activation = "linear")(capa4MLv52)

    capa1MLv53 = keras.layers.Dense(1000, activation = "relu")(input_MLv5)
    capa2MLv53 = keras.layers.Dense(950, activation = "relu")(capa1MLv53)
    capa3MLv53 = keras.layers.Dense(900, activation = "relu")(capa2MLv53)
    capa4MLv53 = keras.layers.Dense(850, activation = "relu")(capa3MLv53)
    outMLv53 = keras.layers.Dense(1, activation = "linear")(capa4MLv53)

    outputdataMLv5 = keras.layers.concatenate([outMLv51, outMLv52, outMLv53])


    modelopred = keras.models.Model(inputs=[inputdataMLv5], outputs=[outputdataMLv5]) 

    modelopred.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=.001))
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.05, mode="min" ,patience=5, restore_best_weights=True)

    history= modelopred.fit(Xnormg[train[0]:(train[-1]+1)], Yg[train[0]:(train[-1]+1)], epochs=2000,
    validation_data=(Xnormg[valid[0]:(valid[-1]+1)], Yg[valid[0]:(valid[-1]+1)]), callbacks=[callback], verbose=1)


    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca() # set the vertical range to [0-1]
    plt.title("Tasa de aprendizaje")
    plt.show()
    scores = [modelopred.evaluate(Xnormg[train[0]:(train[-1]+1)], Yg[train[0]:(train[-1]+1)], verbose=0), modelopred.evaluate(Xnormg[valid[0]:(valid[-1]+1)], Yg[valid[0]:(valid[-1]+1)])]
    if scores[1] < mejor[1]:
        modeloMLv5 = modelopred
        mejor = scores
        validf= valid
    print(f'Score for fold {fold_no}: loss of {scores}')
    fold_no+=1

graficas_result(modeloMLv5, Xnormg[validf[0]:(validf[-1]+1)], Yg[validf[0]:(validf[-1]+1)])
print(f"Scoring del modelo con respecto al MSE: {mejor}")

modeloMLv5.save(r'C:\Users\Administrador\Documents\cosas_universidad\TFG\modelos\modeloMLv4.h5')

