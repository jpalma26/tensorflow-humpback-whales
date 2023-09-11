import numpy as np
import datetime
import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.keras.utils import to_categorical #ponerle categorias a las salidas con keras
import pickle

K.clear_session()
datos_total = 3000 #cantidad de datos totales
D_entrenamiento = int(datos_total*0.8) #80% para entrenamiento
D_validacion = int(datos_total*0.2) #20% para validacion
data_train = './datos/train' # directorio entrenamiento
data_test = './datos/test' #directorio validacion

#Parametros Red Neuronal
epocas = 10 #numero de iteraciones sobre el set de datos
altura, longitud = 150,150 #tamano al que vamos a procesar las imagenes
batch_size = 32 #numero de imagenes que le mandaremos a procesar en cada uno de los pasos
pasos = int(D_entrenamiento/batch_size) #numero de veces que se va  procesar la informacion en cada epoca
pasos_validacion = int(D_validacion/batch_size) #al final de cada epoca se correra con vaidacion
filtrosConv1 = 32 
filtrosConv2 = 64
tamano_filtro1 = (3,3) #para primera conv altura de 3 y anchura 3
tamano_filtro2 = (2,2)
tamano_pool = (2,2) #tamano de filtro para el MaxPooling
clases = 3 #jorobada_blanca, jorobada_negra, jorobada_semiblanca
lr = 0.0005 #learning rate(ajuste de la red para acercarse a la solucion optima)
nombre_clases = ['Jorobada Blanca',"Jorobada Negra", "Jorobada Semi Blanca/Negra"]


#pre procesamiento de imagenes
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. /255, #rescalame todas las imagenes y dividelas por 255
    shear_range= 0.3, #generar imagenes pero inclinarlas
    zoom_range= 0.2, #toma imagenes les hace zoom
    rotation_range= 5,
    horizontal_flip=True #toma imagen y la invierte distinguir redireccionalidad
)

validacion_datagen = ImageDataGenerator(
    rescale=1./255
)

#generar imagenes que usaremos en la CNN
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_train, #extrae las imagenes del directorio entrenamiento
    target_size=(altura,longitud),
    batch_size= batch_size,
    class_mode='categorical' #jorobada_blanca, jorobada_negra, jorobada_semiblanca
)

imagen_validacion = validacion_datagen.flow_from_directory(
    data_test,
    target_size=(altura,longitud),
    batch_size= batch_size,
    class_mode='categorical'
)

#Creamos la red convolucional
cnn = Sequential()
#primera capa con un primer filtro
cnn.add(Convolution2D(filtrosConv1,tamano_filtro1,padding='same',input_shape=(altura,longitud,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
#segunda capa con otro filtro y luego agrupación
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2,padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
#aplanamos los datos, y le damos profundidad
cnn.add(Flatten())
cnn.add(Dense(256,activation='relu'))#256 neuronas
cnn.add(Dropout(0.5))
cnn.add(Dense(clases,activation='softmax'))
#compilamos el modelo con el optimizador Adam con un learning rate de 0.5
cnn.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=lr), 
            metrics=['accuracy'])
#ajustamos el modelo a la data
cnn.fit(imagen_entrenamiento,
        epochs=epocas, 
        steps_per_epoch=pasos,
        validation_data=imagen_validacion,
        validation_steps=pasos_validacion)

archivo_historia = open("historia_deep_learning.pckl","wb")
pickle.dump(cnn.history,archivo_historia)
archivo_historia.close()
print(imagen_entrenamiento.class_indices)
#guardamos el modelo
dir = './modelo/'
if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('./modelo/modelo.h5') #estructura del modelo   
cnn.save_weights('./modelo/pesos.h5')   #pesos de las capas