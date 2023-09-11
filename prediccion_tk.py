from tkinter import Tk,Label,Button,filedialog,messagebox
from os import path
from PIL import Image, ImageTk
import numpy as np
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model
global file
file = None
#Cargamos el modelo
altura, longitud = 150,150
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos)
#Abrimos una instancia de la clase Tk
window = Tk()
window.title("Clasificacion con CNN")
window.geometry('640x480')
#Funcion que carga la imagen y la transoforma a imagenTk
def load_images(original): #Recibe un PATH
    img = Image.open(original)
    nuevo_tam= (300,300)
    img.thumbnail(nuevo_tam,Image.ANTIALIAS)
    tkimage = ImageTk.PhotoImage(image=img)
    lbl3.configure(image=tkimage)
    lbl3.image=tkimage
    return 
#Funcion que recolecta el archivo
def clickbtn0():
    global file 
    x = filedialog.askopenfilename(initialdir= path.dirname(__file__),title ="Seleccione Archivo",filetypes=(("Archivos jpg","*.jpg"),("all files","*.*")))
    nombre_file = ''
    file = x
    if x:
        try:
            nombre_file = path.basename(x)
            lbl2.config(text=nombre_file)
            load_images(x)
        except ValueError as e:
            nombre_file = ''
            messagebox.showinfo("Alerta","Ocurrio un error al cargar el archivo")
    else:
        lbl2.configure(text="Seleccione un archivo")
    return
#Funcion que usa el modelo cargado para realizar una prediccion
def clasifica():
    global file
    if file is None:
        messagebox.showinfo("Alerta","No existe archivo para clasificar")
    else:      
        x = load_img(file, target_size=(altura,longitud))
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        arreglo = cnn.predict(x) 
        resultado = arreglo[0] 
        respuesta = np.argmax(resultado)
        rfinal = ''
        if respuesta == 0:
            rfinal='Clase tipo Blanca'
        elif respuesta == 1:
            rfinal='Clase tipo Negra'
        elif respuesta == 2:
            rfinal='Clase tipo SemiBlanca/SemiNegra'
        lbl4.configure(text=rfinal)
    return
#Titulo
lbl1 = Label(window, text="Prediccion Ballenas Jorobadas", font=("arial",18))
lbl1.place(x=150,y=0)
#label archivo buscado
lbl2 = Label(window, text="Seleccione un archivo", font=("arial",13))
lbl2.place(x=250,y=50)
#boton para buscar
btn0 = Button(window, text="Buscar...",command=clickbtn0)
btn0.place(x=150,y=50)
#Label Imagen
lbl3 = Label(window)
lbl3.place(x=150,y=150)
#Boton clasificar imagen 
btn1 = Button(window, text="Clasificar",command=clasifica)
btn1.place(x=150,y=380)
#label mostrar resultado
lbl4 = Label(window,font=("arial",15))
lbl4.place(x=250,y=380)
window.mainloop()