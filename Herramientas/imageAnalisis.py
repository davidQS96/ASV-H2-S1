import skimage as sk
from skimage import color
import io
import tkinter as tk
from tkinter import filedialog #Manejo de archivos
import os
import numpy as np
import cv2


imagePath = "None"


#Para facilitar uso de variables de uso global
class globalVar():

    def __init__(self):
        self.frameNum = 0

gv = globalVar()


#Funcion que recibe una imagen en ndarray y muestra la imagen que quepa dentro de una pantalla de 1920x1080
def showImage(imageND, name = ""):
    #Cambio de tamaño apto para observar imagenes

    if type(imageND) != np.ndarray or imageND.ndim < 2 or imageND.ndim > 3: #BW o Color
        print("showImage: imageND no es tipo ndarray")
        return "showImage: imageND no es tipo ndarray", -1

    if type(name) != str:
        print("showImage: name no es tipo string")
        return "showImage: name no es tipo string", -1

    resizeFactor = resizeConstRatio(imageND, (1000, 720)) #primer argumento da dimensiones en y, x
    height, width = imageND.shape[:2]
    newImgSize = round(width * resizeFactor), round(height * resizeFactor)
    img = cv2.resize(imageND, newImgSize)

    if name == "":
        cv2.imshow('Frame ' + str(gv.frameNum), img)
        gv.frameNum += 1
    else:
        cv2.imshow(name, img)


#Función que muestra al usuario una ventana de búsqueda de archivos y devuelve el directorio donde se encuentra una imagen válida en una variable global
def openImage():
    currdir = os.getcwd()

    try:
        root = tk.Tk()
        #https://docs.python.org/3.9/library/dialog.html
        tempdir = filedialog.askopenfilename(parent = root, initialdir = currdir, title = 'Seleccione un archivo de imagen') #Ventana emergente
        #https://www.thetopsites.net/article/53470882.shtml

        if len(tempdir) > 0 and tempdir.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')): #Archivos de imagen soportados
            imagePath = tempdir #Asigna la imagen válida que se encontró
            print(imagePath)

        else:
            #Muestra mensaje de error en caso de que no se elija un archivo adecuado
            print("Archivo no válido, elija uno soportado por el programa (.png, .jpg, .jpeg, .tiff, .bmp, .gif)")

        root.destroy()

    except:
        print("Ventana se cerró inesperadamente")

    return


#Función que escala imagen para que se ajuste a la ventana con tamaño winDim
#imageND es un objeto ndarray, con .shape (y, x)
#winDim es una tupla (x, y) de la ventana en la que la imagen se ajusta
#Devuelve el nuevo tamaño y factor de escala por si se necesita
def resizeConstRatio(imageND, winDim):
    if type(imageND) != np.ndarray or imageND.ndim < 2 or imageND.ndim > 3: #BW o Color
        print("resizeConstRatio: imageND no es tipo ndarray")
        return "resizeConstRatio: imageND no es tipo ndarray", -1

    if type(winDim) != tuple or len(winDim) != 2:
        print("resizeConstRatio: winDim no es tipo tupla 2x1")
        return "resizeConstRatio: winDim no es tipo tupla 2x1", -1

    for i in winDim:
        if type(i) != int or i <= 0:
            print("resizeConstRatio: valores maximos en winDim no son validos")
            return "resizeConstRatio: valores maximos en winDim no son validos", -1

    y, x = imageND.shape[:2]
    xM, yM = winDim

    scaleFactor = 1

    relX, relY = (x / xM, y / yM)

    if relX > 1:
        if relY > 1:
            if relX > relY:
                scaleFactor = 1 / relX
            else:
                scaleFactor = 1 / relY

        else:
            scaleFactor = 1 / relX
    else:
        if relY > 1:
            scaleFactor = 1 / relY

        else:
            if relX > relY:
                scaleFactor = 1 / relX
            else:
                scaleFactor = 1 / relY

    return scaleFactor








