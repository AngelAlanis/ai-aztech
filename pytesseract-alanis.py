import pytesseract as tess
from PIL import Image
import PIL
import numpy as np
import cv2

#Definir función para pre-procesar las imágenes.
def preprocesar_imagen(imagen):
    kernel = np.ones((5, 5), np.uint8)

    imagen = np.array(imagen)
    imagen = cv2.resize(imagen, None, fx=2, fy=2)

    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = cv2.blur(imagen,(3,3))
    imagen = cv2.Canny(imagen, 150, 200)
    imagen = cv2.dilate(imagen, None, iterations=1)

    return imagen

#Función para imprimir el resultado dependiendo de si está vacío o no.
def imprimir_resultado(resultado):
    if resultado == "":
        print("No se detectó nada.")
    else:
         print(resultado)

def guardar_imagen(nombre_archivo, imagen):
    cv2.imwrite(nombre_archivo, imagen)

# Definir el PATH De PyTesseract.
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Importar las imágenes de /resources/
imagen1 = Image.open('resources/Screenshot_2.png')
imagen2 = Image.open('resources/pytesseract_prueba1.png')
imagen3 = Image.open('resources/pytesseract_prueba2.png')

# Pre-procesar las imágenes
imagen1 = preprocesar_imagen(imagen1)
imagen2 = preprocesar_imagen(imagen2)
imagen3 = preprocesar_imagen(imagen3)

#Guardando la imagen
guardar_imagen('test1.png', imagen1)
guardar_imagen('test2.png', imagen2)
guardar_imagen('test3.png', imagen3)

#Obtener los resultados
result1 = tess.image_to_string(imagen1, lang='spa+eng', config='--psm 11')
result2 = tess.image_to_string(imagen2, lang='spa+eng', config='--psm 12')
result3 = tess.image_to_string(imagen3, lang='spa+eng', config='--psm 11')

#Imprimir los resultados.
imprimir_resultado(result1)
imprimir_resultado(result2)
imprimir_resultado(result3)
