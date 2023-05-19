# Importación de librerias
import os
import re
from datetime import datetime

import nltk
import numpy as np
import pytesseract
import torch
from langdetect import detect
from libretranslatepy import LibreTranslateAPI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image

# Configuración de la ruta de PyTesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Configuración del idioma al que se desea traducir
IDIOMA_USUARIO = "es"

# Hace la conexión con la api por medio de internet.
traductor = LibreTranslateAPI('https://libretranslate.org/')

# Descargar componentes necesarios en caso de que no estén presentes
nltk.download('punkt')
nltk.download('stopwords')

# Banderas para el flujo del programa
is_detectando_texto = True

# Imagen
imagen = Image.open('resources/texto.png')


def limpiar_texto(texto):
    # Eliminar caracteres no deseados
    texto = re.sub(r'[^\w\s]', '', texto)

    # Convertir el texto a minúsculas
    texto = texto.lower()

    # Segmentación de palabras y eliminación de palabras vacías
    stop_words = set(stopwords.words('spanish'))
    words = word_tokenize(texto)
    filtered_words = [word for word in words if word not in stop_words]
    texto = ' '.join(filtered_words)

    return texto


def detectar_idioma_y_traducir(texto):
    if texto != '':
        print(texto)
        
        try:
            idioma_detectado = detect(texto)
        except Exception as e:
            print("Error con la solicitud de la API de detección de texto:", str(e))

        if idioma_detectado != IDIOMA_USUARIO:
            texto = traducir_texto(texto, idioma_detectado)
            print("Texto traducido:", texto)

    return texto


def traducir_texto(texto, idioma_original):
    try:
        traduccion = traductor.translate(
            texto, idioma_original, IDIOMA_USUARIO)
        return traduccion
    except Exception as e:
        print("Error con la solicitud de la API de traducción:", str(e))


def procesar_imagen():
    # Obtener la ruta absoluta del directorio actual
    current_dir = os.path.abspath(os.path.dirname(__file__))

    # Cargar modelo
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=os.path.join(current_dir, 'aztech.pt'))

    frame = imagen

    # Detectar objetos
    detect = model(frame)

    info = detect.pandas().xyxy[0]

    print(info)

    # Solo detectar texto si el usuario lo
    if is_detectando_texto:
        # Obtener el texto de la imagen
        texto = pytesseract.image_to_string(frame)

        texto = limpiar_texto(texto)

        texto = detectar_idioma_y_traducir(texto)

        print(texto)


def main():
    procesar_imagen()


if __name__ == "__main__":
    main()
