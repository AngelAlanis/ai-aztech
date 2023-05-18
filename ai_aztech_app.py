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
from unidecode import unidecode

# Configuración de la ruta de PyTesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Configuración del idioma al que se desea traducir
IDIOMA_USUARIO = "es"

# Hace la conexion con la api por medio de internet.
print(datetime.now().strftime("%H:%M:%S.%f")[
      :-3], "Haciendo conexión con la API de LibreTranslate.")
traductor = LibreTranslateAPI('https://libretranslate.org/')
print(datetime.now().strftime("%H:%M:%S.%f")[
      :-3], "Conexión con la API terminada.")

# Descargar componentes necesarios en caso de que no estén presentes
print(datetime.now().strftime("%H:%M:%S.%f")
      [:-3], "Comprobando paquetes de nltk.")
nltk.download('punkt')
nltk.download('stopwords')
print(datetime.now().strftime("%H:%M:%S.%f")[
      :-3], "Comprobación de paquetes nltk terminados.")

# Banderas para el flujo del programa
is_detectando_texto = True

# Imagen
print(datetime.now().strftime("%H:%M:%S.%f")[:-3], "Cargando imagen.")
imagen = Image.open('resources/texto.png')
print(datetime.now().strftime("%H:%M:%S.%f")
      [:-3], "Imagen cargada correctamente.")


def limpiar_texto(text):
    print(datetime.now().strftime("%H:%M:%S.%f")[:-3], "Limpiando texto.")
    # Eliminar caracteres no deseados
    text = re.sub(r'[^\w\s]', '', text)

    # Convertir el texto a minúsculas
    text = text.lower()

    # Segmentación de palabras y eliminación de palabras vacías
    stop_words = set(stopwords.words('spanish'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)

    print(datetime.now().strftime("%H:%M:%S.%f")[:-3], "Texto limpiado.")
    return text


def detectar_idioma_y_traducir(texto):
    if texto != '':
        print(texto)
        print(datetime.now().strftime("%H:%M:%S.%f")
              [:-3], "Detectando idioma.")
        idioma_detectado = detect(texto)
        print(datetime.now().strftime("%H:%M:%S.%f")[:-3], "Idioma detectado.")

        if idioma_detectado != IDIOMA_USUARIO:
            print(datetime.now().strftime("%H:%M:%S.%f")
                  [:-3], "Traduciendo texto.")
            texto = traducir_texto(texto, idioma_detectado)
            print(datetime.now().strftime("%H:%M:%S.%f")
                  [:-3], "Texto traducido.")
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
    print(datetime.now().strftime("%H:%M:%S.%f")
          [:-3], "Cargando modelo de YOLO.")
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=os.path.join(current_dir, 'aztech.pt'))

    print(datetime.now().strftime("%H:%M:%S.%f")[
          :-3], "Modelo de YOLO cargado correctamente.")

    frame = imagen

    # Detectar objetos
    detect = model(frame)
    info = []
    print(datetime.now().strftime("%H:%M:%S.%f")[:-3], "Detectando objetos.")
    info = detect.pandas().xyxy[0]
    print(datetime.now().strftime("%H:%M:%S.%f")[:-3], "Objetos detectados.")

    print(info)

    # Solo detectar texto si el usuario lo
    if is_detectando_texto:

        # Obtener el texto de la imagen
        print(datetime.now().strftime("%H:%M:%S.%f")[
              :-3], "Detectando texto con PyTesseract.")
        texto = pytesseract.image_to_string(frame)
        print(datetime.now().strftime("%H:%M:%S.%f")[
              :-3], "Texto detectado con PyTesseract.")

        texto = limpiar_texto(texto)

        texto = detectar_idioma_y_traducir(texto)

        print(texto)


def ordenar_puntos(puntos):
    n_puntos = np.concatenate(
        [puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])

    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]


def main():
    procesar_imagen()


if __name__ == "__main__":
    main()
