# Importación de librerias
import os
import re
from collections import Counter
from datetime import datetime

import cv2
import nltk
import numpy as np
import pytesseract
import torch
from langdetect import detect
from libretranslatepy import LibreTranslateAPI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from aztech_utils import objetos_plural, objetos_singular


def imprimir_timestamp(texto):
    formato = "%H:%M:%S.%f"
    print("[ai_aztech]", datetime.now().strftime(formato)[:-3], texto)


# Configuración de la ruta de PyTesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Configuración del idioma al que se desea traducir
IDIOMA_USUARIO = "es"

# Hace la conexión con la api por medio de internet.
traductor = LibreTranslateAPI('https://libretranslate.org/')

# Descargar componentes necesarios en caso de que no estén presentes
nltk.download('punkt')
nltk.download('stopwords')


# Inicializar el modelo YOLO
def inicializar_modelo():
    # Obtener la ruta absoluta del directorio actual
    current_dir = os.path.abspath(os.path.dirname(__file__))

    # Cargar modelo
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=os.path.join(current_dir, 'aztech.pt'))

    return model


# Inicializar el modelo YOLO una vez al iniciar la aplicación
imprimir_timestamp("Inicializando modelo de YOLO.")
modelo_yolo = inicializar_modelo()
imprimir_timestamp("Modelo de YOLO cargado.")


def inicializar_camara():
    # Inicializar la captura del vídeo con la entrada 0
    cap = cv2.VideoCapture(0)

    # Si la entrada 0 no funciona, intentar con la 1.
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError("No se puede abrir el vídeo.")

    return cap


# Inicializar la cámara
imprimir_timestamp("Inicializando cámara.")
cap = inicializar_camara()
imprimir_timestamp("Cámara iniciada.")


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
        try:
            idioma_detectado = detect(texto)
            if idioma_detectado != IDIOMA_USUARIO:
                texto = traducir_texto(texto, idioma_detectado)
        except Exception as e:
            print("Error con la solicitud de la API de detección de texto:", str(e))
    return texto


def traducir_texto(texto, idioma_original):
    try:
        traduccion = traductor.translate(
            texto, idioma_original, IDIOMA_USUARIO)
        return traduccion
    except Exception as e:
        print("Error con la solicitud de la API de traducción:", str(e))


def construir_salida_tesseract(texto):
    if not texto or texto == '':
        msg = "Parece que detecté un texto, pero no puedo leer su contenido"
    else:
        texto = limpiar_texto(texto)
        texto = detectar_idioma_y_traducir(texto)

        msg = "Se detectó un texto que dice " + str(texto)
        return msg


def construir_salida_yolo(info):
    # Mostrar las detecciones en la imagen
    lista_labels = info.iloc[:, 6].to_string(index=False)

    # Retornar vacío si no se encontró nada.
    if lista_labels.strip() == "Series([], )":
        return ""

    # Separar los objetos y juntarlos en una lista
    labels = lista_labels.split('\n')

    # Quitar espacios en blanco de los elementos
    labels = [label.strip() for label in labels]

    # Inicializar resultado
    resultado = "Se ha detectado" if len(labels) == 1 else "Se han detectado"

    # Contar cuántos objetos se detectaron
    contador = Counter(labels)
    objetos_detectados = []

    for elemento, cantidad in contador.items():
        # Verificar si se necesita hacer plural
        if cantidad > 1:
            # Pone el objeto del diccionario en plural
            elemento = objetos_plural.get(elemento)
        else:
            # Obtiene el objeto del diccionario singular
            elemento = objetos_singular.get(elemento)
        objetos_detectados.append(f"{cantidad} {elemento}")

    # Construir el mensaje final
    resultado += " " + " y ".join(objetos_detectados)

    # Imprimir el resultado final
    return resultado


def procesar_video():
    is_detectando_texto = True

    # Inicio del main.
    while True:
        imprimir_timestamp("Iniciando nueva iteración.")
        # Lectura de frames
        imprimir_timestamp("Leyendo imagen desde la cámara.")
        ret, frame = cap.read()
        imprimir_timestamp("Imagen leída")

        # Detectar objetos
        imprimir_timestamp("Analizando objetos con el modelo de YOLO.")
        detect = modelo_yolo(frame)
        info = detect.pandas().xyxy[0]
        imprimir_timestamp("Análisis de objetos completado.")

        imprimir_timestamp("Construyendo salida de YOLO.")
        info_str = construir_salida_yolo(info)
        imprimir_timestamp("Salida de YOLO completada.")

        # Inicialización de texto
        texto = ""

        # Solo detectar texto si el usuario lo quiere
        if is_detectando_texto:
            # Obtener el texto de la imagen
            imprimir_timestamp("Analizando imagen con PyTesseract.")
            texto = pytesseract.image_to_string(frame)
            imprimir_timestamp("Análisis con PyTesseract completado.")
            if texto:
                imprimir_timestamp("Construyendo salida de PyTesseract.")
                texto = construir_salida_tesseract(texto)
                imprimir_timestamp("Salida de PyTesseract construida.")

        imprimir_timestamp("Construyendo string de salida.")
        if info_str and not texto:  # Si detectó objetos y no texto
            salida = info_str
        elif info_str and texto:  # Si detectó objetos y texto
            salida = info_str + " y " + texto
        elif texto and not info_str:  # Si detectó texto y no objetos
            salida = texto
        else:  # Si no detectó objetos ni texto
            salida = ""
        imprimir_timestamp("String de salida construído.")

        imgH, imgW, _ = frame.shape

        x1, y1, w1, h1 = 0, 0, imgH, imgW

        # Poner en pantalla el texto obtenido
        cv2.putText(frame, salida, (x1 + int(w1 / 50), y1 + int(h1 / 50) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Mostrar la imagen en la ventana
        cv2.imshow('Ai_Aztech', np.squeeze(detect.render()))

        print(salida)

        # Condición de terminación
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    procesar_video()


if __name__ == "__main__":
    main()
