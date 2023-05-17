import pytesseract
import cv2
import numpy as np
import time
from langdetect import detect
from libretranslatepy import LibreTranslateAPI

#
isPausado = False
isDetectandoTexto = True


# Configuración de la ruta de PyTesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configuración de la fuente para mostrar en ventana
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# Configuración del idioma al que se desea traducir
idioma_usuario = "es"

# Hace la conexion con la api por medio de internet.
traductor = LibreTranslateAPI('https://libretranslate.org/')

# Inicializar la captura del vídeo con la entrada 0
cap = cv2.VideoCapture(0)

# Si la entrada 0 no funciona, intentar con la 1.
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("No se puede abrir el vídeo.")

# Iniciar contador para la tasa de refresco.
contador = 0


def procesar_texto(texto):
    if texto != '':
        print(texto)
        idioma_detectado = detect(texto)

        if idioma_detectado != idioma_usuario:
            texto = traducir_texto(texto, idioma_detectado)
            print("Texto traducido:", texto)

    return texto


def traducir_texto(texto, idioma_original):
    traduccion = traductor.translate(texto, idioma_original, idioma_usuario)
    return traduccion


# Inicio del main.
while True:
    # Condición para pausar si el usuario lo decide.
    if not isPausado:
        ret, frame = cap.read()

        contador += 1
        if (contador % 20) == 0:

            # Solo detectar texto si el usuario lo
            if isDetectandoTexto:

                imgH, imgW, _ = frame.shape

                x1, y1, w1, h1 = 0, 0, imgH, imgW

                # Obtener el texto de la imagen
                texto = pytesseract.image_to_string(frame)

                texto = procesar_texto(texto)

                # Poner en pantalla el texto obtenido
                cv2.putText(frame, texto, (x1 + int(w1 / 50), y1 + int(h1 / 50) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Mostrar la imagen en la ventana
                cv2.imshow('PyTesseract', frame)

        # Condición de terminación
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()