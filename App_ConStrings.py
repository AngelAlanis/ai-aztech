#Importación de librerias
import re
import time
import cv2
#Librerias para la detección de textos
import nltk
import numpy as np
import pytesseract
from langdetect import detect
from libretranslatepy import LibreTranslateAPI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
#Librerias de YOLO
import numpy as np
import torch
import os


# Configuración de la ruta de PyTesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
#pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Karely.LapKarely\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract'

# Configuración de la fuente para mostrar en ventana
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# Configuración del idioma al que se desea traducir
IDIOMA_USUARIO = "es"

# Hace la conexion con la api por medio de internet.
traductor = LibreTranslateAPI('https://libretranslate.org/')

# Descargar componentes necesarios en caso de que no estén presentes
nltk.download('punkt')
nltk.download('stopwords')

# Banderas para el flujo del programa
is_pausado = False
is_detectando_texto = True


def limpiar_texto(text):
    # Eliminar caracteres no deseados
    text = re.sub(r'[^\w\s]', '', text)

    # Convertir el texto a minúsculas
    text = text.lower()

    # Eliminar acentos y diacríticos
    text = unidecode(text)

    # Segmentación de palabras y eliminación de palabras vacías
    stop_words = set(stopwords.words('spanish'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)

    return text


def detectar_idioma_y_traducir(texto):
    if texto != '':
        print(texto)
        idioma_detectado = detect(texto)

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


def procesar_video():
    # Inicializar la captura del vídeo con la entrada 0
    cap = cv2.VideoCapture(0)

    # Obtener la ruta absoluta del directorio actual
    current_dir = os.path.abspath(os.path.dirname(__file__))

    # Cargar modelo
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(current_dir,'aztech.pt'))

    # Si la entrada 0 no funciona, intentar con la 1.
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError("No se puede abrir el vídeo.")

    # Iniciar contador para la tasa de refresco.
    contador = 0

    # Inicio del main.
    while True:
        # Condición para pausar si el usuario lo decide.
        if not is_pausado:
        
            contador += 1
            if (contador % 20) == 0:
                #Lectura de frames
                ret, frame = cap.read()

                #Detectar objetos
                detect=model(frame)
                info=[]
                info=detect.pandas().xyxy[0]

                print(info)

                # Solo detectar texto si el usuario lo
                if is_detectando_texto:

                    imgH, imgW, _ = frame.shape

                    x1, y1, w1, h1 = 0, 0, imgH, imgW

                    # Obtener el texto de la imagen
                    texto = pytesseract.image_to_string(frame)

                    #Se verifica que el texto se haya detectado

                    if not texto or texto=="":
                        msg="Parece que detecté un texto, pero no puedo leer su contenido"
                    else:
                        texto = limpiar_texto(texto)
                        texto = detectar_idioma_y_traducir(texto)
                        # Poner en pantalla el texto obtenido
                        cv2.putText(frame, texto, (x1 + int(w1 / 50), y1 + int(h1 / 50) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        #Se guarda en una variable el mensaje a transmitir al usuario
                        msg="Se detectó un texto que dice "+texto

                    # Mostrar la imagen en la ventana
                    cv2.imshow('Ai_Aztech', np.squeeze(detect.render()))

                
            # Condición de terminación
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
       

def ordenar_puntos(puntos):
    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])
    
    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

def main():
    procesar_video()



if __name__ == "__main__":
    main()