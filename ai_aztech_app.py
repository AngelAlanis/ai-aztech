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

from flask import Flask, request, jsonify

app = Flask(__name__)

# Configuración de la ruta de PyTesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Configuración del idioma al que se desea traducir
IDIOMA_USUARIO = "es"

# Hace la conexión con la api por medio de internet.
traductor = LibreTranslateAPI('https://libretranslate.org/')

# Descargar componentes necesarios en caso de que no estén presentes
nltk.download('punkt')
nltk.download('stopwords')

# Diccionario para la detección de objetos

objetos = {
    "persona": "Se ha detectado a una persona.",
    "bicicleta": "Se ha detectado una bicicleta.",
    "carro": "Se ha detectado un carro.",
    "motocicleta": "Se ha detectado una motocicleta.",
    "avión": "Se ha detectado un avión.",
    "autobús": "Se ha detectado un autobús.",
    "tren": "Se ha detectado un tren.",
    "camion": "Se ha detectado un camión.",
    "bote": "Se ha detectado un bote.",
    "luz de semáforo": "Se ha detectado una luz de semáforo.",
    "hidrante": "Se ha detectado un hidrante.",
    "señal de parada": "Se ha detectado una señal de parada.",
    "parquímetro": "Se ha detectado un parquímetro.",
    "banco": "Se ha detectado un banco.",
    "pájaro": "Se ha detectado un pájaro.",
    "gato": "Se ha detectado un gato.",
    "perro": "Se ha detectado un perro.",
    "caballo": "Se ha detectado un caballo.",
    "oveja": "Se ha detectado una oveja.",
    "baca": "Se ha detectado una baca.",
    "elefante": "Se ha detectado un elefante.",
    "oso": "Se ha detectado un oso.",
    "cebra": "Se ha detectado una cebra.",
    "jirafa": "Se ha detectado una jirafa.",
    "mochila": "Se ha detectado una mochila.",
    "paraguas": "Se ha detectado un paraguas.",
    "bolsa de mano": "Se ha detectado una bolsa de mano.",
    "corbata": "Se ha detectado una corbata.",
    "frisbee": "Se ha detectado un frisbee.",
    "esquís": "Se han detectado esquís.",
    "snowboard": "Se ha detectado un snowboard.",
    "pelota": "Se ha detectado una pelota.",
    "cometa": "Se ha detectado una cometa.",
    "bate de beisbol": "Se ha detectado un bate de béisbol.",
    "guante de beisbol": "Se ha detectado un guante de béisbol.",
    "patineta": "Se ha detectado una patineta.",
    "tabla de surf": "Se ha detectado una tabla de surf.",
    "raqueta de tenis": "Se ha detectado una raqueta de tenis.",
    "botella": "Se ha detectado una botella.",
    "copa de vino": "Se ha detectado una copa de vino.",
    "taza": "Se ha detectado una taza.",
    "tenedor": "Se ha detectado un tenedor.",
    "cuchillo": "Se ha detectado un cuchillo.",
    "cuchara": "Se ha detectado una cuchara.",
    "cuenco": "Se ha detectado un cuenco.",
    "platano": "Se ha detectado un plátano.",
    "manzana": "Se ha detectado una manzana.",
    "sandwich": "Se ha detectado un sandwich.",
    "naranja": "Se ha detectado una naranja.",
    "brocoli": "Se ha detectado un brócoli.",
    "zanahoria": "Se ha detectado una zanahoria.",
    "perro caliente": "Se ha detectado un perro caliente.",
    "pizza": "Se ha detectado una pizza.",
    "dona": "Se ha detectado una dona.",
    "pastel": "Se ha detectado un pastel.",
    "silla": "Se ha detectado una silla.",
    "sofa": "Se ha detectado un sofá.",
    "planta en maceta": "Se ha detectado una planta en maceta.",
    "cama": "Se ha detectado una cama.",
    "comedor": "Se ha detectado un comedor.",
    "baño": "Se ha detectado un baño.",
    "tv": "Se ha detectado una TV.",
    "laptop": "Se ha detectado una laptop.",
    "mouse": "Se ha detectado un mouse.",
    "control remoto": "Se ha detectado un control remoto.",
    "teclado": "Se ha detectado un teclado.",
    "telefono": "Se ha detectado un teléfono.",
    "microondas": "Se ha detectado un microondas.",
    "horno": "Se ha detectado un horno.",
    "tostadora": "Se ha detectado una tostadora.",
    "fregadero": "Se ha detectado un fregadero.",
    "refrigerador": "Se ha detectado un refrigerador.",
    "libro": "Se ha detectado un libro.",
    "reloj": "Se ha detectado un reloj.",
    "florero": "Se ha detectado un florero.",
    "tijeras": "Se han detectado tijeras.",
    "oso de peluche": "Se ha detectado un oso de peluche.",
    "secadora de pelo": "Se ha detectado una secadora de pelo.",
    "cepillo de dientes": "Se ha detectado un cepillo de dientes."
}

# Inicializar el modelo YOLO


def inicializar_modelo():
    # Obtener la ruta absoluta del directorio actual
    current_dir = os.path.abspath(os.path.dirname(__file__))

    # Cargar modelo
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=os.path.join(current_dir, 'aztech.pt'))

    return model


# Inicializar el modelo YOLO una vez al iniciar la aplicación
modelo_yolo = inicializar_modelo()


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
        except Exception as e:
            print("Error con la solicitud de la API de detección de texto:", str(e))

        if idioma_detectado != IDIOMA_USUARIO:
            texto = traducir_texto(texto, idioma_detectado)

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

        msg = "Se detectó un texto que dice", texto

    return msg


@app.route('/procesar_imagen', methods=['POST'])
def procesar_imagen():
    # Obtener la imagen y la bandera desde la solicitud POST
    imagen = request.files['imagen']
    is_detectando_texto = request.form['is_detectando_texto']

    # Obtener la ruta absoluta del directorio actual
    current_dir = os.path.abspath(os.path.dirname(__file__))

    # Detectar objetos
    detect = modelo_yolo(imagen)
    info = detect.pandas().xyxy[0]

    info_str = info.iloc[:, 6].to_string(index=False)

    # Solo detectar texto si el usuario lo
    if is_detectando_texto.lower() == 'true':
        # Obtener el texto de la imagen
        texto = pytesseract.image_to_string(imagen)
        texto = construir_salida_tesseract(texto)

    salida = texto + info_str

    return jsonify(resultado=salida)


if __name__ == "__main__":
    app.run()
