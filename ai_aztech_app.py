# Importación de librerias
import os
import re
from collections import Counter
from datetime import datetime

import nltk
import numpy as np
import pytesseract
import torch
from flask import Flask, jsonify, render_template, request, url_for
from langdetect import detect
from libretranslatepy import LibreTranslateAPI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image

from aztech_utils import objetos_plural, objetos_singular


def imprimir_timestamp(texto):
    formato = "%H:%M:%S.%f"
    print(datetime.now().strftime(formato)[:-3], texto)


app = Flask(__name__)

# Configuración de la ruta de PyTesseract
pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"

# Configuración del idioma al que se desea traducir
IDIOMA_USUARIO = "es"

# Hace la conexión con la api por medio de internet.
traductor = LibreTranslateAPI("https://libretranslate.org/")

# Descargar componentes necesarios en caso de que no estén presentes
nltk.download("punkt")
nltk.download("stopwords")

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


def limpiar_texto(texto):
    # Eliminar caracteres no deseados
    texto = re.sub(r"[^\w\s]", "", texto)

    # Convertir el texto a minúsculas
    texto = texto.lower()

    # Segmentación de palabras y eliminación de palabras vacías
    stop_words = set(stopwords.words("spanish"))
    words = word_tokenize(texto)
    filtered_words = [word for word in words if word not in stop_words]
    texto = " ".join(filtered_words)

    return texto


def detectar_idioma_y_traducir(texto):
    if texto != "":
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


@app.route("/")
def home():
    return render_template("inicio.html")


@app.route("/paginaPrincipal")
def main():
    return render_template("principal.html")


@app.route("/iniciarSesion")
def login():
    return render_template("iniciarSesion.html")


@app.route("/registropt1")
def singUp1():
    return render_template("registro1.html")


@app.route("/registropt2")
def singUp2():
    return render_template("registro2.html")


@app.route("/camara")
def cameraView():
    return render_template("home.html")


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

    print("Lista de labels:", lista_labels)

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


@app.route('/guardar_imagen', methods=['POST'])
def guardar_imagen():
    imagen_base64 = request.form['image']

    # Decodificar la imagen base64
    imagen_decodificada = imagen_base64.split(',')[1].encode()

    # Guardar la imagen en la ruta especificada
    ruta_guardado = 'images.png'
    with open(ruta_guardado, 'wb') as archivo:
        archivo.write(imagen_decodificada)

    return 'Imagen guardada exitosamente'


@app.route("/procesar_imagen", methods=["POST"])
def procesar_imagen():
    # Obtener la imagen y la bandera desde la solicitud POST
    imprimir_timestamp("Obteniendo imagen y bandera desde el servidor.")
    is_detectando_texto = request.form['is_detectando_texto']

    if 'imagen' not in request.files:
        return jsonify(error='No se proporcionó ninguna imagen')

    imagen_file = request.files['imagen']

    imprimir_timestamp("Imagen y bandera obtenidas desde el servidor.")

    # Leer la imagen
    imprimir_timestamp("Convirtiendo imagen.")
    imagen = Image.open(imagen_file)
    imprimir_timestamp("Imagen convertida correctamente.")

    # Detectar objetos
    imprimir_timestamp("Analizando objetos con el modelo de YOLO.")
    detect = modelo_yolo(imagen)
    info = detect.pandas().xyxy[0]
    imprimir_timestamp("Análisis de objetos completado.")

    imprimir_timestamp("Construyendo salida del string de objetos.")
    info_str = construir_salida_yolo(info)
    imprimir_timestamp("Salida con YOLO construida.")

    # Inicialización de texto
    texto = ""

    # Solo detectar texto si el usuario lo quiere
    if is_detectando_texto:
        # Obtener el texto de la imagen
        imprimir_timestamp("Analizando imagen con PyTesseract.")
        texto = pytesseract.image_to_string(imagen)
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

    imprimir_timestamp("Enviando salida a Javascript..")

    return jsonify(resultado=salida)


if __name__ == "__main__":
    app.run(debug=True)
