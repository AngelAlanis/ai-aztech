import os

import cv2
import numpy as np
import torch

from collections import Counter

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.abspath(os.path.dirname(__file__))

# Cargar modelo
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=os.path.join(current_dir, 'aztech.pt'), force_reload=True)

# Cargar imagen
image = cv2.imread(os.path.join(current_dir, 'resources/personas.jpg'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Creación del diccionario

mensajes = {
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

# Realizar detección de objetos
results = model(image)
info = results.pandas().xyxy[0]


def construir_salida_yolo(info):
    # Mostrar las detecciones en la imagen
    lista_labels = info.iloc[:, 6].to_string(index=False)

    print("Lista de labels:", lista_labels)

    labels = lista_labels.split('\n')

    label = labels[0]

    # Se guarda el objeto detectado en una variable de tipo texto
    TObj = label

    print('TObj:', "\'" + TObj + "\'")

    # Obtener el mensaje correspondiente al objeto detectado
    mensaje = mensajes.get(TObj, "No se ha detectado ningún objeto conocido.")

    print(mensaje)

    # Contar cuántos objetos se detectaron
    contador = Counter(labels)
    for elemento, cantidad in contador.items():
        print(f"{cantidad} {elemento}")


if __name__ == "__main__":
    construir_salida_yolo(info)
