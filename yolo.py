import os
from collections import Counter

import cv2
import numpy as np
import torch

from aztech_utils import objetos_plural, objetos_singular

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.abspath(os.path.dirname(__file__))

# Cargar modelo
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=os.path.join(current_dir, 'aztech.pt'), force_reload=True)

# Cargar imagen
image = cv2.imread(os.path.join(current_dir, 'resources/imagen.jpg'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Realizar detección de objetos
results = model(image)
info = results.pandas().xyxy[0]


def construir_salida_yolo(info):
    # Mostrar las detecciones en la imagen
    lista_labels = info.iloc[:, 6].to_string(index=False)

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
    print(resultado)


if __name__ == "__main__":
    construir_salida_yolo(info)
