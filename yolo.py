import os
from collections import Counter

import cv2
import numpy as np
import torch

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.abspath(os.path.dirname(__file__))

# Cargar modelo
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=os.path.join(current_dir, 'aztech.pt'), force_reload=True)

# Cargar imagen
image = cv2.imread(os.path.join(current_dir, 'resources/imagen.jpg'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Cargar diccionarios
objetos_singular = {
    "persona": "persona",
    "bicicleta": "bicicleta",
    "carro": "carro",
    "motocicleta": "motocicleta",
    "avion": "avión",
    "autobus": "autobús",
    "tren": "tren",
    "camion": "camión",
    "barco": "barco",
    "semaforo": "semáforo",
    "hidrante": "hidrante",
    "señal de alto": "señal de alto",
    "parquimetro": "parquímetro",
    "banca": "banco",
    "pajaro": "pájaro",
    "gato": "gato",
    "perro": "perro",
    "caballo": "caballo",
    "oveja": "oveja",
    "vaca": "vaca",
    "elefante": "elefante",
    "oso": "oso",
    "cebra": "cebra",
    "jirafa": "jirafa",
    "mochila": "mochila",
    "paraguas": "paraguas",
    "bolsa de mano": "bolsa de mano",
    "corbata": "corbata",
    "maletin": "maletín",
    "frisbee": "frisbee",
    "esquis": "esquís",
    "tabla de nieve": "tabla de nieve",
    "pelota": "pelota",
    "cometa": "cometa",
    "bate de baseball": "bate de béisbol",
    "guante de baseball": "guante de béisbol",
    "patineta": "patineta",
    "tabla de surf": "tabla de surf",
    "raqueta de tenis": "raqueta de tenis",
    "botella": "botella",
    "copa de vino": "copa de vino",
    "taza": "taza",
    "tenedor": "tenedor",
    "cuchillo": "cuchillo",
    "cuchara": "cuchara",
    "tazon": "tazón",
    "platano": "plátano",
    "manzana": "manzana",
    "sandwich": "sandwich",
    "naranja": "naranja",
    "brocoli": "brócoli",
    "zanahoria": "zanahoria",
    "hot dog": "perro caliente",
    "pizza": "pizza",
    "dona": "dona",
    "pastel": "pastel",
    "silla": "silla",
    "sillon": "sofá",
    "planta en maceta": "planta en maceta",
    "cama": "cama",
    "comedor": "comedor",
    "inodoro": "baño",
    "tv": "TV",
    "laptop": "laptop",
    "mouse": "mouse",
    "control remoto": "control remoto",
    "teclado": "teclado",
    "celular": "teléfono",
    "microondas": "microondas",
    "horno": "horno",
    "tostador": "tostadora",
    "fregadero": "fregadero",
    "refrigerador": "refrigerador",
    "libro": "libro",
    "reloj": "reloj",
    "florero": "florero",
    "tijeras": "tijeras",
    "osito de peluche": "oso de peluche",
    "secador de pelo": "secador de pelo",
    "cepillo de dientes": "cepillo de dientes",
    "moneda50": "moneda de 50 centavos",
    "moneda1": "moneda de 1 peso",
    "moneda2": "moneda de 2 pesos",
    "moneda5": "moneda de 5 pesos",
    "moneda10": "moneda de 10 pesos",
    "billete20": "billete de 20 pesos",
    "billete50": "billete de 50 pesos",
    "billete100": "billete de 100 pesos",
    "billete200": "billete de 200 pesos",
    "billete500": "billete de 500 pesos",
    "billete1000": "billete de 1000 pesos"
}

objetos_plural = {
    "persona": "personas",
    "bicicleta": "bicicletas",
    "carro": "carros",
    "motocicleta": "motocicletas",
    "avion": "aviones",
    "autobus": "autobuses",
    "tren": "trenes",
    "camion": "camiones",
    "barco": "barcos",
    "semaforo": "semáforos",
    "hidrante": "hidrantes",
    "señal de alto": "señales de alto",
    "parquimetro": "parquímetros",
    "banca": "bancas",
    "pajaro": "pájaros",
    "gato": "gatos",
    "perro": "perros",
    "caballo": "caballos",
    "oveja": "ovejas",
    "vaca": "vacas",
    "elefante": "elefantes",
    "oso": "osos",
    "cebra": "cebras",
    "jirafa": "jirafas",
    "mochila": "mochilas",
    "paraguas": "paraguas",
    "bolsa de mano": "bolsas de mano",
    "corbata": "corbatas",
    "maletin": "maletines",
    "frisbee": "frisbees",
    "esquis": "esquís",
    "tabla de nieve": "tablas de nieve",
    "pelota": "pelotas",
    "cometa": "cometas",
    "bate de baseball": "bates de béisbol",
    "guante de baseball": "guantes de béisbol",
    "patineta": "patinetas",
    "tabla de surf": "tablas de surf",
    "raqueta de tenis": "raquetas de tenis",
    "botella": "botellas",
    "copa de vino": "copas de vino",
    "taza": "tazas",
    "tenedor": "tenedores",
    "cuchillo": "cuchillos",
    "cuchara": "cucharas",
    "tazon": "tazones",
    "platano": "plátanos",
    "manzana": "manzanas",
    "sandwich": "sandwiches",
    "naranja": "naranjas",
    "brocoli": "brócolis",
    "zanahoria": "zanahorias",
    "hot dog": "hot dogs",
    "pizza": "pizzas",
    "dona": "donas",
    "pastel": "pasteles",
    "silla": "sillas",
    "sillon": "sillones",
    "planta en maceta": "plantas en maceta",
    "cama": "camas",
    "comedor": "comedores",
    "inodoro": "inodoros",
    "tv": "TVs",
    "laptop": "laptops",
    "mouse": "mouses",
    "control remoto": "controles remotos",
    "teclado": "teclados",
    "celular": "celulares",
    "microondas": "microondas",
    "horno": "hornos",
    "tostador": "tostadores",
    "fregadero": "fregaderos",
    "refrigerador": "refrigeradores",
    "libro": "libros",
    "reloj": "relojes",
    "florero": "floreros",
    "tijeras": "tijeras",
    "osito de peluche": "osos de peluche",
    "secador de pelo": "secadores de pelo",
    "cepillo de dientes": "cepillos de dientes",
    "moneda50": "monedas de 50 centavos",
    "moneda1": "monedas de 1 peso",
    "moneda2": "monedas de 2 pesos",
    "moneda5": "monedas de 5 pesos",
    "moneda10": "monedas de 10 pesos",
    "billete20": "billetes de 20 pesos",
    "billete50": "billetes de 50 pesos",
    "billete100": "billetes de 100 pesos",
    "billete200": "billetes de 200 pesos",
    "billete500": "billetes de 500 pesos",
    "billete1000": "billetes de 1000 pesos"
}



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
            elemento += "s"  # Agregar 's' al final para hacerlo plural

        objetos_detectados.append(f"{cantidad} {elemento}")

    # Construir el mensaje final
    resultado += " " + " y ".join(objetos_detectados)

    # Imprimir el resultado final
    print(resultado)


if __name__ == "__main__":
    construir_salida_yolo(info)
