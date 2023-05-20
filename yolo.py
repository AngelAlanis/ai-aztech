import cv2
import numpy as np
import torch
import os
from collections import Counter

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.abspath(os.path.dirname(__file__))

# Cargar las etiquetas de los objetos
labels_path = os.path.join(current_dir,'aztech.names')
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Cargar modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(current_dir,'aztech.pt'), force_reload=True)

# Cargar imagen
image = cv2.imread(os.path.join(current_dir,'imagen.jpg'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Realizar detección de objetos
results = model(image)
detections = results.xyxy[0].numpy()

# Listas para almacenar los objetos detectados y los porcentajes de confianza
objetos_detectados = []
porcentajes_confianza = []

# Mostrar las detecciones en la imagen
for detection in detections:
    label_idx = int(detection[5])
    label = labels[label_idx]
    confidence = float(detection[4])
    x1, y1, x2, y2 = map(int, detection[:4])
    color = (255, 0, 0)  # Rojo para la detección

    # Dibujar un rectángulo alrededor de la detección
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Mostrar el nombre de la detección
    cv2.putText(image, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    #Se guarda el objeto detectado en una variable de tipo texto
    TObj=label

    # Ejemplo de valor para TObj, puedes cambiarlo según tus necesidades
if TObj == "persona":
    mensaje = "Se ha detectado a una persona."
elif TObj == "bicicleta":
    mensaje = "Se ha detectado una bicicleta."
elif TObj == "carro":
    mensaje = "Se ha detectado un carro."
elif TObj == "motocicleta":
    mensaje = "Se ha detectado una motocicleta."
elif TObj == "avión":
    mensaje = "Se ha detectado un avión."
elif TObj == "autobús":
    mensaje = "Se ha detectado un autobús."
elif TObj == "tren":
    mensaje = "Se ha detectado un tren."
elif TObj == "camion":
    mensaje = "Se ha detectado un camión."
elif TObj == "bote":
    mensaje = "Se ha detectado un bote."
elif TObj == "luz de semáforo":
    mensaje = "Se ha detectado una luz de semáforo."
elif TObj == "hidrante":
    mensaje = "Se ha detectado un hidrante."
elif TObj == "señal de parada":
    mensaje = "Se ha detectado una señal de parada."
elif TObj == "parquímetro":
    mensaje = "Se ha detectado un parquímetro."
elif TObj == "banco":
    mensaje = "Se ha detectado un banco."
elif TObj == "pájaro":
    mensaje = "Se ha detectado un pájaro."
elif TObj == "gato":
    mensaje = "Se ha detectado un gato."
elif TObj == "perro":
    mensaje = "Se ha detectado un perro."
elif TObj == "caballo":
    mensaje = "Se ha detectado un caballo."
elif TObj == "oveja":
    mensaje = "Se ha detectado una oveja."
elif TObj == "baca":
    mensaje = "Se ha detectado una baca."
elif TObj == "elefante":
    mensaje = "Se ha detectado un elefante."
elif TObj == "oso":
    mensaje = "Se ha detectado un oso."
elif TObj == "cebra":
    mensaje = "Se ha detectado una cebra."
elif TObj == "jirafa":
    mensaje = "Se ha detectado una jirafa."
elif TObj == "mochila":
    mensaje = "Se ha detectado una mochila."
elif TObj == "paraguas":
    mensaje = "Se ha detectado un paraguas."
elif TObj == "bolsa de mano":
    mensaje = "Se ha detectado una bolsa de mano."
elif TObj == "corbata":
    mensaje = "Se ha detectado una corbata."
elif TObj == "frisbee":
    mensaje = "Se ha detectado un frisbee."
elif TObj == "esquís":
    mensaje = "Se han detectado esquís."
elif TObj == "snowboard":
    mensaje = "Se ha detectado un snowboard."
elif TObj == "pelota":
    mensaje = "Se ha detectado una pelota."
elif TObj == "cometa":
    mensaje = "Se ha detectado una cometa."
elif TObj == "bate de beisbol":
    mensaje = "Se ha detectado un bate de béisbol."
elif TObj == "guante de beisbol":
    mensaje = "Se ha detectado un guante de béisbol."
elif TObj == "patineta":
    mensaje = "Se ha detectado una patineta."
elif TObj == "tabla de surf":
    mensaje = "Se ha detectado una tabla de surf."
elif TObj == "raqueta de tenis":
    mensaje = "Se ha detectado una raqueta de tenis."
elif TObj == "botella":
    mensaje = "Se ha detectado una botella."
elif TObj == "copa de vino":
    mensaje = "Se ha detectado una copa de vino."
elif TObj == "taza":
    mensaje = "Se ha detectado una taza."
elif TObj == "tenedor":
    mensaje = "Se ha detectado un tenedor."
elif TObj == "cuchillo":
    mensaje = "Se ha detectado un cuchillo."
elif TObj == "cuchara":
    mensaje = "Se ha detectado una cuchara."
elif TObj == "cuenco":
    mensaje = "Se ha detectado un cuenco."
elif TObj == "platano":
    mensaje = "Se ha detectado un plátano."
elif TObj == "manzana":
    mensaje = "Se ha detectado una manzana."
elif TObj == "sandwich":
    mensaje = "Se ha detectado un sandwich."
elif TObj == "naranja":
    mensaje = "Se ha detectado una naranja."
elif TObj == "brocoli":
    mensaje = "Se ha detectado un brócoli."
elif TObj == "zanahoria":
    mensaje = "Se ha detectado una zanahoria."
elif TObj == "perro caliente":
    mensaje = "Se ha detectado un perro caliente."
elif TObj == "pizza":
    mensaje = "Se ha detectado una pizza."
elif TObj == "dona":
    mensaje = "Se ha detectado una dona."
elif TObj == "pastel":
    mensaje = "Se ha detectado un pastel."
elif TObj == "silla":
    mensaje = "Se ha detectado una silla."
elif TObj == "sofa":
    mensaje = "Se ha detectado un sofá."
elif TObj == "planta en maceta":
    mensaje = "Se ha detectado una planta en maceta."
elif TObj == "cama":
    mensaje = "Se ha detectado una cama."
elif TObj == "comedor":
    mensaje = "Se ha detectado un comedor."
elif TObj == "baño":
    mensaje = "Se ha detectado un baño."
elif TObj == "tv":
    mensaje = "Se ha detectado una TV."
elif TObj == "laptop":
    mensaje = "Se ha detectado una laptop."
elif TObj == "mouse":
    mensaje = "Se ha detectado un mouse."
elif TObj == "control remoto":
    mensaje = "Se ha detectado un control remoto."
elif TObj == "teclado":
    mensaje = "Se ha detectado un teclado."
elif TObj == "telefono":
    mensaje = "Se ha detectado un teléfono."
elif TObj == "microondas":
    mensaje = "Se ha detectado un microondas."
elif TObj == "horno":
    mensaje = "Se ha detectado un horno."
elif TObj == "tostadora":
    mensaje = "Se ha detectado una tostadora."
elif TObj == "fregadero":
    mensaje = "Se ha detectado un fregadero."
elif TObj == "refrigerador":
    mensaje = "Se ha detectado un refrigerador."
elif TObj == "libro":
    mensaje = "Se ha detectado un libro."
elif TObj == "reloj":
    mensaje = "Se ha detectado un reloj."
elif TObj == "florero":
    mensaje = "Se ha detectado un florero."
elif TObj == "tijeras":
    mensaje = "Se han detectado tijeras."
elif TObj == "oso de peluche":
    mensaje = "Se ha detectado un oso de peluche."
elif TObj == "secadora de pelo":
    mensaje = "Se ha detectado una secadora de pelo."
elif TObj == "cepillo de dientes":
    mensaje = "Se ha detectado un cepillo de dientes."
else:
    mensaje = "No se ha detectado ningún objeto conocido."

print(mensaje)

    
# Almacenar el objeto detectado y su porcentaje de confianza
 objetos_detectados.append(label)
 porcentajes_confianza.append(round(confidence, 2))

# Imprimir las listas de objetos detectados y porcentajes de confianza
print("Objetos detectados:", objetos_detectados)
print("Porcentajes de confianza:", porcentajes_confianza)

print("")
contador=Counter(objetos_detectados)
for elemento,cantidad in contador.items():
    print(f"{cantidad} {elemento}")

# Mostrar la imagen con las detecciones
cv2.imshow("Detecciones", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
