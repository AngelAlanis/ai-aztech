import cv2
import numpy as np
import torch
import os

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.abspath(os.path.dirname(__file__))

# Cargar las etiquetas de los objetos
labels_path = os.path.join(current_dir,'coco.names')
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Cargar modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(current_dir,'prueba.pt'), force_reload=True)

# Cargar imagen
image = cv2.imread(os.path.join(current_dir,'imagen.jpg'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Realizar detección de objetos
results = model(image)
detections = results.xyxy[0].numpy()

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

# Mostrar la imagen con las detecciones
cv2.imshow("Detecciones", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
