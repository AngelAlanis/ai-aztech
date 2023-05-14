import pytesseract
import cv2
import numpy as np
import time

# Configuración de la ruta de PyTesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configuración de la fuente para mostrar en ventana
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# Inicializar la captura del vídeo con la entrada 0
cap = cv2.VideoCapture(0)

# Si la entrada 0 no funciona, intentar con la 1.
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("No se puede abrir el vídeo.")

# Iniciar contador para la tasa de refresco.
counter = 0
while True:
    ret, frame = cap.read()

    counter += 1
    if (counter % 2) == 0:

        imgH, imgW, _ = frame.shape

        x1, y1, w1, h1 = 0, 0, imgH, imgW

        # Obtener el texto de la imagen
        imgchar = pytesseract.image_to_string(frame)
        # Obtener las boxes de los caracteres
        imgboxes = pytesseract.image_to_boxes(frame)

        # Dibujar el rectángulo en la ventana
        for boxes in imgboxes.splitlines():
            boxes = boxes.split(' ')
            x, y, w, h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
            cv2.rectangle(frame, (x, imgH - y), (w, imgH - h), (0, 0, 255), 3)

        # Poner en pantalla el texto obtenido 
        cv2.putText(frame, imgchar, (x1 + int(w1 / 50), y1 + int(h1 / 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Mostrar la imagen en la ventana
        cv2.imshow('PyTesseract', frame)

        # Condición de terminación
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()