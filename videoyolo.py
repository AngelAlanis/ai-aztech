import cv2
import numpy as np
import torch
import os

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.abspath(os.path.dirname(__file__))

# Cargar modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(current_dir,'aztech.pt'))

#Realiza la videocaptura
cap=cv2.VideoCapture(0)

#Ciclo
while True:
    #Lectura de frames
    ret, frame=cap.read()

    #Detección
    detect=model(frame)

    info=detect.pandas().xyxy[0]
    print(info)

    #Mostrar FPS
    cv2.imshow('Detector de objetos', np.squeeze(detect.render()))

    #Lee el teclado
    t=cv2.waitKey(5)
    if t==27: #Tecla Esc
        break

cap.release()
cv2.destroyAllWindows()
