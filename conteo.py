#Importación de OpenCV
import cv2
#Procesamiento de imágenes
imagen = cv2.imread('sc4.png')
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
bordes = cv2.Canny(grises, 100, 200)

#ENcontrar y contar contornos en la imagen
ctns, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagen, ctns, -1, (0,0,255), 2)
print('Número de contornos encontrados: ', len(ctns))
texto = 'Contornos encontrados: '+ str(len(ctns))
cv2.putText(imagen, texto, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
  (255, 0, 0), 1)
cv2.imshow('Imagen', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()