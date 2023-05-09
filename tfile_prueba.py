import cv2
import numpy as np
import os
import tensorflow as tf

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.abspath(os.path.dirname(__file__))

# Cargar las etiquetas de los objetos
labels_path = os.path.join(current_dir, 'tlite.txt')
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Cargar modelo TFLite
model_path = os.path.join(current_dir, 'aztecho.tflite')

# Cargar el archivo TFLite utilizando la biblioteca tflite_runtime
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cargar imagen

# Cargar imagen
image = cv2.imread(os.path.join(current_dir, 'imagen.jpg'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Redimensionar la imagenS
target_size = (416, 416)
image = cv2.resize(image, target_size)

# Preprocesar la imagen
input_shape = interpreter.get_input_details()[0]['shape']
input_data = np.expand_dims(image, axis=0)
input_data = input_data.astype(np.uint8)

# Establecer los datos de entrada del modelo TFLite
interpreter.set_tensor(input_details[0]['index'], input_data)

# Realizar la inferencia
interpreter.invoke()

# Obtener los resultados de la inferencia
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
#print(output_data.shape)

# Mostrar las detecciones en la imagen
for detection in output_data:
    label_idx = np.argmax(detection[4:])  # Obtener el índice de la etiqueta con mayor confianza

    # Verificar si el índice está dentro del rango de los datos de detección
    if label_idx < detection.shape[0] - 4:
        confidence = float(detection[label_idx + 4])  # El índice se desplaza en 4 debido a los valores previos en la detección

        # Verificar si el índice está dentro del rango de etiquetas
        if label_idx < len(labels):
            label = labels[label_idx]

            x1, y1, x2, y2 = map(int, detection[:4])  # Obtener las coordenadas del rectángulo

            # Dibujar un rectángulo alrededor de la detección
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            print(f"Etiqueta: {label}, Confianza: {confidence:.2f}, Coordenadas: ({x1}, {y1}) - ({x2}, {y2})")

            # Mostrar el nombre de la detección
            cv2.putText(image, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# Mostrar la imagen con las detecciones
cv2.imshow("Detecciones", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
