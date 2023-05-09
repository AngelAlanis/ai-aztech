import onnx
import tensorflow as tf
import tf2onnx
import os

current_dir = os.path.abspath(os.path.dirname(__file__))

# Ruta al archivo ONNX
#onnx_model_p = onnx.load(os.path.join(current_dir, 'model.onnx'))
onnx_model_path = "C:/Users/jose_/Documents/imgyolo/model.onnx"

onnx_model = tf2onnx.utils.load_model(onnx_model_path)

saved_model_dir = "C:/Users/jose_/Documents/imgyolo"

# Convertir el modelo ONNX a formato SavedModel
tf2onnx.tfonnx.process.onnx_tf.export_model(onnx_model, saved_model_dir, tf2onnx.utils.build_opset_dict(None))

# Convertir el modelo ONNX a formato TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Guardar el modelo TFLite en un archivo
tflite_model_path = os.path.join(current_dir,'model.tflite') 
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
