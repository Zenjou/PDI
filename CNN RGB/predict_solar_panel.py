import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import sys
import os
import warnings
import logging

# Suprimir todos los warnings de Python
warnings.filterwarnings("ignore")

# Redirigir la salida estándar y de error para suprimir cualquier mensaje
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Restaurar la salida estándar para mensajes importantes
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Configuración de TensorFlow para evitar logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Configuración de la imagen
img_height = 244
img_width = 244

# Ruta al modelo guardado
model_path = "solar_panel_cnn_model.keras"

# Cargar el modelo entrenado
def load_trained_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: El modelo no se encuentra en la ruta especificada: {model_path}")
        sys.exit(1)
    return load_model(model_path)

# Preprocesar la imagen para el modelo
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: La imagen no se encuentra en la ruta especificada: {image_path}")
        sys.exit(1)
    
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Crear batch con tamaño 1
    img_array = preprocess_input(img_array)
    return img_array

# Realizar la predicción
def predict_image(model, image_path, class_names):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    print(f"Predicción: {predicted_class} (Confianza: {confidence:.2f}%)")
    return predicted_class, confidence

# Función principal
def main():
    if len(sys.argv) < 2:
        print("Uso: python predict_solar_panel.py <ruta_de_la_imagen>")
        sys.exit(1)

    image_path = sys.argv[1]
    model = load_trained_model(model_path)

    # Clases del dataset
    class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

    # Realizar predicción
    predicted_class, confidence = predict_image(model, image_path, class_names)

    # Obtener solo el nombre del archivo
    image_name = os.path.basename(image_path)

    # Escribir el resultado en el archivo "data.txt"
    with open("data.txt", "a") as file:
        file.write(f"{image_name}; {predicted_class} ({confidence:.2f}%)\n")

    # Mostrar la imagen con la predicción
    img = tf.keras.utils.load_img(image_path)
    plt.imshow(img)
    plt.title(f"Predicción: {predicted_class} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
