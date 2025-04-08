import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from huggingface_hub import upload_file

# Cargar los datos MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocesar los datos
X_train = X_train / 255.0  # Normalizar las imágenes
X_test = X_test / 255.0

y_train = to_categorical(y_train, 10)  # Convertir las etiquetas en vectores one-hot
y_test = to_categorical(y_test, 10)

# Crear el modelo secuencial
model = Sequential()

# Añadir una capa de entrada que aplana las imágenes
model.add(Flatten(input_shape=(28, 28)))

# Añadir una capa densa con 128 neuronas y activación ReLU
model.add(Dense(128, activation='relu'))

# Capa de salida con 10 neuronas (una por cada dígito) y activación softmax
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer=optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Guardar el modelo como archivo .h5
model_path = 'modelo.h5'
model.save(model_path)

# Evaluar el modelo en los datos de prueba
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Subir el modelo a Hugging Face Model Hub
def upload_to_huggingface(model_path):
    # Obtener el token desde las variables de entorno
    token = os.getenv('HF_TOKEN')  # Lee el token de la variable de entorno 'HF_TOKEN'

    print("HF_TOKEN existe:", "HF_TOKEN" in os.environ)
    print("HF_TOKEN length:", len(os.getenv("HF_TOKEN", "")))

    if token is None:
        print("Error: No se encontró el token de Hugging Face en las variables de entorno.")
        return
    else:
        print("TOKEN LENGTH:", len(token))

    # Subir el archivo .h5 al Hugging Face Model Hub
    repo_id = "albertlnz/test-jupyter-ubuntu"  # Cambia esto por el nombre de tu repositorio en Hugging Face

    # Subir el archivo al repositorio en Hugging Face!
    try:
        upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_path,
            repo_id=repo_id,
            token=token
        )
        print(f"Modelo {model_path} guardado correctamente en Hugging Face.")
    except Exception as e:
        print(f"Error al subir el modelo a Hugging Face: {e}")

# Llamar a la función para subir el archivo al repositorio de Hugging Face
upload_to_huggingface(model_path)
