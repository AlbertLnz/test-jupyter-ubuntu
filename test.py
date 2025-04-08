import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers

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

# Función para subir el modelo a GitHub Releases
def upload_to_github_release(model_path):
    # Configurar la autenticación usando el token de GitHub
    token = os.getenv('GITHUB_TOKEN')  # Lee el token de la variable de entorno 'GITHUB_TOKEN'

    if token is None:
        print("Error: No se encontró el token de GitHub en las variables de entorno.")
        return

    # Usar 'softprops/action-gh-release' para crear un release en GitHub y subir el archivo
    try:
        import subprocess
        subprocess.run([
            "gh", "release", "create", "v1.0", model_path,
            "--title", "Modelo entrenado",
            "--notes", "Modelo MNIST entrenado con Keras"
        ], check=True)
        print(f"Modelo {model_path} guardado correctamente en GitHub Releases.")
    except Exception as e:
        print(f"Error al subir el modelo a GitHub Releases: {e}")

# Llamar a la función para subir el archivo al release de GitHub
upload_to_github_release(model_path)
