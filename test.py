import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
import subprocess

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

# Guardar el modelo en el repositorio de GitHub
def commit_and_push(model_path):
    # Asegurarse de que el archivo .h5 esté en el directorio correcto
    os.rename(model_path, f'./{model_path}')

    # Hacer git commit y push
    try:
        # Añadir el archivo al staging de git
        subprocess.run(['git', 'add', model_path], check=True)

        # Hacer el commit
        subprocess.run(['git', 'commit', '-m', 'Añadir modelo entrenado'], check=True)

        # Hacer push a la rama actual
        subprocess.run(['git', 'push'], check=True)

        print(f"Modelo {model_path} guardado correctamente en el repositorio.")
    except subprocess.CalledProcessError as e:
        print(f"Error al hacer git commit y push: {e}")

# Llamar a la función para hacer commit y push del archivo
commit_and_push(model_path)
