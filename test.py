import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
import smtplib
from email.message import EmailMessage

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

def enviar_modelo_por_correo(model_path, destinatario, remitente, clave_app):
    # Crear el mensaje
    msg = EmailMessage()
    msg['Subject'] = 'Modelo MNIST Entrenado'
    msg['From'] = remitente
    msg['To'] = destinatario
    msg.set_content('Adjunto el modelo entrenado en formato .h5.')

    # Leer y adjuntar el archivo del modelo
    with open(model_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(model_path)
        msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

    # Enviar el correo (aquí se usa SMTP de Gmail como ejemplo)!
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(remitente, clave_app)
            smtp.send_message(msg)
        print("Correo enviado correctamente.")
    except Exception as e:
        print(f"Error al enviar el correo: {e}")

remitente = 'albert2000.lanza@gmail.com'
destinatario = 'albert.lnz.rio@gmail.com'
clave_app = os.getenv('GOOGLE_API_KEY_GMAIL')
enviar_modelo_por_correo(model_path, destinatario, remitente, clave_app)
