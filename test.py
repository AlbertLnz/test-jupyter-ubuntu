import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
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

# Enviar el archivo por correo electrónico
def enviar_correo(email_destino, archivo_adjunto):
    # Configuración del servidor SMTP
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    from_email = 'tu_correo@gmail.com'  # Tu correo de envío
    from_password = 'tu_contraseña'  # Tu contraseña o una App Password si usas Gmail

    # Crear el mensaje
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = email_destino
    msg['Subject'] = 'Modelo entrenado - archivo .h5'

    # Adjuntar el archivo
    with open(archivo_adjunto, 'rb') as file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(archivo_adjunto)}')
        msg.attach(part)

    # Conectar con el servidor SMTP y enviar el mensaje
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Activar seguridad
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, email_destino, text)
        server.quit()
        print("Correo enviado exitosamente.")
    except Exception as e:
        print(f"Error al enviar el correo: {e}")

# Llamar a la función para enviar el correo con el archivo adjunto
enviar_correo('wanir65649@buides.com', model_path)
