import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Ruta base del proyecto
BASE_DIR = r"D:\UAO\DESARROLLO DE PROYECTOS DE INTELIGENCIA ARTIFICIAL\Neumonia\UAO-Neumonia"

# Ruta del modelo
MODEL_PATH = os.path.join(BASE_DIR, "modelo", "modelo_neumonia.h5")

# Cargar el modelo solo si existe
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None
    print("⚠️ No se encontró el modelo en:", MODEL_PATH)

def cargar_imagen(ruta_imagen, mostrar=False):
    """Carga y preprocesa una radiografía"""
    img = image.load_img(ruta_imagen, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if mostrar:
        plt.imshow(image.load_img(ruta_imagen), cmap="gray")
        plt.title("Radiografía cargada")
        plt.axis("off")
        plt.show()

    return img_array

def predecir_neumonia(ruta_imagen):
    """Predice neumonía en la imagen cargada"""
    if model is None:
        return "Modelo no disponible"
    
    img_array = cargar_imagen(ruta_imagen)
    prediccion = model.predict(img_array)
    return "Neumonía detectada" if prediccion[0][0] > 0.5 else "Radiografía normal"

def main():
    print("=== Sistema de Detección de Neumonía ===")
    ruta_test = os.path.join(BASE_DIR, "tests", "ejemplo_neumonia.jpeg")

    if not os.path.exists(ruta_test):
        print(f"No se encontró la imagen de prueba en: {ruta_test}")
        return

    resultado = predecir_neumonia(ruta_test)
    print(f"Resultado de la predicción: {resultado}")

if __name__ == "__main__":
    main()

