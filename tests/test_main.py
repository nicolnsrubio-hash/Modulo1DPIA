import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import cargar_imagen, predecir_neumonia, BASE_DIR




def test_cargar_imagen():
    ruta_imagen = os.path.join(BASE_DIR, "tests", "ejemplo_neumonia.jpeg")
    img = cargar_imagen(ruta_imagen)
    assert img.shape == (1, 150, 150, 3)  # Debe devolver tensor 4D con batch

def test_prediccion():
    ruta_imagen = os.path.join(BASE_DIR, "tests", "ejemplo_neumonia.jpeg")
    resultado = predecir_neumonia(ruta_imagen)
    assert resultado in ["Neumonía detectada", "Radiografía normal"]
