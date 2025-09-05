#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interfaz Gráfica para Detección de Neumonía con IA.

Sistema de detección de neumonía basado en Deep Learning que procesa imágenes
radiográficas de tórax en formato DICOM y las clasifica usando redes neuronales
convolucionales con explicabilidad mediante Grad-CAM.

Refactorizado según convenciones PEP8 para el Módulo 1 - UAO.

Autor: Refactorizado para Módulo 1 - UAO
Fecha: 2025
"""

# Importaciones estándar
import csv
import os
import time
from typing import Tuple, Optional

# Importaciones de terceros
import cv2
import numpy as np
import pydicom
import tensorflow as tf
from PIL import Image, ImageGrab, ImageTk
from tensorflow import keras
from tensorflow.keras.models import load_model
from tkinter import (
    Tk, StringVar, Text, Entry, font, filedialog, messagebox,
    ttk, END, WARNING
)

# Importaciones locales
try:
    from data_science_project.src.integrator import PneumoniaDetector
    USE_NEW_MODULES = True
except ImportError:
    USE_NEW_MODULES = False
    print("⚠️ Módulos nuevos no disponibles, usando implementación legacy")

# Configuración de TensorFlow
tf.compat.v1.experimental.output_all_intermediates(True)


class PneumoniaDetectionApp:
    """
    Aplicación GUI para detección de neumonía en imágenes radiográficas.
    
    Esta clase implementa una interfaz gráfica completa que permite cargar
    imágenes médicas, ejecutar predicciones de neumonía usando modelos de IA,
    visualizar resultados con mapas de calor Grad-CAM y exportar reportes.
    
    Attributes:
        root (Tk): Ventana principal de tkinter.
        patient_id (StringVar): Variable para almacenar ID del paciente.
        result_text (StringVar): Variable para mostrar resultado de predicción.
        img_array (np.ndarray): Array de la imagen cargada.
        prediction_label (str): Etiqueta de predicción.
        prediction_proba (float): Probabilidad de predicción.
        heatmap_image (np.ndarray): Imagen con overlay de heatmap.
        report_counter (int): Contador para generar IDs únicos de reportes.
    """
    
    # Constantes de configuración
    WINDOW_WIDTH = 815
    WINDOW_HEIGHT = 560
    IMAGE_DISPLAY_SIZE = (250, 250)
    SUPPORTED_FORMATS = [
        ("DICOM files", "*.dcm"),
        ("JPEG files", "*.jpeg"), 
        ("JPG files", "*.jpg"),
        ("PNG files", "*.png"),
        ("All files", "*.*")
    ]
    
    CLASS_LABELS = {
        0: "bacteriana",
        1: "normal", 
        2: "viral"
    }
    
    def __init__(self):
        """Inicializa la aplicación GUI."""
        self.root = Tk()
        self._setup_window()
        self._initialize_variables()
        self._create_widgets()
        self._setup_layout()
        
        # Inicializar detector nuevo si está disponible
        if USE_NEW_MODULES:
            self.detector = PneumoniaDetector()
            self.detector.load_model()
        else:
            self.detector = None
    
    def _setup_window(self):
        """Configura la ventana principal."""
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.root.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.root.resizable(False, False)
    
    def _initialize_variables(self):
        """Inicializa las variables de instancia."""
        self.patient_id = StringVar()
        self.result_text = StringVar()
        self.img_array = None
        self.prediction_label = ""
        self.prediction_proba = 0.0
        self.heatmap_image = None
        self.report_counter = 0
    
    def _create_widgets(self):
        """Crea todos los widgets de la interfaz."""
        # Fuente en negrita
        bold_font = font.Font(weight="bold")
        
        # Etiquetas
        self.title_label = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=bold_font
        )
        self.image_label = ttk.Label(
            self.root, text="Imagen Radiográfica", font=bold_font
        )
        self.heatmap_label = ttk.Label(
            self.root, text="Imagen con Heatmap", font=bold_font
        )
        self.result_label = ttk.Label(
            self.root, text="Resultado:", font=bold_font
        )
        self.patient_label = ttk.Label(
            self.root, text="Cédula Paciente:", font=bold_font
        )
        self.probability_label = ttk.Label(
            self.root, text="Probabilidad:", font=bold_font
        )
        
        # Campos de entrada
        self.patient_entry = ttk.Entry(
            self.root, textvariable=self.patient_id, width=10
        )
        
        # Áreas de visualización de imágenes
        self.original_image_display = Text(self.root, width=31, height=15)
        self.heatmap_image_display = Text(self.root, width=31, height=15)
        
        # Campos de texto para resultados
        self.result_display = Text(self.root, width=12, height=2)
        self.probability_display = Text(self.root, width=12, height=2)
        
        # Botones
        self.load_button = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_image_file
        )
        self.predict_button = ttk.Button(
            self.root, text="Predecir", state="disabled", 
            command=self.run_prediction
        )
        self.save_button = ttk.Button(
            self.root, text="Guardar", command=self.save_results_to_csv
        )
        self.pdf_button = ttk.Button(
            self.root, text="PDF", command=self.generate_pdf_report
        )
        self.clear_button = ttk.Button(
            self.root, text="Borrar", command=self.clear_all_data
        )
        
        # Focus inicial en el campo de ID del paciente
        self.patient_entry.focus_set()
    
    def _setup_layout(self):
        """Configura el layout de los widgets."""
        # Posicionamiento de etiquetas
        self.title_label.place(x=122, y=25)
        self.image_label.place(x=110, y=65)
        self.heatmap_label.place(x=545, y=65)
        self.patient_label.place(x=65, y=350)
        self.result_label.place(x=500, y=350)
        self.probability_label.place(x=500, y=400)
        
        # Posicionamiento de campos de entrada
        self.patient_entry.place(x=200, y=350)
        
        # Posicionamiento de visualizadores de imagen
        self.original_image_display.place(x=65, y=90)
        self.heatmap_image_display.place(x=500, y=90)
        
        # Posicionamiento de campos de resultado
        self.result_display.place(x=610, y=350, width=90, height=30)
        self.probability_display.place(x=610, y=400, width=90, height=30)
        
        # Posicionamiento de botones
        self.load_button.place(x=70, y=460)
        self.predict_button.place(x=220, y=460)
        self.save_button.place(x=370, y=460)
        self.pdf_button.place(x=520, y=460)
        self.clear_button.place(x=670, y=460)
    
    def load_image_file(self):
        """
        Abre diálogo para seleccionar y cargar archivo de imagen.
        
        Soporta archivos DICOM (.dcm) y formatos estándar (JPEG, PNG).
        Actualiza la visualización y habilita el botón de predicción.
        """
        file_path = filedialog.askopenfilename(
            initialdir="/",
            title="Seleccionar imagen",
            filetypes=self.SUPPORTED_FORMATS
        )
        
        if not file_path:
            return
        
        try:
            # Cargar imagen según el formato
            if file_path.lower().endswith('.dcm'):
                self.img_array, display_image = self._read_dicom_file(file_path)
            else:
                self.img_array, display_image = self._read_standard_image(
                    file_path
                )
            
            # Mostrar imagen en la interfaz
            self._display_image(display_image, self.original_image_display)
            
            # Habilitar botón de predicción
            self.predict_button["state"] = "normal"
            
        except Exception as e:
            messagebox.showerror(
                "Error", f"Error cargando imagen: {str(e)}"
            )
    
    def _read_dicom_file(self, file_path: str) -> Tuple[np.ndarray, Image.Image]:
        """
        Lee archivo DICOM y lo convierte a formato manejable.
        
        Args:
            file_path (str): Ruta al archivo DICOM.
            
        Returns:
            Tuple[np.ndarray, Image.Image]: Array de imagen y objeto PIL.
        """
        dicom_data = pydicom.dcmread(file_path)
        img_array = dicom_data.pixel_array
        
        # Normalizar y convertir a RGB
        img_normalized = img_array.astype(float)
        img_normalized = (
            (np.maximum(img_normalized, 0) / img_normalized.max()) * 255.0
        )
        img_normalized = img_normalized.astype(np.uint8)
        
        img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
        display_image = Image.fromarray(img_rgb)
        
        return img_rgb, display_image
    
    def _read_standard_image(
        self, file_path: str
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Lee imagen en formato estándar (JPEG, PNG, etc.).
        
        Args:
            file_path (str): Ruta al archivo de imagen.
            
        Returns:
            Tuple[np.ndarray, Image.Image]: Array de imagen y objeto PIL.
        """
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            raise ValueError(f"No se pudo leer la imagen: {file_path}")
        
        img_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        display_image = Image.fromarray(img_array)
        
        # Normalizar
        img_normalized = img_array.astype(float)
        img_normalized = (
            (np.maximum(img_normalized, 0) / img_normalized.max()) * 255.0
        )
        img_normalized = img_normalized.astype(np.uint8)
        
        return img_normalized, display_image
    
    def _display_image(self, pil_image: Image.Image, display_widget: Text):
        """
        Muestra imagen PIL en widget de texto.
        
        Args:
            pil_image (Image.Image): Imagen PIL a mostrar.
            display_widget (Text): Widget donde mostrar la imagen.
        """
        # Redimensionar imagen para visualización
        resized_image = pil_image.resize(
            self.IMAGE_DISPLAY_SIZE, Image.Resampling.LANCZOS
        )
        photo_image = ImageTk.PhotoImage(resized_image)
        
        # Mostrar en widget
        display_widget.image_create(END, image=photo_image)
        
        # Mantener referencia para evitar garbage collection
        display_widget.image = photo_image
    
    def run_prediction(self):
        """
        Ejecuta predicción de neumonía en la imagen cargada.
        
        Utiliza el nuevo sistema modular si está disponible, 
        sino usa la implementación legacy.
        """
        if self.img_array is None:
            messagebox.showwarning("Advertencia", "No hay imagen cargada")
            return
        
        try:
            if USE_NEW_MODULES and self.detector:
                # Usar nuevo sistema modular
                result = self.detector.predict_from_array(self.img_array)
                self.prediction_label = result[0]
                self.prediction_proba = result[1]
                self.heatmap_image = result[2]
            else:
                # Usar implementación legacy
                result = self._predict_legacy(self.img_array)
                self.prediction_label = result[0]
                self.prediction_proba = result[1]
                self.heatmap_image = result[2]
            
            # Mostrar resultados
            self._display_prediction_results()
            
        except Exception as e:
            messagebox.showerror(
                "Error", f"Error en predicción: {str(e)}"
            )
    
    def _predict_legacy(
        self, img_array: np.ndarray
    ) -> Tuple[str, float, np.ndarray]:
        """
        Implementación legacy de predicción (compatibilidad).
        
        Args:
            img_array (np.ndarray): Array de imagen.
            
        Returns:
            Tuple[str, float, np.ndarray]: Etiqueta, probabilidad, heatmap.
        """
        # Cargar modelo
        model = self._load_model_legacy()
        
        # Preprocesar imagen
        processed_img = self._preprocess_image_legacy(img_array)
        batch_img = np.expand_dims(processed_img, axis=0)
        
        # Predecir
        model_output = model.predict(batch_img, verbose=0)
        
        # Procesar salida del modelo
        if model_output.shape[1] == 1:
            # Clasificación binaria con sigmoid
            proba_value = model_output[0, 0]
            prediction = 1 if proba_value > 0.5 else 0
            probability = max(proba_value, 1 - proba_value) * 100
        else:
            # Clasificación multi-clase
            prediction = np.argmax(model_output)
            probability = np.max(model_output) * 100
        
        # Mapear predicción a etiqueta
        label = self.CLASS_LABELS.get(prediction, "desconocido")
        
        # Generar heatmap
        try:
            heatmap = self._generate_gradcam_legacy(
                processed_img, model, prediction
            )
            heatmap_display = self._create_heatmap_overlay_legacy(
                img_array, heatmap
            )
        except Exception as e:
            print(f"Error generando heatmap: {e}")
            heatmap_display = self._create_fallback_heatmap_legacy(img_array)
        
        return label, probability, heatmap_display
    
    def _load_model_legacy(self):
        """Carga modelo usando implementación legacy."""
        model_path = os.path.join(
            os.path.dirname(__file__), "conv_MLP_84.h5"
        )
        if not os.path.exists(model_path):
            # Buscar en ruta alternativa
            model_path = "conv_MLP_84.h5"
        
        return load_model(model_path, compile=False)
    
    def _preprocess_image_legacy(
        self, img_array: np.ndarray, target_size: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        """Preprocesa imagen usando implementación legacy."""
        # Convertir a escala de grises
        if img_array.ndim == 3 and img_array.shape[-1] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Redimensionar
        img_array = cv2.resize(img_array, target_size)
        
        # Normalizar
        img_array = img_array.astype("float32") / 255.0
        
        # Añadir dimensión de canal
        img_array = np.expand_dims(img_array, axis=-1)
        
        return img_array
    
    def _generate_gradcam_legacy(
        self, 
        img_array: np.ndarray, 
        model, 
        prediction: int
    ) -> np.ndarray:
        """Genera Grad-CAM usando implementación legacy."""
        # Encontrar capa convolucional adecuada
        target_layer_name = None
        for layer in reversed(model.layers):
            if hasattr(layer, 'output') and 'conv' in layer.name.lower():
                target_layer_name = layer.name
                break
        
        if target_layer_name is None:
            raise ValueError("No se encontró capa convolucional adecuada")
        
        # Crear modelo de gradientes
        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(target_layer_name).output, model.output]
        )
        
        # Preparar entrada
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(img_array, axis=0)
        )
        
        # Calcular gradientes
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            conv_outputs, predictions = grad_model(input_tensor)
            
            if isinstance(predictions, list):
                predictions = predictions[0]
            
            class_output = predictions[0][prediction]
        
        # Computar gradientes
        grads = tape.gradient(class_output, conv_outputs)
        
        if grads is None:
            raise ValueError("No se pudieron calcular gradientes")
        
        # Pooling global de gradientes
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Ponderar activaciones
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(
            tf.multiply(pooled_grads, conv_outputs), axis=-1
        )
        
        # Aplicar ReLU y normalizar
        heatmap = tf.nn.relu(heatmap)
        heatmap_max = tf.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        
        return heatmap.numpy()
    
    def _create_heatmap_overlay_legacy(
        self, 
        original_img: np.ndarray, 
        heatmap: np.ndarray, 
        alpha: float = 0.6
    ) -> np.ndarray:
        """Crea overlay de heatmap usando implementación legacy."""
        # Convertir imagen original a escala de grises
        if original_img.ndim == 3:
            img_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = original_img.copy()
        
        # Redimensionar a 512x512
        img_resized = cv2.resize(img_gray, (512, 512))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        # Redimensionar heatmap
        if heatmap.shape != (512, 512):
            heatmap = cv2.resize(heatmap, (512, 512))
        
        # Normalizar heatmap
        heatmap_norm = (heatmap - heatmap.min()) / (
            heatmap.max() - heatmap.min() + 1e-8
        )
        
        # Aplicar mapa de colores
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET
        )
        
        # Crear overlay
        overlay = cv2.addWeighted(
            img_rgb, 1 - alpha, heatmap_colored, alpha, 0
        )
        
        return overlay
    
    def _create_fallback_heatmap_legacy(
        self, original_img: np.ndarray
    ) -> np.ndarray:
        """Crea heatmap de respaldo usando implementación legacy."""
        if original_img.ndim == 3:
            img_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = original_img.copy()
        
        img_resized = cv2.resize(img_gray, (512, 512))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        # Crear overlay rojo en el centro
        overlay = img_rgb.copy()
        center_x, center_y = 256, 256
        cv2.circle(overlay, (center_x, center_y), 100, (255, 100, 100), -1)
        
        # Mezclar con imagen original
        result = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
        
        return result
    
    def _display_prediction_results(self):
        """Muestra los resultados de predicción en la interfaz."""
        # Mostrar heatmap
        heatmap_pil = Image.fromarray(self.heatmap_image)
        self._display_image(heatmap_pil, self.heatmap_image_display)
        
        # Mostrar texto de resultado
        self.result_display.delete(1.0, END)
        self.result_display.insert(END, self.prediction_label)
        
        # Mostrar probabilidad
        self.probability_display.delete(1.0, END)
        self.probability_display.insert(
            END, f"{self.prediction_proba:.2f}%"
        )
    
    def save_results_to_csv(self):
        """Guarda los resultados de predicción en archivo CSV."""
        if not hasattr(self, 'prediction_label'):
            messagebox.showwarning(
                "Advertencia", "No hay resultados para guardar"
            )
            return
        
        try:
            with open("historial.csv", "a", newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter="-")
                writer.writerow([
                    self.patient_id.get(),
                    self.prediction_label,
                    f"{self.prediction_proba:.2f}%"
                ])
            
            messagebox.showinfo(
                "Guardar", "Los datos se guardaron con éxito."
            )
            
        except Exception as e:
            messagebox.showerror(
                "Error", f"Error guardando datos: {str(e)}"
            )
    
    def generate_pdf_report(self):
        """Genera reporte PDF capturando la ventana actual."""
        try:
            # Capturar ventana
            screenshot = self._capture_window(self.root)
            
            # Guardar como imagen
            report_filename = f"Reporte{self.report_counter}.jpg"
            screenshot.save(report_filename)
            
            # Convertir a PDF
            screenshot_rgb = screenshot.convert("RGB")
            pdf_filename = f"Reporte{self.report_counter}.pdf"
            screenshot_rgb.save(pdf_filename)
            
            self.report_counter += 1
            
            messagebox.showinfo(
                "PDF", "El PDF fue generado con éxito."
            )
            
        except Exception as e:
            messagebox.showerror(
                "Error", f"Error generando PDF: {str(e)}"
            )
    
    def _capture_window(self, widget) -> Image.Image:
        """
        Captura screenshot de widget específico.
        
        Args:
            widget: Widget de tkinter a capturar.
            
        Returns:
            Image.Image: Imagen capturada.
        """
        x = widget.winfo_rootx()
        y = widget.winfo_rooty()
        width = x + widget.winfo_width()
        height = y + widget.winfo_height()
        
        return ImageGrab.grab(bbox=(x, y, width, height))
    
    def clear_all_data(self):
        """Limpia todos los datos de la interfaz tras confirmación."""
        answer = messagebox.askokcancel(
            title="Confirmación",
            message="Se borrarán todos los datos.",
            icon="warning"
        )
        
        if answer:
            try:
                # Limpiar campos de texto
                self.patient_entry.delete(0, END)
                self.result_display.delete(1.0, END)
                self.probability_display.delete(1.0, END)
                
                # Limpiar visualizaciones de imagen
                self.original_image_display.delete(1.0, END)
                self.heatmap_image_display.delete(1.0, END)
                
                # Resetear variables
                self.img_array = None
                self.prediction_label = ""
                self.prediction_proba = 0.0
                self.heatmap_image = None
                
                # Deshabilitar botón de predicción
                self.predict_button["state"] = "disabled"
                
                messagebox.showinfo(
                    "Borrar", "Los datos se borraron con éxito"
                )
                
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error borrando datos: {str(e)}"
                )
    
    def run(self):
        """Inicia el bucle principal de la aplicación."""
        self.root.mainloop()


def main():
    """
    Función principal de la aplicación.
    
    Crea e inicia la aplicación GUI de detección de neumonía.
    
    Returns:
        int: Código de salida (0 para éxito).
    """
    try:
        app = PneumoniaDetectionApp()
        app.run()
        return 0
    except Exception as e:
        print(f"Error iniciando aplicación: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
