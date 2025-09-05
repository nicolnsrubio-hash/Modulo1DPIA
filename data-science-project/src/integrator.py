#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrador Principal del Sistema de Detección de Neumonía.

Este módulo coordina todos los componentes del sistema de detección de neumonía,
integrando lectura, preprocesamiento, predicción y visualización.

Autor: Refactorizado para Módulo 1 - UAO
Fecha: 2025
"""

# Importaciones estándar
import os
import sys
from typing import Tuple, Optional, Union
from pathlib import Path

# Importaciones de terceros
import numpy as np
from PIL import Image

# Importaciones locales
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data_science_project.src.data.read_img import ImageReader
from data_science_project.src.data.preprocess_img import ImagePreprocessor
from data_science_project.src.models.load_model import ModelLoader
from data_science_project.src.models.grad_cam import GradCAMGenerator
from data_science_project.src.visualizations.heatmap_overlay import (
    HeatmapOverlay
)


class PneumoniaDetector:
    """
    Detector de Neumonía basado en Deep Learning.
    
    Esta clase integra todos los componentes necesarios para la detección
    de neumonía en imágenes radiográficas, proporcionando una interfaz
    unificada para el sistema completo.
    
    Attributes:
        model_path (str): Ruta al archivo del modelo entrenado.
        model: Modelo de red neuronal cargado.
        image_reader: Instancia para lectura de imágenes.
        preprocessor: Instancia para preprocesamiento.
        grad_cam: Instancia para generación de Grad-CAM.
        heatmap_overlay: Instancia para overlay de heatmaps.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el detector de neumonía.
        
        Args:
            model_path (str, optional): Ruta personalizada al modelo.
                Si es None, usa la ruta por defecto.
        """
        self.model_path = (
            model_path or self._get_default_model_path()
        )
        self.model = None
        
        # Inicializar componentes
        self.image_reader = ImageReader()
        self.preprocessor = ImagePreprocessor()
        self.grad_cam = None
        self.heatmap_overlay = HeatmapOverlay()
        
        # Etiquetas de clasificación
        self.class_labels = {
            0: "bacteriana",
            1: "normal",
            2: "viral"
        }
    
    def _get_default_model_path(self) -> str:
        """
        Obtiene la ruta por defecto del modelo.
        
        Returns:
            str: Ruta por defecto al modelo conv_MLP_84.h5.
        """
        base_dir = Path(__file__).parent.parent.parent
        return str(base_dir / "conv_MLP_84.h5")
    
    def load_model(self) -> bool:
        """
        Carga el modelo de red neuronal.
        
        Returns:
            bool: True si el modelo se cargó exitosamente, False en caso contrario.
        """
        try:
            model_loader = ModelLoader(self.model_path)
            self.model = model_loader.load_model()
            self.grad_cam = GradCAMGenerator(self.model)
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def predict_pneumonia(
        self, 
        image_path: str
    ) -> Tuple[str, float, np.ndarray]:
        """
        Predice neumonía desde archivo de imagen.
        
        Args:
            image_path (str): Ruta a la imagen radiográfica.
            
        Returns:
            Tuple[str, float, np.ndarray]: Tupla con:
                - Clase predicha ("bacteriana", "normal", "viral")
                - Probabilidad de la predicción (0-100)
                - Imagen con heatmap Grad-CAM superpuesto
                
        Raises:
            ValueError: Si el modelo no está cargado.
            FileNotFoundError: Si el archivo de imagen no existe.
        """
        if self.model is None:
            raise ValueError(
                "Modelo no cargado. Ejecute load_model() primero."
            )
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        
        try:
            # 1. Leer imagen
            img_array = self.image_reader.read_image(image_path)
            
            # 2. Preprocesar imagen
            processed_img = self.preprocessor.preprocess_image(img_array)
            
            # 3. Realizar predicción
            prediction_class, probability = self._predict(processed_img)
            
            # 4. Generar Grad-CAM
            heatmap = self.grad_cam.generate_gradcam(
                processed_img, 
                prediction_class
            )
            
            # 5. Crear overlay de heatmap
            heatmap_overlay = self.heatmap_overlay.create_overlay(
                img_array, 
                heatmap
            )
            
            return (
                self.class_labels[prediction_class],
                probability,
                heatmap_overlay
            )
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            # Retornar valores por defecto en caso de error
            fallback_overlay = self._create_fallback_heatmap(
                self.image_reader.read_image(image_path)
            )
            return ("error", 0.0, fallback_overlay)
    
    def predict_from_array(
        self, 
        img_array: np.ndarray
    ) -> Tuple[str, float, np.ndarray]:
        """
        Predice neumonía desde array de imagen.
        
        Args:
            img_array (np.ndarray): Array de imagen ya cargada.
            
        Returns:
            Tuple[str, float, np.ndarray]: Tupla con clase, probabilidad y heatmap.
        """
        if self.model is None:
            raise ValueError(
                "Modelo no cargado. Ejecute load_model() primero."
            )
        
        try:
            # Preprocesar imagen
            processed_img = self.preprocessor.preprocess_image(img_array)
            
            # Realizar predicción
            prediction_class, probability = self._predict(processed_img)
            
            # Generar Grad-CAM
            heatmap = self.grad_cam.generate_gradcam(
                processed_img, 
                prediction_class
            )
            
            # Crear overlay
            heatmap_overlay = self.heatmap_overlay.create_overlay(
                img_array, 
                heatmap
            )
            
            return (
                self.class_labels[prediction_class],
                probability,
                heatmap_overlay
            )
            
        except Exception as e:
            print(f"Error en predicción desde array: {e}")
            fallback_overlay = self._create_fallback_heatmap(img_array)
            return ("error", 0.0, fallback_overlay)
    
    def _predict(
        self, 
        processed_img: np.ndarray
    ) -> Tuple[int, float]:
        """
        Realiza la predicción con el modelo cargado.
        
        Args:
            processed_img (np.ndarray): Imagen preprocesada.
            
        Returns:
            Tuple[int, float]: Clase predicha y probabilidad.
        """
        # Expandir dimensiones para batch
        batch_img = np.expand_dims(processed_img, axis=0)
        
        # Realizar predicción
        model_output = self.model.predict(batch_img, verbose=0)
        
        # Procesar salida según formato del modelo
        if model_output.shape[1] == 1:
            # Clasificación binaria con sigmoid
            proba_value = model_output[0, 0]
            prediction = 1 if proba_value > 0.5 else 0
            probability = max(proba_value, 1 - proba_value) * 100
        else:
            # Clasificación multi-clase
            prediction = np.argmax(model_output)
            probability = np.max(model_output) * 100
        
        return int(prediction), float(probability)
    
    def _create_fallback_heatmap(
        self, 
        original_img: np.ndarray
    ) -> np.ndarray:
        """
        Crea un heatmap de respaldo cuando falla Grad-CAM.
        
        Args:
            original_img (np.ndarray): Imagen original.
            
        Returns:
            np.ndarray: Imagen con overlay básico.
        """
        return self.heatmap_overlay.create_fallback_overlay(original_img)
    
    def get_model_info(self) -> dict:
        """
        Obtiene información del modelo cargado.
        
        Returns:
            dict: Información del modelo (arquitectura, parámetros, etc.).
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        try:
            return {
                "model_path": self.model_path,
                "input_shape": self.model.input_shape,
                "output_shape": self.model.output_shape,
                "total_params": self.model.count_params(),
                "layers": len(self.model.layers),
                "status": "loaded"
            }
        except Exception as e:
            return {"status": f"Error getting info: {e}"}


# Función de conveniencia para uso directo
def detect_pneumonia(
    image_path: str, 
    model_path: Optional[str] = None
) -> Tuple[str, float, np.ndarray]:
    """
    Función de conveniencia para detección rápida de neumonía.
    
    Args:
        image_path (str): Ruta a la imagen radiográfica.
        model_path (str, optional): Ruta personalizada al modelo.
        
    Returns:
        Tuple[str, float, np.ndarray]: Clase, probabilidad y heatmap.
    """
    detector = PneumoniaDetector(model_path)
    
    if not detector.load_model():
        raise RuntimeError("No se pudo cargar el modelo")
    
    return detector.predict_pneumonia(image_path)


if __name__ == "__main__":
    # Ejemplo de uso
    detector = PneumoniaDetector()
    
    if detector.load_model():
        print("✅ Detector de neumonía inicializado correctamente")
        print(f"📊 Información del modelo: {detector.get_model_info()}")
    else:
        print("❌ Error inicializando el detector")
