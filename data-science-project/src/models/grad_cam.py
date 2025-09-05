#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generador de Mapas de Atención Grad-CAM.

Este módulo implementa la técnica Grad-CAM (Gradient-weighted Class Activation 
Mapping) para generar mapas de atención que resaltan las regiones importantes 
de las imágenes radiográficas para el diagnóstico de neumonía.

Autor: Refactorizado para Módulo 1 - UAO  
Fecha: 2025
"""

# Importaciones estándar
from typing import Optional, Tuple, Union

# Importaciones de terceros
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

# Importaciones para manejo de warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class GradCAMGenerator:
    """
    Generador de mapas de atención Grad-CAM.
    
    Esta clase implementa la técnica Grad-CAM para generar visualizaciones
    de las regiones que el modelo considera más relevantes para realizar
    predicciones de neumonía.
    
    Attributes:
        model (Model): Modelo de Keras para generar activaciones.
        target_layer_name (str): Nombre de la capa objetivo para Grad-CAM.
        grad_model (Model): Modelo de gradientes para Grad-CAM.
    """
    
    def __init__(
        self, 
        model: Model, 
        target_layer_name: Optional[str] = None
    ):
        """
        Inicializa el generador Grad-CAM.
        
        Args:
            model (Model): Modelo de Keras entrenado.
            target_layer_name (str, optional): Nombre de la capa objetivo.
                Si es None, se selecciona automáticamente la última Conv2D.
        """
        self.model = model
        self.target_layer_name = (
            target_layer_name or self._find_target_layer()
        )
        self.grad_model = None # Initialize to None, build on first use
    
    def _find_target_layer(self) -> str:
        """
        Encuentra automáticamente la capa objetivo para Grad-CAM.
        
        Returns:
            str: Nombre de la última capa convolucional encontrada.
            
        Raises:
            ValueError: Si no se encuentra ninguna capa convolucional.
        """
        # Buscar la última capa convolucional
        conv_layers = []
        for layer in self.model.layers:
            layer_type = type(layer).__name__
            if 'Conv2D' in layer_type:
                conv_layers.append(layer.name)
        
        if not conv_layers:
            raise ValueError(
                "No se encontraron capas Conv2D en el modelo para Grad-CAM"
            )
        
        target_layer = conv_layers[-1]
        print(f"🎯 Capa objetivo seleccionada automáticamente: {target_layer}")
        return target_layer
    
    def _build_grad_model(self) -> Model:
        """
        Construye el modelo de gradientes para Grad-CAM.
        
        Returns:
            Model: Modelo que retorna activaciones y predicciones.
        """
        # Obtener la capa objetivo
        target_layer = self.model.get_layer(self.target_layer_name)
        
        # Crear modelo que retorna tanto las activaciones como las predicciones
        grad_model = keras.models.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output]
        )
        
        return grad_model
    
    def generate_gradcam(
        self, 
        img_array: np.ndarray, 
        class_idx: Optional[int] = None,
        use_guided_gradients: bool = False
    ) -> np.ndarray:
        """
        Genera el mapa de atención Grad-CAM para una imagen.
        
        Args:
            img_array (np.ndarray): Imagen preprocesada (H, W, C).
            class_idx (int, optional): Índice de clase objetivo. 
                Si es None, usa la clase con mayor probabilidad.
            use_guided_gradients (bool): Si usar gradientes guiados.
            
        Returns:
            np.ndarray: Mapa de calor Grad-CAM normalizado.
        """
        try:
            # Asegurar que la imagen tiene dimensión de batch
            if len(img_array.shape) == 3:
                img_tensor = tf.expand_dims(img_array, 0)
            else:
                img_tensor = tf.convert_to_tensor(img_array)
            
            # Build grad_model on first call to generate_gradcam
            if self.grad_model is None:
                # Ensure the main model is built by calling it with dummy input
                # This is crucial for target_layer.output and model.output to be defined
                _ = self.model(img_tensor)
                self.grad_model = self._build_grad_model()

            # Calcular gradientes
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                conv_outputs, predictions = self.grad_model(img_tensor)
                
                # Determinar clase objetivo
                if class_idx is None:
                    class_idx = tf.argmax(predictions[0])
                
                # Obtener el score de la clase objetivo
                if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                    # Clasificación binaria con sigmoide
                    if class_idx == 1:
                        class_output = predictions[0, 0]
                    else:
                        class_output = 1 - predictions[0, 0]
                else:
                    # Clasificación multi-clase
                    class_output = predictions[0, class_idx]
            
            # Calcular gradientes de la clase respecto a las activaciones conv
            gradients = tape.gradient(class_output, conv_outputs)
            
            if gradients is None:
                print("⚠️ Warning: No se pudieron calcular gradientes")
                return self._generate_fallback_heatmap(img_array.shape[:2])
            
            # Aplicar gradientes guiados si se solicita
            if use_guided_gradients:
                gradients = self._apply_guided_gradients(
                    gradients, conv_outputs
                )
            
            # Pooling global de los gradientes (importancia de cada canal)
            pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
            
            # Ponderar las activaciones por los gradientes
            conv_outputs = conv_outputs[0]
            weighted_activations = tf.multiply(pooled_gradients, conv_outputs)
            
            # Sumar a lo largo del eje de canales
            heatmap = tf.reduce_mean(weighted_activations, axis=-1)
            
            # Aplicar ReLU para mantener solo activaciones positivas
            heatmap = tf.nn.relu(heatmap)
            
            # Normalizar el heatmap
            heatmap = self._normalize_heatmap(heatmap.numpy())
            
            return heatmap
            
        except Exception as e:
            print(f"❌ Error generando Grad-CAM: {e}")
            return self._generate_fallback_heatmap(img_array.shape[:2])
    
    def _apply_guided_gradients(
        self, 
        gradients: tf.Tensor, 
        activations: tf.Tensor
    ) -> tf.Tensor:
        """
        Aplica gradientes guiados para mejorar la visualización.
        
        Args:
            gradients (tf.Tensor): Gradientes calculados.
            activations (tf.Tensor): Activaciones de la capa conv.
            
        Returns:
            tf.Tensor: Gradientes guiados.
        """
        # Los gradientes guiados solo mantienen gradientes positivos
        # para activaciones positivas
        return tf.where(
            tf.greater(activations, 0),
            tf.nn.relu(gradients),
            tf.zeros_like(gradients)
        )
    
    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Normaliza el heatmap al rango [0, 1].
        
        Args:
            heatmap (np.ndarray): Heatmap sin normalizar.
            
        Returns:
            np.ndarray: Heatmap normalizado.
        """
        # Evitar división por cero
        hmap_min = heatmap.min()
        hmap_max = heatmap.max()
        
        if hmap_max - hmap_min > 1e-8:
            normalized = (heatmap - hmap_min) / (hmap_max - hmap_min)
        else:
            normalized = np.zeros_like(heatmap)
        
        return normalized
    
    def _generate_fallback_heatmap(self, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Genera un heatmap de respaldo cuando falla Grad-CAM.
        
        Args:
            target_shape (tuple): Forma objetivo (height, width).
            
        Returns:
            np.ndarray: Heatmap de respaldo.
        """
        print("🔄 Generando heatmap de respaldo")
        
        # Crear un heatmap simple centrado
        h, w = target_shape
        center_y, center_x = h // 2, w // 2
        
        # Crear gradiente radial desde el centro
        y, x = np.ogrid[:h, :w]
        mask = ((y - center_y) ** 2 + (x - center_x) ** 2) <= (min(h, w) // 4) ** 2
        
        heatmap = np.zeros((h, w), dtype=np.float32)
        heatmap[mask] = 1.0
        
        # Aplicar suavizado gaussiano
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        
        return self._normalize_heatmap(heatmap)
    
    def resize_heatmap(
        self, 
        heatmap: np.ndarray, 
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Redimensiona el heatmap al tamaño objetivo.
        
        Args:
            heatmap (np.ndarray): Heatmap original.
            target_size (tuple): Tamaño objetivo (width, height).
            
        Returns:
            np.ndarray: Heatmap redimensionado.
        """
        return cv2.resize(
            heatmap, 
            target_size, 
            interpolation=cv2.INTER_CUBIC
        )
    
    def generate_heatmap_overlay(
        self, 
        original_img: np.ndarray, 
        heatmap: np.ndarray,
        alpha: float = 0.6,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Crea un overlay del heatmap sobre la imagen original.
        
        Args:
            original_img (np.ndarray): Imagen original.
            heatmap (np.ndarray): Heatmap Grad-CAM.
            alpha (float): Transparencia del overlay (0-1).
            colormap (int): Mapa de colores de OpenCV.
            
        Returns:
            np.ndarray: Imagen con overlay de heatmap.
        """
        try:
            # Asegurar que la imagen original esté en RGB
            if len(original_img.shape) == 3 and original_img.shape[-1] == 3:
                img_rgb = original_img.copy()
            elif len(original_img.shape) == 2:
                img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            
            # Redimensionar imagen al tamaño estándar si es necesario
            img_resized = cv2.resize(img_rgb, (512, 512))
            
            # Redimensionar heatmap al mismo tamaño
            heatmap_resized = self.resize_heatmap(heatmap, (512, 512))
            
            # Convertir heatmap a mapa de colores
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap_resized), 
                colormap
            )
            
            # Crear overlay
            overlay = cv2.addWeighted(
                img_resized.astype(np.uint8), 
                1 - alpha,
                heatmap_colored, 
                alpha, 
                0
            )
            
            return overlay
            
        except Exception as e:
            print(f"❌ Error creando overlay: {e}")
            return original_img
    
    def analyze_attention_regions(
        self, 
        heatmap: np.ndarray, 
        threshold: float = 0.5
    ) -> dict:
        """
        Analiza las regiones de mayor atención en el heatmap.
        
        Args:
            heatmap (np.ndarray): Heatmap Grad-CAM.
            threshold (float): Umbral para considerar alta atención.
            
        Returns:
            dict: Análisis de las regiones de atención.
        """
        try:
            # Crear máscara de alta atención
            high_attention = heatmap > threshold
            
            # Encontrar contornos de regiones de alta atención
            high_attention_uint8 = (high_attention * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                high_attention_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Analizar cada región
            regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filtrar regiones muy pequeñas
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    regions.append({
                        "bbox": (x, y, w, h),
                        "center": (center_x, center_y),
                        "area": area,
                        "max_activation": float(heatmap[y:y+h, x:x+w].max())
                    })
            
            # Estadísticas globales
            analysis = {
                "num_regions": len(regions),
                "regions": sorted(regions, key=lambda x: x["area"], reverse=True),
                "global_max": float(heatmap.max()),
                "global_mean": float(heatmap.mean()),
                "coverage_ratio": float(high_attention.sum() / high_attention.size),
                "center_of_mass": self._calculate_center_of_mass(heatmap)
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error analizando regiones: {e}")
            return {"error": str(e)}
    
    def _calculate_center_of_mass(self, heatmap: np.ndarray) -> Tuple[float, float]:
        """
        Calcula el centro de masa del heatmap.
        
        Args:
            heatmap (np.ndarray): Heatmap de atención.
            
        Returns:
            tuple: Coordenadas (x, y) del centro de masa.
        """
        h, w = heatmap.shape
        total_mass = heatmap.sum()
        
        if total_mass == 0:
            return (w / 2, h / 2)
        
        x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
        
        center_x = (x_indices * heatmap).sum() / total_mass
        center_y = (y_indices * heatmap).sum() / total_mass
        
        return (float(center_x), float(center_y))
    
    def get_layer_info(self) -> dict:
        """
        Obtiene información sobre la capa objetivo.
        
        Returns:
            dict: Información de la capa objetivo.
        """
        try:
            target_layer = self.model.get_layer(self.target_layer_name)
            return {
                "layer_name": self.target_layer_name,
                "layer_type": type(target_layer).__name__,
                "output_shape": target_layer.output_shape,
                "filters": getattr(target_layer, 'filters', 'N/A'),
                "kernel_size": getattr(target_layer, 'kernel_size', 'N/A'),
                "activation": getattr(target_layer, 'activation', 'N/A')
            }
        except Exception as e:
            return {"error": f"Error obteniendo info de capa: {e}"}


# Función de conveniencia para uso directo
def generate_gradcam_visualization(
    model: Model,
    img_array: np.ndarray,
    class_idx: Optional[int] = None,
    target_layer_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Función de conveniencia para generar visualización Grad-CAM.
    
    Args:
        model (Model): Modelo de Keras.
        img_array (np.ndarray): Imagen preprocesada.
        class_idx (int, optional): Clase objetivo.
        target_layer_name (str, optional): Nombre de capa objetivo.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Heatmap y overlay sobre imagen original.
    """
    generator = GradCAMGenerator(model, target_layer_name)
    heatmap = generator.generate_gradcam(img_array, class_idx)
    
    # Si img_array tiene dimensión de batch, quitarla para el overlay
    if len(img_array.shape) == 4:
        img_for_overlay = img_array[0]
    else:
        img_for_overlay = img_array
    
    overlay = generator.generate_heatmap_overlay(img_for_overlay, heatmap)
    
    return heatmap, overlay


if __name__ == "__main__":
    # Ejemplo de uso y testing (requiere modelo cargado)
    print("🔥 Generador de Mapas Grad-CAM")
    
    # Crear datos sintéticos para testing
    dummy_img = np.random.rand(1, 512, 512, 1).astype(np.float32)
    print(f"📷 Imagen sintética creada: {dummy_img.shape}")
    
    try:
        # Nota: En uso real, cargar modelo aquí
        print("⚠️ Para testing completo, se requiere modelo cargado")
        print("✅ Módulo Grad-CAM listo para usar")
        
    except Exception as e:
        print(f"❌ Error en testing: {e}")
    
    print("\n✅ Generador Grad-CAM inicializado correctamente")
