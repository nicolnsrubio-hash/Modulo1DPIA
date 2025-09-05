#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline de Preprocesamiento de Imágenes Radiográficas.

Este módulo implementa el pipeline completo de preprocesamiento de imágenes
radiográficas para el sistema de detección de neumonía, incluyendo resize,
conversión a escala de grises, ecualización CLAHE y normalización.

Autor: Refactorizado para Módulo 1 - UAO
Fecha: 2025
"""

# Importaciones estándar
from typing import Tuple, Optional, Union

# Importaciones de terceros
import numpy as np
import cv2
from tensorflow.keras.utils import img_to_array

# Importaciones para manejo de warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class ImagePreprocessor:
    """
    Pipeline de preprocesamiento de imágenes radiográficas.
    
    Esta clase implementa todas las transformaciones necesarias para preparar
    las imágenes radiográficas antes de ser procesadas por el modelo de
    detección de neumonía.
    
    Attributes:
        target_size (tuple): Tamaño objetivo para resize (width, height).
        clahe_enabled (bool): Si aplicar ecualización CLAHE.
        normalization_range (tuple): Rango de normalización (min, max).
    """
    
    def __init__(
        self, 
        target_size: Tuple[int, int] = (512, 512),
        clahe_enabled: bool = True,
        normalization_range: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Inicializa el preprocesador de imágenes.
        
        Args:
            target_size (tuple): Tamaño objetivo (width, height).
            clahe_enabled (bool): Si aplicar ecualización CLAHE.
            normalization_range (tuple): Rango de normalización (min, max).
        """
        self.target_size = target_size
        self.clahe_enabled = clahe_enabled
        self.normalization_range = normalization_range
        
        # Configurar CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0, 
            tileGridSize=(4, 4)
        )
    
    def preprocess_image(
        self, 
        img_array: np.ndarray,
        apply_batch_format: bool = False
    ) -> np.ndarray:
        """
        Aplica el pipeline completo de preprocesamiento.
        
        Args:
            img_array (np.ndarray): Array de imagen de entrada.
            apply_batch_format (bool): Si agregar dimensión de batch.
            
        Returns:
            np.ndarray: Imagen preprocesada lista para el modelo.
        """
        # Pipeline de preprocesamiento
        processed_img = self._resize_image(img_array)
        processed_img = self._convert_to_grayscale(processed_img)
        
        if self.clahe_enabled:
            processed_img = self._apply_clahe(processed_img)
            
        processed_img = self._normalize_image(processed_img)
        processed_img = self._add_channel_dimension(processed_img)
        
        if apply_batch_format:
            processed_img = self._add_batch_dimension(processed_img)
            
        return processed_img
    
    def _resize_image(self, img_array: np.ndarray) -> np.ndarray:
        """
        Redimensiona la imagen al tamaño objetivo.
        
        Args:
            img_array (np.ndarray): Array de imagen.
            
        Returns:
            np.ndarray: Imagen redimensionada.
        """
        if img_array.shape[:2] != self.target_size[::-1]:  # OpenCV usa (height, width)
            resized = cv2.resize(
                img_array, 
                self.target_size, 
                interpolation=cv2.INTER_AREA
            )
            return resized
        return img_array
    
    def _convert_to_grayscale(self, img_array: np.ndarray) -> np.ndarray:
        """
        Convierte la imagen a escala de grises si es necesario.
        
        Args:
            img_array (np.ndarray): Array de imagen.
            
        Returns:
            np.ndarray: Imagen en escala de grises.
        """
        if len(img_array.shape) == 3 and img_array.shape[-1] == 3:
            # Convertir RGB a escala de grises
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            return gray_img
        elif len(img_array.shape) == 3 and img_array.shape[-1] == 1:
            # Ya está en escala de grises pero con canal extra
            return img_array.squeeze(-1)
        else:
            # Ya está en escala de grises
            return img_array
    
    def _apply_clahe(self, img_array: np.ndarray) -> np.ndarray:
        """
        Aplica ecualización adaptativa del histograma (CLAHE).
        
        Args:
            img_array (np.ndarray): Imagen en escala de grises.
            
        Returns:
            np.ndarray: Imagen con CLAHE aplicado.
        """
        # Asegurar que la imagen está en uint8
        if img_array.dtype != np.uint8:
            # Normalizar a 0-255 y convertir a uint8
            img_normalized = ((img_array - img_array.min()) / 
                            (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        else:
            img_normalized = img_array
        
        # Aplicar CLAHE
        clahe_applied = self.clahe.apply(img_normalized)
        return clahe_applied
    
    def _normalize_image(self, img_array: np.ndarray) -> np.ndarray:
        """
        Normaliza los valores de pixel al rango especificado.
        
        Args:
            img_array (np.ndarray): Array de imagen.
            
        Returns:
            np.ndarray: Imagen normalizada.
        """
        # Convertir a float32 para operaciones numéricas
        img_float = img_array.astype(np.float32)
        
        # Normalizar al rango especificado
        min_val, max_val = self.normalization_range
        
        # Normalizar de [0, 255] a [min_val, max_val]
        normalized = img_float / 255.0
        normalized = normalized * (max_val - min_val) + min_val
        
        return normalized
    
    def _add_channel_dimension(self, img_array: np.ndarray) -> np.ndarray:
        """
        Agrega dimensión de canal si es necesario.
        
        Args:
            img_array (np.ndarray): Array de imagen 2D.
            
        Returns:
            np.ndarray: Array con dimensión de canal (H, W, 1).
        """
        if len(img_array.shape) == 2:
            return np.expand_dims(img_array, axis=-1)
        return img_array
    
    def _add_batch_dimension(self, img_array: np.ndarray) -> np.ndarray:
        """
        Agrega dimensión de batch para inferencia.
        
        Args:
            img_array (np.ndarray): Array de imagen.
            
        Returns:
            np.ndarray: Array con dimensión de batch (1, H, W, C).
        """
        return np.expand_dims(img_array, axis=0)
    
    def preprocess_batch(
        self, 
        img_batch: list
    ) -> np.ndarray:
        """
        Preprocesa un lote de imágenes.
        
        Args:
            img_batch (list): Lista de arrays de imágenes.
            
        Returns:
            np.ndarray: Lote de imágenes preprocesadas.
        """
        processed_batch = []
        
        for img in img_batch:
            processed_img = self.preprocess_image(img, apply_batch_format=False)
            processed_batch.append(processed_img)
        
        return np.array(processed_batch)
    
    def get_preprocessing_params(self) -> dict:
        """
        Obtiene los parámetros de preprocesamiento actuales.
        
        Returns:
            dict: Parámetros de configuración.
        """
        return {
            "target_size": self.target_size,
            "clahe_enabled": self.clahe_enabled,
            "normalization_range": self.normalization_range,
            "clahe_clip_limit": self.clahe.getClipLimit(),
            "clahe_tile_grid_size": self.clahe.getTilesGridSize()
        }
    
    def update_clahe_params(
        self, 
        clip_limit: float = 2.0, 
        tile_grid_size: Tuple[int, int] = (4, 4)
    ):
        """
        Actualiza los parámetros de CLAHE.
        
        Args:
            clip_limit (float): Límite de clip para CLAHE.
            tile_grid_size (tuple): Tamaño de la grilla de tiles.
        """
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
    
    def visualize_preprocessing_steps(
        self, 
        img_array: np.ndarray
    ) -> dict:
        """
        Visualiza cada paso del preprocesamiento para debugging.
        
        Args:
            img_array (np.ndarray): Imagen de entrada.
            
        Returns:
            dict: Diccionario con cada paso del preprocesamiento.
        """
        steps = {
            "original": img_array.copy(),
            "resized": self._resize_image(img_array),
        }
        
        # Conversión a escala de grises
        gray_img = self._convert_to_grayscale(steps["resized"])
        steps["grayscale"] = gray_img
        
        # CLAHE si está habilitado
        if self.clahe_enabled:
            clahe_img = self._apply_clahe(gray_img)
            steps["clahe"] = clahe_img
            normalized_img = self._normalize_image(clahe_img)
        else:
            normalized_img = self._normalize_image(gray_img)
        
        steps["normalized"] = normalized_img
        
        # Formato final con canal y batch
        final_img = self._add_channel_dimension(normalized_img)
        steps["with_channel"] = final_img
        
        final_batch = self._add_batch_dimension(final_img)
        steps["with_batch"] = final_batch
        
        return steps


# Función de conveniencia para compatibilidad con código existente
def preprocess_image(
    img_array: np.ndarray, 
    target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Función de conveniencia para preprocesamiento básico.
    
    Args:
        img_array (np.ndarray): Array de imagen.
        target_size (tuple): Tamaño objetivo.
        
    Returns:
        np.ndarray: Imagen preprocesada.
    """
    preprocessor = ImagePreprocessor(target_size=target_size)
    return preprocessor.preprocess_image(img_array)


def preprocess(img_array: np.ndarray) -> np.ndarray:
    """
    Función de compatibilidad con el código existente.
    
    Args:
        img_array (np.ndarray): Array de imagen.
        
    Returns:
        np.ndarray: Imagen preprocesada con formato batch.
    """
    preprocessor = ImagePreprocessor()
    return preprocessor.preprocess_image(img_array, apply_batch_format=True)


if __name__ == "__main__":
    # Ejemplo de uso y testing
    preprocessor = ImagePreprocessor()
    
    print("🔄 Pipeline de Preprocesamiento de Imágenes")
    print(f"⚙️ Parámetros: {preprocessor.get_preprocessing_params()}")
    
    # Ejemplo con imagen sintética
    test_img = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
    print(f"\n📷 Imagen de prueba: {test_img.shape}")
    
    # Procesar imagen
    processed = preprocessor.preprocess_image(test_img)
    print(f"✅ Imagen procesada: {processed.shape}")
    print(f"📊 Rango de valores: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Procesar con formato batch
    batch_processed = preprocessor.preprocess_image(test_img, apply_batch_format=True)
    print(f"📦 Con batch: {batch_processed.shape}")
    
    # Visualizar pasos (sin mostrar las imágenes, solo info)
    steps = preprocessor.visualize_preprocessing_steps(test_img)
    print(f"\n🔍 Pasos del preprocesamiento:")
    for step_name, step_img in steps.items():
        print(f"   {step_name}: {step_img.shape} | "
              f"tipo: {step_img.dtype} | "
              f"rango: [{step_img.min():.3f}, {step_img.max():.3f}]")
    
    print("\n✅ Pipeline de preprocesamiento listo")
