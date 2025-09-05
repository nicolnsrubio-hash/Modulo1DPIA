#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de Lectura de Imágenes Radiográficas.

Este script se encarga de la lectura de imágenes médicas en diferentes formatos,
especialmente DICOM, para su posterior procesamiento en el sistema de detección
de neumonía.

Autor: Refactorizado para Módulo 1 - UAO
Fecha: 2025
"""

# Importaciones estándar
import os
from typing import Tuple, Optional, Union
from pathlib import Path

# Importaciones de terceros
import numpy as np
import cv2
import pydicom
from PIL import Image

# Importaciones para manejo de warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class ImageReader:
    """
    Clase para lectura de imágenes médicas en múltiples formatos.
    
    Esta clase proporciona métodos para leer imágenes radiográficas en formatos
    DICOM, JPEG, PNG y otros formatos comunes, normalizando la salida para su
    uso posterior en el pipeline de procesamiento.
    
    Attributes:
        supported_formats (tuple): Formatos de imagen soportados.
    """
    
    def __init__(self):
        """Inicializa el lector de imágenes."""
        self.supported_formats = (
            '.dcm', '.dicom', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'
        )
    
    def read_image(self, image_path: str) -> np.ndarray:
        """
        Lee una imagen desde archivo y la convierte a array numpy.
        
        Args:
            image_path (str): Ruta al archivo de imagen.
            
        Returns:
            np.ndarray: Array de imagen en formato RGB o escala de grises.
            
        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si el formato no es soportado.
            RuntimeError: Si hay error al leer la imagen.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Archivo de imagen no encontrado: {image_path}"
            )
        
        # Obtener extensión del archivo
        file_extension = Path(image_path).suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(
                f"Formato no soportado: {file_extension}. "
                f"Formatos soportados: {self.supported_formats}"
            )
        
        try:
            # Leer según el formato
            if file_extension in ['.dcm', '.dicom']:
                return self._read_dicom_file(image_path)
            else:
                return self._read_standard_image(image_path)
                
        except Exception as e:
            raise RuntimeError(f"Error leyendo imagen {image_path}: {e}")
    
    def _read_dicom_file(self, dicom_path: str) -> np.ndarray:
        """
        Lee un archivo DICOM y lo convierte a array RGB.
        
        Args:
            dicom_path (str): Ruta al archivo DICOM.
            
        Returns:
            np.ndarray: Array de imagen en formato RGB.
        """
        # Leer archivo DICOM
        ds = pydicom.dcmread(dicom_path)
        img_array = ds.pixel_array
        
        # Normalizar valores de pixel
        img_normalized = self._normalize_pixel_values(img_array)
        
        # Convertir a RGB si es escala de grises
        if len(img_normalized.shape) == 2:
            img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_normalized
            
        return img_rgb
    
    def _read_standard_image(self, image_path: str) -> np.ndarray:
        """
        Lee una imagen estándar (JPEG, PNG, etc.).
        
        Args:
            image_path (str): Ruta a la imagen.
            
        Returns:
            np.ndarray: Array de imagen en formato RGB.
        """
        # Leer imagen con OpenCV
        img_array = cv2.imread(image_path)
        
        if img_array is None:
            raise RuntimeError(f"No se pudo leer la imagen: {image_path}")
        
        # Convertir BGR a RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Normalizar si es necesario
        if img_rgb.dtype != np.uint8:
            img_rgb = self._normalize_pixel_values(img_rgb)
            
        return img_rgb
    
    def _normalize_pixel_values(self, img_array: np.ndarray) -> np.ndarray:
        """
        Normaliza los valores de pixel a rango 0-255.
        
        Args:
            img_array (np.ndarray): Array de imagen.
            
        Returns:
            np.ndarray: Array normalizado como uint8.
        """
        # Convertir a float para operaciones
        img_float = img_array.astype(float)
        
        # Normalizar al rango 0-255
        img_min = np.min(img_float)
        img_max = np.max(img_float)
        
        if img_max > img_min:
            img_normalized = (img_float - img_min) / (img_max - img_min) * 255.0
        else:
            img_normalized = img_float
        
        # Convertir a uint8
        return img_normalized.astype(np.uint8)
    
    def read_image_for_display(
        self, 
        image_path: str, 
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Lee imagen y la prepara para visualización en interfaz gráfica.
        
        Args:
            image_path (str): Ruta a la imagen.
            target_size (tuple, optional): Tamaño objetivo (width, height).
                Si es None, mantiene el tamaño original.
                
        Returns:
            Tuple[np.ndarray, Image.Image]: Array de imagen y objeto PIL 
            para mostrar en GUI.
        """
        # Leer imagen como array
        img_array = self.read_image(image_path)
        
        # Crear objeto PIL para visualización
        img_pil = Image.fromarray(img_array)
        
        # Redimensionar si se especifica
        if target_size is not None:
            img_pil = img_pil.resize(target_size, Image.Resampling.LANCZOS)
            # También redimensionar el array
            img_array = cv2.resize(img_array, target_size)
        
        return img_array, img_pil
    
    def get_image_info(self, image_path: str) -> dict:
        """
        Obtiene información de la imagen sin cargarla completamente.
        
        Args:
            image_path (str): Ruta a la imagen.
            
        Returns:
            dict: Información de la imagen (dimensiones, formato, etc.).
        """
        if not os.path.exists(image_path):
            return {"error": "Archivo no encontrado"}
        
        try:
            file_extension = Path(image_path).suffix.lower()
            file_size = os.path.getsize(image_path)
            
            info = {
                "path": image_path,
                "format": file_extension,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2)
            }
            
            # Información específica según formato
            if file_extension in ['.dcm', '.dicom']:
                info.update(self._get_dicom_info(image_path))
            else:
                info.update(self._get_standard_image_info(image_path))
            
            return info
            
        except Exception as e:
            return {"error": f"Error obteniendo información: {e}"}
    
    def _get_dicom_info(self, dicom_path: str) -> dict:
        """
        Obtiene información específica de archivo DICOM.
        
        Args:
            dicom_path (str): Ruta al archivo DICOM.
            
        Returns:
            dict: Información específica del DICOM.
        """
        try:
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            
            return {
                "width": getattr(ds, 'Columns', 'Unknown'),
                "height": getattr(ds, 'Rows', 'Unknown'),
                "patient_name": str(getattr(ds, 'PatientName', 'Unknown')),
                "study_date": str(getattr(ds, 'StudyDate', 'Unknown')),
                "modality": str(getattr(ds, 'Modality', 'Unknown')),
                "bits_allocated": getattr(ds, 'BitsAllocated', 'Unknown')
            }
        except Exception as e:
            return {"dicom_error": f"Error leyendo DICOM: {e}"}
    
    def _get_standard_image_info(self, image_path: str) -> dict:
        """
        Obtiene información de imagen estándar.
        
        Args:
            image_path (str): Ruta a la imagen.
            
        Returns:
            dict: Información de la imagen.
        """
        try:
            with Image.open(image_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "channels": len(img.getbands()) if hasattr(img, 'getbands') else 1
                }
        except Exception as e:
            return {"image_error": f"Error leyendo imagen: {e}"}
    
    def is_supported_format(self, file_path: str) -> bool:
        """
        Verifica si el formato de archivo es soportado.
        
        Args:
            file_path (str): Ruta al archivo.
            
        Returns:
            bool: True si el formato es soportado.
        """
        file_extension = Path(file_path).suffix.lower()
        return file_extension in self.supported_formats


# Funciones de conveniencia para compatibilidad con código existente
def read_dicom_file(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Función de conveniencia para leer archivos DICOM (compatibilidad).
    
    Args:
        path (str): Ruta al archivo DICOM.
        
    Returns:
        Tuple[np.ndarray, Image.Image]: Array RGB y objeto PIL.
    """
    reader = ImageReader()
    return reader.read_image_for_display(path)


def read_jpg_file(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Función de conveniencia para leer archivos JPG (compatibilidad).
    
    Args:
        path (str): Ruta al archivo JPG.
        
    Returns:
        Tuple[np.ndarray, Image.Image]: Array RGB y objeto PIL.
    """
    reader = ImageReader()
    return reader.read_image_for_display(path)


if __name__ == "__main__":
    # Ejemplo de uso y testing
    reader = ImageReader()
    
    print("🔍 Lector de Imágenes Radiográficas")
    print(f"📋 Formatos soportados: {reader.supported_formats}")
    
    # Ejemplo con archivo de prueba
    test_files = [
        "ejemplo.dcm",
        "ejemplo.jpg", 
        "ejemplo.png"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n📁 Procesando: {test_file}")
            info = reader.get_image_info(test_file)
            print(f"ℹ️ Información: {info}")
            
            try:
                img_array = reader.read_image(test_file)
                print(f"✅ Imagen cargada: {img_array.shape}")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"⏭️ Saltando {test_file} (no existe)")
    
    print("\n✅ Módulo de lectura de imágenes listo")
