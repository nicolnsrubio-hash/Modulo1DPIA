#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paquete de procesamiento de datos para imágenes médicas.

Este paquete contiene módulos para leer, cargar y preprocesar imágenes
médicas en formato DICOM y formatos estándares (PNG, JPG) para
su uso en modelos de machine learning.

Módulos:
    read_img: Lectura de archivos de imagen DICOM y estándares.
    preprocess_img: Preprocesamiento de imágenes para modelos ML.
    
Clases principales:
    DicomImageReader: Lector especializado de imágenes DICOM.
    ImagePreprocessor: Preprocesador de imágenes médicas.
    
Funciones utilitarias:
    read_dicom_file: Lee archivo DICOM desde disco.
    read_image_file: Lee imagen estándar desde disco.
    preprocess_for_model: Pipeline completo de preprocesamiento.
    
Autor: Universidad Autónoma de Occidente - Módulo 1
Fecha: 2025
"""

try:
    from .read_img import (
        read_dicom_file,
        read_image_file,
        read_dicom_from_bytes,
        read_image_from_bytes,
        DicomImageReader
    )
    from .preprocess_img import (
        resize_image,
        convert_to_grayscale,
        apply_clahe,
        normalize_image,
        preprocess_for_model,
        ImagePreprocessor
    )
except ImportError as e:
    print(f"Warning: Could not import all data modules: {e}")
    # Definir funciones dummy para evitar errores
    def read_dicom_file(*args, **kwargs):
        raise NotImplementedError("read_dicom_file not available")
    
    def preprocess_for_model(*args, **kwargs):
        raise NotImplementedError("preprocess_for_model not available")

__version__ = "1.0.0"
__author__ = "Universidad Autónoma de Occidente"

__all__ = [
    'read_dicom_file',
    'read_image_file',
    'read_dicom_from_bytes',
    'read_image_from_bytes',
    'DicomImageReader',
    'resize_image',
    'convert_to_grayscale',
    'apply_clahe',
    'normalize_image',
    'preprocess_for_model',
    'ImagePreprocessor'
]
