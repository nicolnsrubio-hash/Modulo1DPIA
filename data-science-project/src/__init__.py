#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paquete principal del código fuente para detección de neumonía con IA.

Este paquete contiene todos los módulos y subpaquetes necesarios para
el sistema de detección de neumonía usando técnicas de deep learning
y visualización explicable con Grad-CAM.

Subpaquetes:
    data: Módulos para lectura y preprocesamiento de imágenes médicas.
    models: Módulos para carga de modelos ML y generación de Grad-CAM.
    visualizations: Módulos para visualización de resultados.
    
Módulos principales:
    integrator: Interface principal del sistema (PneumoniaDetector).
    
Autor: Universidad Autónoma de Occidente - Módulo 1
Fecha: 2025
"""

try:
    from .integrator import PneumoniaDetector, detect_pneumonia
except ImportError:
    # Definir clases dummy para evitar errores en pruebas
    class PneumoniaDetector:
        pass
    
    def detect_pneumonia(*args, **kwargs):
        raise NotImplementedError("detect_pneumonia not available")

__version__ = "1.0.0"
__author__ = "Universidad Autónoma de Occidente"

# Exportar clases y funciones principales
__all__ = [
    'PneumoniaDetector',
    'detect_pneumonia'
]
