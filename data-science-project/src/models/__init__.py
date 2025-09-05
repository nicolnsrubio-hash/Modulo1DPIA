#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paquete de modelos de IA para detección de neumonía.

Este paquete contiene módulos para carga de modelos de machine learning
y generación de mapas de explicabilidad usando técnicas como Grad-CAM.

Autor: Refactorizado para Módulo 1 - UAO
Fecha: 2025
"""

from .load_model import ModelLoader, load_pneumonia_model, get_default_model_path
from .grad_cam import GradCAMGenerator, generate_gradcam_visualization

__all__ = [
    "ModelLoader",
    "load_pneumonia_model",
    "get_default_model_path", 
    "GradCAMGenerator",
    "generate_gradcam_visualization"
]
