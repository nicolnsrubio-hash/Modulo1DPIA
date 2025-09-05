#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paquete de visualización para detección de neumonía.

Este paquete contiene módulos para crear visualizaciones de los resultados
del modelo, incluyendo overlays de heatmaps y otras técnicas de explicabilidad.

Autor: Refactorizado para Módulo 1 - UAO
Fecha: 2025
"""

from .heatmap_overlay import HeatmapOverlay, create_heatmap_overlay

__all__ = [
    "HeatmapOverlay",
    "create_heatmap_overlay"
]
