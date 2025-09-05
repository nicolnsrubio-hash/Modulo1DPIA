#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de Overlay de Heatmaps para Visualización.

Este módulo proporciona funcionalidades para crear overlays de heatmaps sobre
imágenes originales, facilitando la visualización de los resultados de Grad-CAM
y otras técnicas de explicabilidad de IA.

Autor: Refactorizado para Módulo 1 - UAO
Fecha: 2025
"""

# Importaciones estándar
from typing import Tuple, Optional, Union

# Importaciones de terceros
import numpy as np
import cv2
from PIL import Image

# Importaciones para manejo de warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class HeatmapOverlay:
    """
    Generador de overlays de heatmap sobre imágenes.
    
    Esta clase proporciona métodos para crear visualizaciones combinadas
    de imágenes originales con mapas de calor superpuestos, utilizados
    principalmente para mostrar resultados de Grad-CAM.
    
    Attributes:
        default_alpha (float): Transparencia por defecto para overlays.
        default_colormap (int): Mapa de colores por defecto.
        target_size (tuple): Tamaño estándar para las visualizaciones.
    """
    
    def __init__(
        self, 
        default_alpha: float = 0.6,
        default_colormap: int = cv2.COLORMAP_JET,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        Inicializa el generador de overlays.
        
        Args:
            default_alpha (float): Transparencia por defecto (0.0-1.0).
            default_colormap (int): Mapa de colores de OpenCV por defecto.
            target_size (tuple): Tamaño objetivo (width, height).
        """
        self.default_alpha = default_alpha
        self.default_colormap = default_colormap
        self.target_size = target_size
    
    def create_overlay(
        self,
        original_img: np.ndarray,
        heatmap: np.ndarray,
        alpha: Optional[float] = None,
        colormap: Optional[int] = None,
        normalize_heatmap: bool = True
    ) -> np.ndarray:
        """
        Crea un overlay de heatmap sobre la imagen original.
        
        Args:
            original_img (np.ndarray): Imagen original.
            heatmap (np.ndarray): Heatmap a superponer.
            alpha (float, optional): Transparencia del overlay.
            colormap (int, optional): Mapa de colores a usar.
            normalize_heatmap (bool): Si normalizar el heatmap.
            
        Returns:
            np.ndarray: Imagen con overlay de heatmap.
        """
        # Usar valores por defecto si no se especifican
        alpha = alpha if alpha is not None else self.default_alpha
        colormap = colormap if colormap is not None else self.default_colormap
        
        try:
            # Preparar imagen original
            img_prepared = self._prepare_base_image(original_img)
            
            # Preparar heatmap
            heatmap_prepared = self._prepare_heatmap(
                heatmap, 
                img_prepared.shape[:2],
                normalize_heatmap
            )
            
            # Convertir heatmap a colores
            heatmap_colored = self._apply_colormap(heatmap_prepared, colormap)
            
            # Crear overlay
            overlay = self._blend_images(
                img_prepared, 
                heatmap_colored, 
                alpha
            )
            
            return overlay
            
        except Exception as e:
            print(f"❌ Error creando overlay: {e}")
            return self.create_fallback_overlay(original_img)
    
    def _prepare_base_image(self, img: np.ndarray) -> np.ndarray:
        """
        Prepara la imagen base para el overlay.
        
        Args:
            img (np.ndarray): Imagen original.
            
        Returns:
            np.ndarray: Imagen preparada en formato RGB de 3 canales.
        """
        # Convertir a RGB si es necesario
        if len(img.shape) == 2:
            # Escala de grises -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            if img.shape[-1] == 1:
                # Escala de grises con canal -> RGB
                img_rgb = cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2RGB)
            elif img.shape[-1] == 3:
                # Ya está en RGB
                img_rgb = img.copy()
            else:
                # Formato no reconocido, convertir primer canal
                img_rgb = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Formato de imagen no soportado: {img.shape}")
        
        # Redimensionar al tamaño objetivo
        img_resized = cv2.resize(img_rgb, self.target_size)
        
        # Asegurar tipo uint8
        if img_resized.dtype != np.uint8:
            img_resized = self._normalize_to_uint8(img_resized)
        
        return img_resized
    
    def _prepare_heatmap(
        self, 
        heatmap: np.ndarray, 
        target_shape: Tuple[int, int],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Prepara el heatmap para el overlay.
        
        Args:
            heatmap (np.ndarray): Heatmap original.
            target_shape (tuple): Forma objetivo (height, width).
            normalize (bool): Si normalizar valores.
            
        Returns:
            np.ndarray: Heatmap preparado.
        """
        # Asegurar 2D
        if len(heatmap.shape) > 2:
            heatmap = heatmap.squeeze()
        
        # Normalizar si se solicita
        if normalize:
            heatmap = self._normalize_heatmap(heatmap)
        
        # Redimensionar
        heatmap_resized = cv2.resize(
            heatmap, 
            self.target_size, 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Convertir a uint8 en rango 0-255
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        
        return heatmap_uint8
    
    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Normaliza el heatmap al rango [0, 1].
        
        Args:
            heatmap (np.ndarray): Heatmap original.
            
        Returns:
            np.ndarray: Heatmap normalizado.
        """
        hmap_min = heatmap.min()
        hmap_max = heatmap.max()
        
        if hmap_max - hmap_min > 1e-8:
            normalized = (heatmap - hmap_min) / (hmap_max - hmap_min)
        else:
            normalized = np.zeros_like(heatmap)
        
        return normalized.astype(np.float32)
    
    def _normalize_to_uint8(self, img: np.ndarray) -> np.ndarray:
        """
        Normaliza imagen al rango uint8 [0, 255].
        
        Args:
            img (np.ndarray): Imagen en cualquier rango.
            
        Returns:
            np.ndarray: Imagen como uint8.
        """
        img_min = img.min()
        img_max = img.max()
        
        if img_max - img_min > 1e-8:
            normalized = (img - img_min) / (img_max - img_min) * 255
        else:
            normalized = np.zeros_like(img)
        
        return normalized.astype(np.uint8)
    
    def _apply_colormap(
        self, 
        heatmap: np.ndarray, 
        colormap: int
    ) -> np.ndarray:
        """
        Aplica mapa de colores al heatmap.
        
        Args:
            heatmap (np.ndarray): Heatmap en escala de grises.
            colormap (int): Código de mapa de colores de OpenCV.
            
        Returns:
            np.ndarray: Heatmap con colores aplicados.
        """
        try:
            colored = cv2.applyColorMap(heatmap, colormap)
            return colored
        except Exception as e:
            print(f"⚠️ Error aplicando colormap: {e}")
            # Usar colormap por defecto como respaldo
            return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    def _blend_images(
        self, 
        base_img: np.ndarray, 
        overlay_img: np.ndarray, 
        alpha: float
    ) -> np.ndarray:
        """
        Mezcla la imagen base con el overlay.
        
        Args:
            base_img (np.ndarray): Imagen base.
            overlay_img (np.ndarray): Imagen de overlay.
            alpha (float): Factor de mezcla para el overlay.
            
        Returns:
            np.ndarray: Imagen mezclada.
        """
        try:
            # Asegurar mismo tamaño
            if base_img.shape != overlay_img.shape:
                overlay_img = cv2.resize(overlay_img, base_img.shape[:2][::-1])
            
            # Mezclar usando addWeighted
            blended = cv2.addWeighted(
                base_img.astype(np.uint8),
                1.0 - alpha,
                overlay_img.astype(np.uint8),
                alpha,
                0
            )
            
            return blended
            
        except Exception as e:
            print(f"⚠️ Error mezclando imágenes: {e}")
            return base_img
    
    def create_fallback_overlay(self, original_img: np.ndarray) -> np.ndarray:
        """
        Crea un overlay de respaldo cuando fallan otros métodos.
        
        Args:
            original_img (np.ndarray): Imagen original.
            
        Returns:
            np.ndarray: Imagen con overlay de respaldo.
        """
        try:
            # Preparar imagen base
            img_prepared = self._prepare_base_image(original_img)
            
            # Crear heatmap de respaldo simple
            h, w = img_prepared.shape[:2]
            center_y, center_x = h // 2, w // 2
            
            # Crear círculo en el centro
            fallback_heatmap = np.zeros((h, w), dtype=np.float32)
            cv2.circle(
                fallback_heatmap, 
                (center_x, center_y), 
                min(h, w) // 8, 
                1.0, 
                -1
            )
            
            # Aplicar suavizado
            fallback_heatmap = cv2.GaussianBlur(
                fallback_heatmap, 
                (31, 31), 
                0
            )
            
            # Crear overlay con heatmap de respaldo
            heatmap_uint8 = (fallback_heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(
                heatmap_uint8, 
                cv2.COLORMAP_HOT
            )
            
            # Mezclar con menor intensidad
            overlay = cv2.addWeighted(
                img_prepared, 
                0.8, 
                heatmap_colored, 
                0.2, 
                0
            )
            
            return overlay
            
        except Exception as e:
            print(f"❌ Error en overlay de respaldo: {e}")
            # En último caso, retornar imagen original
            return self._prepare_base_image(original_img)
    
    def create_side_by_side_comparison(
        self,
        original_img: np.ndarray,
        heatmap: np.ndarray,
        overlay_img: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Crea visualización lado a lado: original | heatmap | overlay.
        
        Args:
            original_img (np.ndarray): Imagen original.
            heatmap (np.ndarray): Heatmap.
            overlay_img (np.ndarray, optional): Imagen con overlay.
                Si es None, se genera automáticamente.
                
        Returns:
            np.ndarray: Imagen con comparación lado a lado.
        """
        try:
            # Preparar imágenes
            img_prepared = self._prepare_base_image(original_img)
            
            # Crear overlay si no se proporciona
            if overlay_img is None:
                overlay_img = self.create_overlay(original_img, heatmap)
            else:
                overlay_img = self._prepare_base_image(overlay_img)
            
            # Preparar heatmap para visualización
            heatmap_viz = self._prepare_heatmap(heatmap, img_prepared.shape[:2])
            heatmap_colored = cv2.applyColorMap(heatmap_viz, cv2.COLORMAP_JET)
            
            # Concatenar horizontalmente
            comparison = np.hstack([
                img_prepared,
                heatmap_colored,
                overlay_img
            ])
            
            return comparison
            
        except Exception as e:
            print(f"❌ Error creando comparación: {e}")
            return self._prepare_base_image(original_img)
    
    def save_overlay(
        self, 
        overlay_img: np.ndarray, 
        filepath: str, 
        quality: int = 95
    ) -> bool:
        """
        Guarda el overlay en archivo.
        
        Args:
            overlay_img (np.ndarray): Imagen con overlay.
            filepath (str): Ruta del archivo destino.
            quality (int): Calidad de compresión (para JPEG).
            
        Returns:
            bool: True si se guardó exitosamente.
        """
        try:
            # Convertir BGR a RGB para guardado correcto
            if len(overlay_img.shape) == 3:
                overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
            else:
                overlay_bgr = overlay_img
            
            # Guardar con calidad especificada
            if filepath.lower().endswith(('.jpg', '.jpeg')):
                success = cv2.imwrite(
                    filepath, 
                    overlay_bgr, 
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
            else:
                success = cv2.imwrite(filepath, overlay_bgr)
            
            if success:
                print(f"✅ Overlay guardado en: {filepath}")
            else:
                print(f"❌ Error guardando overlay en: {filepath}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error guardando overlay: {e}")
            return False
    
    def get_overlay_stats(
        self, 
        original_img: np.ndarray, 
        heatmap: np.ndarray
    ) -> dict:
        """
        Calcula estadísticas del overlay.
        
        Args:
            original_img (np.ndarray): Imagen original.
            heatmap (np.ndarray): Heatmap.
            
        Returns:
            dict: Estadísticas del overlay.
        """
        try:
            # Preparar datos
            img_prepared = self._prepare_base_image(original_img)
            heatmap_prepared = self._prepare_heatmap(heatmap, img_prepared.shape[:2])
            
            # Calcular estadísticas
            stats = {
                "original_shape": original_img.shape,
                "prepared_shape": img_prepared.shape,
                "heatmap_shape": heatmap.shape,
                "heatmap_prepared_shape": heatmap_prepared.shape,
                "heatmap_min": float(heatmap.min()),
                "heatmap_max": float(heatmap.max()),
                "heatmap_mean": float(heatmap.mean()),
                "heatmap_std": float(heatmap.std()),
                "non_zero_ratio": float(np.count_nonzero(heatmap) / heatmap.size),
                "target_size": self.target_size,
                "default_alpha": self.default_alpha
            }
            
            return stats
            
        except Exception as e:
            return {"error": f"Error calculando estadísticas: {e}"}


# Funciones de conveniencia
def create_heatmap_overlay(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Función de conveniencia para crear overlay rápido.
    
    Args:
        original_img (np.ndarray): Imagen original.
        heatmap (np.ndarray): Heatmap.
        alpha (float): Transparencia.
        
    Returns:
        np.ndarray: Imagen con overlay.
    """
    overlay_generator = HeatmapOverlay(default_alpha=alpha)
    return overlay_generator.create_overlay(original_img, heatmap)


if __name__ == "__main__":
    # Ejemplo de uso y testing
    print("🎨 Generador de Overlays de Heatmap")
    
    # Crear datos sintéticos para testing
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_heatmap = np.random.rand(64, 64).astype(np.float32)
    
    print(f"📷 Imagen de prueba: {test_img.shape}")
    print(f"🔥 Heatmap de prueba: {test_heatmap.shape}")
    
    # Crear generador
    overlay_gen = HeatmapOverlay()
    
    # Generar overlay
    overlay = overlay_gen.create_overlay(test_img, test_heatmap)
    print(f"✅ Overlay creado: {overlay.shape}")
    
    # Crear comparación lado a lado
    comparison = overlay_gen.create_side_by_side_comparison(
        test_img, 
        test_heatmap
    )
    print(f"📊 Comparación creada: {comparison.shape}")
    
    # Obtener estadísticas
    stats = overlay_gen.get_overlay_stats(test_img, test_heatmap)
    print(f"📈 Estadísticas: {len(stats)} campos calculados")
    
    print("\n✅ Generador de overlays listo para usar")
