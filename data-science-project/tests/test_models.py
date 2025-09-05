#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pruebas unitarias para módulos de modelos y visualización.

Este archivo contiene pruebas para validar el funcionamiento correcto de
los módulos de carga de modelos, Grad-CAM y visualización de heatmaps.

Autor: Refactorizado para Módulo 1 - UAO
Fecha: 2025
"""

# Importaciones estándar
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Importaciones de terceros
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Importaciones locales
try:
    from src.models.load_model import (
        ModelLoader, 
        get_default_model_path
    )
    from src.models.grad_cam import GradCAMGenerator
    from src.visualizations.heatmap_overlay import HeatmapOverlay
except ImportError:
    import sys
    import os
    # Agregar el directorio padre al path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.models.load_model import (
        ModelLoader, 
        get_default_model_path
    )
    from src.models.grad_cam import GradCAMGenerator
    from src.visualizations.heatmap_overlay import HeatmapOverlay


class TestModelLoader:
    """Pruebas para la clase ModelLoader."""
    
    @pytest.fixture
    def mock_model(self):
        """Fixture que crea un modelo de prueba simple."""
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(512, 512, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model
    
    @pytest.fixture
    def temp_model_file(self, mock_model):
        """Fixture que crea un archivo temporal de modelo."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            mock_model.save(tmp.name)
            yield tmp.name
        # Limpiar archivo temporal
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    def test_initialization_with_valid_path(self, temp_model_file):
        """Prueba la inicialización con ruta válida."""
        loader = ModelLoader(temp_model_file)
        assert loader.model_path == os.path.abspath(temp_model_file)
        assert loader.model is None
        assert isinstance(loader.model_info, dict)
    
    def test_initialization_with_invalid_path(self):
        """Prueba el manejo de rutas inválidas."""
        with pytest.raises(FileNotFoundError):
            ModelLoader("archivo_inexistente.h5")
    
    def test_initialization_with_invalid_extension(self):
        """Prueba el manejo de extensiones inválidas."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            with pytest.raises(ValueError, match="Extensión de modelo no válida"):
                ModelLoader(tmp.name)
    
    def test_load_model_success(self, temp_model_file):
        """Prueba la carga exitosa de un modelo."""
        loader = ModelLoader(temp_model_file)
        model = loader.load_model()
        
        assert model is not None
        assert loader.model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'layers')
    
    def test_model_info_collection(self, temp_model_file):
        """Prueba la recolección de información del modelo."""
        loader = ModelLoader(temp_model_file)
        loader.load_model()
        
        info = loader.get_model_info()
        assert isinstance(info, dict)
        assert 'model_path' in info
        assert 'input_shape' in info
        assert 'output_shape' in info
        assert 'total_params' in info
        assert 'layers_count' in info
    
    def test_model_validation(self, temp_model_file):
        """Prueba la validación de arquitectura del modelo."""
        loader = ModelLoader(temp_model_file)
        loader.load_model()
        
        validation = loader.validate_model_architecture()
        assert isinstance(validation, dict)
        assert 'is_valid' in validation
        assert 'warnings' in validation
        assert 'errors' in validation
        assert 'recommendations' in validation
    
    def test_conv_layer_detection(self, temp_model_file):
        """Prueba la detección de capas convolucionales."""
        loader = ModelLoader(temp_model_file)
        loader.load_model()
        
        conv_layers = loader.get_conv_layer_names()
        assert isinstance(conv_layers, list)
        assert len(conv_layers) > 0
        
        last_conv = loader.get_last_conv_layer_name()
        assert last_conv is not None
        assert last_conv in conv_layers
    
    def test_get_default_model_path(self):
        """Prueba la función de ruta por defecto."""
        default_path = get_default_model_path()
        assert isinstance(default_path, str)
        assert default_path.endswith('conv_MLP_84.h5')


class TestGradCAMGenerator:
    """Pruebas para la clase GradCAMGenerator."""
    
    @pytest.fixture
    def mock_model(self):
        """Fixture que crea un modelo simple para testing."""
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(512, 512, 1), 
                   name='conv2d_test'),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', name='conv2d_1_test'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model
    
    @pytest.fixture
    def sample_image(self):
        """Fixture que crea una imagen de prueba."""
        return np.random.rand(512, 512, 1).astype(np.float32)
    
    def test_initialization(self, mock_model):
        """Prueba la inicialización del generador Grad-CAM."""
        generator = GradCAMGenerator(mock_model)
        
        assert generator.model is mock_model
        assert generator.target_layer_name is not None
        assert generator.grad_model is not None
    
    def test_target_layer_detection(self, mock_model):
        """Prueba la detección automática de capa objetivo."""
        generator = GradCAMGenerator(mock_model)
        
        # Debe detectar la última capa convolucional
        assert 'conv' in generator.target_layer_name.lower()
    
    def test_gradcam_generation(self, mock_model, sample_image):
        """Prueba la generación de Grad-CAM."""
        generator = GradCAMGenerator(mock_model)
        
        # Generar heatmap
        heatmap = generator.generate_gradcam(sample_image, class_idx=0)
        
        assert isinstance(heatmap, np.ndarray)
        assert len(heatmap.shape) == 2  # Debe ser 2D
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
    
    def test_gradcam_with_auto_class(self, mock_model, sample_image):
        """Prueba Grad-CAM con selección automática de clase."""
        generator = GradCAMGenerator(mock_model)
        
        # Sin especificar clase (selección automática)
        heatmap = generator.generate_gradcam(sample_image, class_idx=None)
        
        assert isinstance(heatmap, np.ndarray)
        assert len(heatmap.shape) == 2
    
    def test_fallback_heatmap(self, mock_model):
        """Prueba la generación de heatmap de respaldo."""
        generator = GradCAMGenerator(mock_model)
        
        # Crear imagen que cause error
        invalid_image = np.array([])  # Array vacío
        
        heatmap = generator.generate_gradcam(invalid_image, class_idx=0)
        
        # Debe retornar un heatmap de respaldo
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (512, 512)
    
    def test_heatmap_resize(self, mock_model, sample_image):
        """Prueba el redimensionamiento de heatmaps."""
        generator = GradCAMGenerator(mock_model)
        
        # Generar heatmap pequeño
        small_heatmap = np.random.rand(64, 64).astype(np.float32)
        
        # Redimensionar
        resized = generator.resize_heatmap(small_heatmap, (256, 256))
        
        assert resized.shape == (256, 256)
        assert resized.dtype == np.float32
    
    def test_layer_info(self, mock_model):
        """Prueba la obtención de información de capa."""
        generator = GradCAMGenerator(mock_model)
        
        info = generator.get_layer_info()
        assert isinstance(info, dict)
        assert 'layer_name' in info
        assert 'layer_type' in info
        assert 'output_shape' in info
    
    def test_attention_regions_analysis(self, mock_model):
        """Prueba el análisis de regiones de atención."""
        generator = GradCAMGenerator(mock_model)
        
        # Crear heatmap sintético con regiones de alta activación
        heatmap = np.zeros((100, 100))
        heatmap[20:40, 20:40] = 0.8  # Región de alta atención
        heatmap[60:80, 60:80] = 0.6  # Otra región
        
        analysis = generator.analyze_attention_regions(heatmap, threshold=0.5)
        
        assert isinstance(analysis, dict)
        assert 'num_regions' in analysis
        assert 'regions' in analysis
        assert 'global_max' in analysis
        assert 'center_of_mass' in analysis
        assert analysis['num_regions'] >= 1


class TestHeatmapOverlay:
    """Pruebas para la clase HeatmapOverlay."""
    
    @pytest.fixture
    def overlay_generator(self):
        """Fixture que crea un generador de overlays."""
        return HeatmapOverlay()
    
    @pytest.fixture
    def sample_image(self):
        """Fixture que crea una imagen de prueba."""
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_heatmap(self):
        """Fixture que crea un heatmap de prueba."""
        return np.random.rand(64, 64).astype(np.float32)
    
    def test_initialization(self, overlay_generator):
        """Prueba la inicialización del generador de overlays."""
        assert overlay_generator.default_alpha == 0.6
        assert overlay_generator.default_colormap is not None
        assert overlay_generator.target_size == (512, 512)
    
    def test_create_overlay(self, overlay_generator, sample_image, sample_heatmap):
        """Prueba la creación de overlay básico."""
        overlay = overlay_generator.create_overlay(sample_image, sample_heatmap)
        
        assert isinstance(overlay, np.ndarray)
        assert len(overlay.shape) == 3  # RGB
        assert overlay.shape[2] == 3     # 3 canales
        assert overlay.dtype == np.uint8
        assert overlay.shape[:2] == overlay_generator.target_size
    
    def test_overlay_with_custom_params(self, overlay_generator, sample_image, sample_heatmap):
        """Prueba overlay con parámetros personalizados."""
        overlay = overlay_generator.create_overlay(
            sample_image, 
            sample_heatmap,
            alpha=0.3,
            colormap=2  # COLORMAP_HOT
        )
        
        assert isinstance(overlay, np.ndarray)
        assert overlay.shape[:2] == overlay_generator.target_size
    
    def test_fallback_overlay(self, overlay_generator, sample_image):
        """Prueba la creación de overlay de respaldo."""
        fallback = overlay_generator.create_fallback_overlay(sample_image)
        
        assert isinstance(fallback, np.ndarray)
        assert len(fallback.shape) == 3
        assert fallback.shape[:2] == overlay_generator.target_size
    
    def test_side_by_side_comparison(self, overlay_generator, sample_image, sample_heatmap):
        """Prueba la comparación lado a lado."""
        comparison = overlay_generator.create_side_by_side_comparison(
            sample_image, 
            sample_heatmap
        )
        
        assert isinstance(comparison, np.ndarray)
        assert len(comparison.shape) == 3
        # Debe ser 3 veces más ancho (original + heatmap + overlay)
        assert comparison.shape[1] == overlay_generator.target_size[0] * 3
    
    def test_overlay_statistics(self, overlay_generator, sample_image, sample_heatmap):
        """Prueba el cálculo de estadísticas del overlay."""
        stats = overlay_generator.get_overlay_stats(sample_image, sample_heatmap)
        
        assert isinstance(stats, dict)
        assert 'original_shape' in stats
        assert 'heatmap_min' in stats
        assert 'heatmap_max' in stats
        assert 'heatmap_mean' in stats
        assert 'target_size' in stats
    
    def test_grayscale_image_handling(self, overlay_generator, sample_heatmap):
        """Prueba el manejo de imágenes en escala de grises."""
        grayscale_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        
        overlay = overlay_generator.create_overlay(grayscale_img, sample_heatmap)
        
        assert isinstance(overlay, np.ndarray)
        assert len(overlay.shape) == 3  # Debe convertirse a RGB
        assert overlay.shape[2] == 3


class TestIntegrationModelsVisualization:
    """Pruebas de integración para modelos y visualización."""
    
    @pytest.fixture
    def complete_pipeline_components(self):
        """Fixture que crea componentes para pipeline completo."""
        # Crear modelo simple
        model = Sequential([
            Conv2D(8, (3, 3), activation='relu', input_shape=(512, 512, 1)),
            MaxPooling2D((4, 4)),
            Flatten(),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Crear imagen y heatmap de prueba
        image = np.random.rand(512, 512, 1).astype(np.float32)
        
        return {
            'model': model,
            'image': image
        }
    
    def test_gradcam_to_overlay_pipeline(self, complete_pipeline_components):
        """Prueba el pipeline completo desde Grad-CAM hasta overlay."""
        components = complete_pipeline_components
        model = components['model']
        image = components['image']
        
        # Generar Grad-CAM
        gradcam_gen = GradCAMGenerator(model)
        heatmap = gradcam_gen.generate_gradcam(image, class_idx=0)
        
        # Crear overlay
        overlay_gen = HeatmapOverlay()
        # Convertir imagen a RGB para overlay
        image_rgb = np.repeat(image, 3, axis=-1)
        overlay = overlay_gen.create_overlay(image_rgb, heatmap)
        
        # Verificar resultado final
        assert isinstance(overlay, np.ndarray)
        assert len(overlay.shape) == 3
        assert overlay.shape[2] == 3
        assert overlay.dtype == np.uint8
    
    def test_error_handling_pipeline(self):
        """Prueba el manejo de errores en el pipeline."""
        # Crear datos inválidos
        invalid_model = None
        invalid_image = np.array([])
        
        # El sistema debe manejar errores graciosamente
        overlay_gen = HeatmapOverlay()
        
        # Crear heatmap y overlay de respaldo
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        fallback = overlay_gen.create_fallback_overlay(dummy_image)
        
        assert isinstance(fallback, np.ndarray)
        assert len(fallback.shape) == 3


# Configuración de fixtures globales
@pytest.fixture(scope="session", autouse=True)
def setup_tensorflow():
    """Configuración global de TensorFlow para tests."""
    # Configurar TensorFlow para usar menos memoria
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception:
        # Ignorar errores de configuración GPU en entorno de pruebas
        pass


if __name__ == "__main__":
    # Ejecutar pruebas si se ejecuta directamente
    pytest.main([__file__, "-v"])
