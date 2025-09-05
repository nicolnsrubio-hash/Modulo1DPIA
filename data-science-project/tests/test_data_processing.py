#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pruebas unitarias para módulos de procesamiento de datos.

Este archivo contiene pruebas para validar el funcionamiento correcto de
los módulos de lectura y preprocesamiento de imágenes radiográficas.

Autor: Refactorizado para Módulo 1 - UAO
Fecha: 2025
"""

# Importaciones estándar
import os
import tempfile
from pathlib import Path

# Importaciones de terceros
import pytest
import numpy as np
from PIL import Image
import cv2

# Importaciones locales
try:
    from src.data.read_img import ImageReader
    from src.data.preprocess_img import ImagePreprocessor
except ImportError:
    import sys
    import os
    # Agregar el directorio padre al path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.data.read_img import ImageReader
    from src.data.preprocess_img import ImagePreprocessor


class TestImageReader:
    """Pruebas para la clase ImageReader."""
    
    @pytest.fixture
    def image_reader(self):
        """Fixture que crea una instancia de ImageReader."""
        return ImageReader()
    
    @pytest.fixture
    def sample_image(self):
        """Fixture que crea una imagen de prueba."""
        # Crear imagen sintética RGB
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        return img_array
    
    @pytest.fixture
    def temp_image_file(self, sample_image):
        """Fixture que crea un archivo temporal de imagen."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Convertir array a PIL y guardar
            pil_image = Image.fromarray(sample_image)
            pil_image.save(tmp.name, 'JPEG')
            yield tmp.name
        # Limpiar archivo temporal
        os.unlink(tmp.name)
    
    def test_initialization(self, image_reader):
        """Prueba la inicialización correcta del ImageReader."""
        assert image_reader.supported_formats is not None
        assert len(image_reader.supported_formats) > 0
        assert '.jpg' in image_reader.supported_formats
        assert '.dcm' in image_reader.supported_formats
    
    def test_supported_format_check(self, image_reader):
        """Prueba la verificación de formatos soportados."""
        # Formatos válidos
        assert image_reader.is_supported_format('test.jpg') is True
        assert image_reader.is_supported_format('test.jpeg') is True
        assert image_reader.is_supported_format('test.png') is True
        assert image_reader.is_supported_format('test.dcm') is True
        
        # Formatos inválidos
        assert image_reader.is_supported_format('test.txt') is False
        assert image_reader.is_supported_format('test.doc') is False
    
    def test_read_standard_image(self, image_reader, temp_image_file):
        """Prueba la lectura de imágenes en formato estándar."""
        img_array = image_reader.read_image(temp_image_file)
        
        # Verificar que se lee correctamente
        assert isinstance(img_array, np.ndarray)
        assert len(img_array.shape) == 3  # RGB
        assert img_array.shape[2] == 3    # 3 canales
        assert img_array.dtype == np.uint8
    
    def test_read_nonexistent_file(self, image_reader):
        """Prueba el manejo de archivos inexistentes."""
        with pytest.raises(FileNotFoundError):
            image_reader.read_image("archivo_inexistente.jpg")
    
    def test_read_unsupported_format(self, image_reader):
        """Prueba el manejo de formatos no soportados."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            with pytest.raises(ValueError, match="Formato no soportado"):
                image_reader.read_image(tmp.name)
    
    def test_get_image_info(self, image_reader, temp_image_file):
        """Prueba la obtención de información de imagen."""
        info = image_reader.get_image_info(temp_image_file)
        
        assert isinstance(info, dict)
        assert 'path' in info
        assert 'format' in info
        assert 'size_bytes' in info
        assert 'width' in info
        assert 'height' in info
        assert info['format'] == '.jpg'
        assert info['size_bytes'] > 0


class TestImagePreprocessor:
    """Pruebas para la clase ImagePreprocessor."""
    
    @pytest.fixture
    def preprocessor(self):
        """Fixture que crea una instancia de ImagePreprocessor."""
        return ImagePreprocessor()
    
    @pytest.fixture
    def sample_rgb_image(self):
        """Fixture que crea una imagen RGB de prueba."""
        return np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_grayscale_image(self):
        """Fixture que crea una imagen en escala de grises."""
        return np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    def test_initialization(self, preprocessor):
        """Prueba la inicialización correcta del ImagePreprocessor."""
        assert preprocessor.target_size == (512, 512)
        assert preprocessor.clahe_enabled is True
        assert preprocessor.normalization_range == (0.0, 1.0)
        assert preprocessor.clahe is not None
    
    def test_preprocess_rgb_image(self, preprocessor, sample_rgb_image):
        """Prueba el preprocesamiento de imagen RGB."""
        processed = preprocessor.preprocess_image(sample_rgb_image)
        
        # Verificar dimensiones y formato
        assert isinstance(processed, np.ndarray)
        assert processed.shape == (512, 512, 1)  # Escala grises con canal
        assert processed.dtype == np.float32
        
        # Verificar normalización
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0
    
    def test_preprocess_grayscale_image(self, preprocessor, sample_grayscale_image):
        """Prueba el preprocesamiento de imagen en escala de grises."""
        processed = preprocessor.preprocess_image(sample_grayscale_image)
        
        assert isinstance(processed, np.ndarray)
        assert processed.shape == (512, 512, 1)
        assert processed.dtype == np.float32
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0
    
    def test_preprocess_with_batch_format(self, preprocessor, sample_rgb_image):
        """Prueba el preprocesamiento con formato batch."""
        processed = preprocessor.preprocess_image(
            sample_rgb_image, 
            apply_batch_format=True
        )
        
        # Verificar que incluye dimensión de batch
        assert processed.shape == (1, 512, 512, 1)
        assert processed.dtype == np.float32
    
    def test_resize_functionality(self, preprocessor):
        """Prueba la funcionalidad de redimensionamiento."""
        # Imagen de tamaño diferente
        test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        processed = preprocessor.preprocess_image(test_image)
        
        # Debe estar redimensionada al tamaño objetivo
        assert processed.shape[:2] == (512, 512)
    
    def test_clahe_toggle(self, sample_rgb_image):
        """Prueba el funcionamiento de CLAHE habilitado/deshabilitado."""
        # Con CLAHE
        preprocessor_with_clahe = ImagePreprocessor(clahe_enabled=True)
        processed_with_clahe = preprocessor_with_clahe.preprocess_image(
            sample_rgb_image
        )
        
        # Sin CLAHE
        preprocessor_no_clahe = ImagePreprocessor(clahe_enabled=False)
        processed_no_clahe = preprocessor_no_clahe.preprocess_image(
            sample_rgb_image
        )
        
        # Ambos deben tener la misma forma pero valores potencialmente diferentes
        assert processed_with_clahe.shape == processed_no_clahe.shape
        assert processed_with_clahe.dtype == processed_no_clahe.dtype
    
    def test_batch_processing(self, preprocessor):
        """Prueba el procesamiento por lotes."""
        # Crear lote de imágenes sintéticas
        batch = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        processed_batch = preprocessor.preprocess_batch(batch)
        
        # Verificar formato de lote
        assert isinstance(processed_batch, np.ndarray)
        assert processed_batch.shape == (3, 512, 512, 1)
        assert processed_batch.dtype == np.float32
    
    def test_preprocessing_parameters(self, preprocessor):
        """Prueba la obtención de parámetros de preprocesamiento."""
        params = preprocessor.get_preprocessing_params()
        
        assert isinstance(params, dict)
        assert 'target_size' in params
        assert 'clahe_enabled' in params
        assert 'normalization_range' in params
        assert params['target_size'] == (512, 512)
    
    def test_clahe_parameter_update(self, preprocessor):
        """Prueba la actualización de parámetros CLAHE."""
        # Actualizar parámetros
        new_clip_limit = 3.0
        new_tile_size = (8, 8)
        
        preprocessor.update_clahe_params(
            clip_limit=new_clip_limit,
            tile_grid_size=new_tile_size
        )
        
        # Verificar actualización
        params = preprocessor.get_preprocessing_params()
        assert params['clahe_clip_limit'] == new_clip_limit
        assert params['clahe_tile_grid_size'] == new_tile_size
    
    def test_visualization_steps(self, preprocessor, sample_rgb_image):
        """Prueba la visualización de pasos de preprocesamiento."""
        steps = preprocessor.visualize_preprocessing_steps(sample_rgb_image)
        
        assert isinstance(steps, dict)
        assert 'original' in steps
        assert 'resized' in steps
        assert 'grayscale' in steps
        assert 'normalized' in steps
        assert 'with_channel' in steps
        assert 'with_batch' in steps
        
        # Verificar progresión de formas
        original = steps['original']
        final = steps['with_batch']
        
        assert len(original.shape) == 3  # RGB original
        assert len(final.shape) == 4     # Con batch y canal
        assert final.shape == (1, 512, 512, 1)
    
    def test_edge_cases(self, preprocessor):
        """Prueba casos extremos y manejo de errores."""
        # Imagen muy pequeña
        tiny_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        processed = preprocessor.preprocess_image(tiny_image)
        assert processed.shape == (512, 512, 1)
        
        # Imagen con valores extremos
        extreme_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        processed = preprocessor.preprocess_image(extreme_image)
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0


class TestIntegrationDataProcessing:
    """Pruebas de integración para el pipeline completo."""
    
    @pytest.fixture
    def reader(self):
        return ImageReader()
    
    @pytest.fixture
    def preprocessor(self):
        return ImagePreprocessor()
    
    def test_complete_pipeline(self, reader, preprocessor):
        """Prueba el pipeline completo de procesamiento."""
        # Crear imagen temporal
        test_image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Guardar imagen
            pil_image = Image.fromarray(test_image)
            pil_image.save(tmp_path, 'PNG')
            
            # Pipeline completo: lectura + preprocesamiento
            img_array = reader.read_image(tmp_path)
            processed_img = preprocessor.preprocess_image(
                img_array, 
                apply_batch_format=True
            )
            
            # Verificar resultado final
            assert isinstance(processed_img, np.ndarray)
            assert processed_img.shape == (1, 512, 512, 1)
            assert processed_img.dtype == np.float32
            assert 0.0 <= processed_img.min() <= processed_img.max() <= 1.0
            
        finally:
            # Limpiar archivo temporal con manejo de errores para Windows
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except (OSError, PermissionError):
                # Ignorar errores de permisos en Windows
                pass


if __name__ == "__main__":
    # Ejecutar pruebas si se ejecuta directamente
    pytest.main([__file__, "-v"])
