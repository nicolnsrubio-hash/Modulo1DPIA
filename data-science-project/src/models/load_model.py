#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cargador y Gestor del Modelo de Red Neuronal.

Este módulo se encarga de la carga, validación y gestión del modelo de red
neuronal convolucional previamente entrenado para la detección de neumonía.

Autor: Refactorizado para Módulo 1 - UAO
Fecha: 2025
"""

# Importaciones estándar
import os
from typing import Optional, Dict, Any
from pathlib import Path

# Importaciones de terceros
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

# Importaciones para manejo de warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class ModelLoader:
    """
    Gestor para la carga y validación del modelo de detección de neumonía.
    
    Esta clase proporciona métodos para cargar el modelo de red neuronal
    convolucional, validar su arquitectura y obtener información relevante
    del modelo.
    
    Attributes:
        model_path (str): Ruta al archivo del modelo .h5.
        model (Model): Modelo de Keras cargado.
        model_info (dict): Información del modelo cargado.
    """
    
    def __init__(self, model_path: str):
        """
        Inicializa el cargador de modelo.
        
        Args:
            model_path (str): Ruta al archivo del modelo .h5.
            
        Raises:
            FileNotFoundError: Si el archivo del modelo no existe.
            ValueError: Si la ruta no es válida.
        """
        self.model_path = self._validate_model_path(model_path)
        self.model: Optional[Model] = None
        self.model_info: Dict[str, Any] = {}
    
    def _validate_model_path(self, model_path: str) -> str:
        """
        Valida la ruta del modelo.
        
        Args:
            model_path (str): Ruta al archivo del modelo.
            
        Returns:
            str: Ruta validada del modelo.
            
        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si la extensión no es válida.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        # Verificar extensión
        if not model_path.lower().endswith(('.h5', '.hdf5')):
            raise ValueError(
                f"Extensión de modelo no válida. "
                f"Se esperaba .h5 o .hdf5, se recibió: {Path(model_path).suffix}"
            )
        
        return os.path.abspath(model_path)
    
    def load_model(self, compile_model: bool = False) -> Model:
        """
        Carga el modelo de red neuronal desde el archivo.
        
        Args:
            compile_model (bool): Si compilar el modelo después de cargarlo.
            
        Returns:
            Model: Modelo de Keras cargado.
            
        Raises:
            RuntimeError: Si hay error al cargar el modelo.
        """
        try:
            print(f"📁 Cargando modelo desde: {self.model_path}")
            
            # Cargar modelo con configuración personalizada
            self.model = load_model(
                self.model_path, 
                compile=compile_model,
                custom_objects=self._get_custom_objects()
            )
            
            # Recolectar información del modelo
            self._collect_model_info()
            
            print(f"✅ Modelo cargado exitosamente")
            print(f"📊 Arquitectura: {len(self.model.layers)} capas")
            print(f"📊 Parámetros: {self.model.count_params():,}")
            
            return self.model
            
        except Exception as e:
            error_msg = f"Error cargando modelo: {e}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def _get_custom_objects(self) -> dict:
        """
        Obtiene objetos personalizados necesarios para cargar el modelo.
        
        Returns:
            dict: Diccionario con objetos personalizados.
        """
        # Aquí se pueden agregar funciones de activación o capas personalizadas
        # si el modelo las requiere
        return {}
    
    def _collect_model_info(self):
        """Recolecta información detallada del modelo cargado."""
        if self.model is None:
            return
        
        try:
            self.model_info = {
                "model_path": self.model_path,
                "input_shape": self.model.input_shape,
                "output_shape": self.model.output_shape,
                "total_params": self.model.count_params(),
                "trainable_params": self._count_trainable_params(),
                "non_trainable_params": self._count_non_trainable_params(),
                "layers_count": len(self.model.layers),
                "layer_types": self._get_layer_types(),
                "model_size_mb": self._get_model_size_mb(),
                "has_batch_normalization": self._has_batch_normalization(),
                "has_dropout": self._has_dropout(),
                "activation_functions": self._get_activation_functions()
            }
        except Exception as e:
            print(f"⚠️ Warning: No se pudo recolectar toda la información del modelo: {e}")
    
    def _count_trainable_params(self) -> int:
        """Cuenta parámetros entrenables."""
        return int(
            sum([tf.keras.backend.count_params(w) 
                 for w in self.model.trainable_weights])
        )
    
    def _count_non_trainable_params(self) -> int:
        """Cuenta parámetros no entrenables."""
        return int(
            sum([tf.keras.backend.count_params(w) 
                 for w in self.model.non_trainable_weights])
        )
    
    def _get_layer_types(self) -> Dict[str, int]:
        """Obtiene tipos de capas y su conteo."""
        layer_types = {}
        for layer in self.model.layers:
            layer_type = type(layer).__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        return layer_types
    
    def _get_model_size_mb(self) -> float:
        """Obtiene el tamaño del archivo del modelo en MB."""
        try:
            size_bytes = os.path.getsize(self.model_path)
            return round(size_bytes / (1024 * 1024), 2)
        except Exception:
            return 0.0
    
    def _has_batch_normalization(self) -> bool:
        """Verifica si el modelo tiene capas de normalización."""
        return any('BatchNormalization' in type(layer).__name__ 
                   for layer in self.model.layers)
    
    def _has_dropout(self) -> bool:
        """Verifica si el modelo tiene capas de dropout."""
        return any('Dropout' in type(layer).__name__ 
                   for layer in self.model.layers)
    
    def _get_activation_functions(self) -> set:
        """Obtiene las funciones de activación usadas."""
        activations = set()
        for layer in self.model.layers:
            if hasattr(layer, 'activation') and layer.activation is not None:
                activations.add(layer.activation.__name__)
        return activations
    
    def get_model_summary(self, print_fn=print) -> str:
        """
        Obtiene un resumen detallado del modelo.
        
        Args:
            print_fn: Función para imprimir (por defecto print).
            
        Returns:
            str: Resumen del modelo en formato string.
        """
        if self.model is None:
            return "Modelo no cargado"
        
        try:
            # Capturar el summary en string
            summary_lines = []
            self.model.summary(print_fn=lambda x: summary_lines.append(x))
            return '\n'.join(summary_lines)
        except Exception as e:
            return f"Error obteniendo resumen: {e}"
    
    def validate_model_architecture(self) -> Dict[str, Any]:
        """
        Valida la arquitectura del modelo para detección de neumonía.
        
        Returns:
            dict: Resultados de la validación.
        """
        if self.model is None:
            return {"error": "Modelo no cargado"}
        
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        try:
            # Verificar forma de entrada
            input_shape = self.model.input_shape
            if input_shape[1:3] != (512, 512):
                validation_results["warnings"].append(
                    f"Tamaño de entrada no estándar: {input_shape[1:3]} "
                    f"(se esperaba 512x512)"
                )
            
            # Verificar canales de entrada
            if len(input_shape) == 4 and input_shape[-1] != 1:
                validation_results["warnings"].append(
                    f"Canales de entrada: {input_shape[-1]} "
                    f"(se esperaba 1 para escala de grises)"
                )
            
            # Verificar salida
            output_shape = self.model.output_shape
            if len(output_shape) == 2:
                num_classes = output_shape[1]
                if num_classes not in [1, 2, 3]:
                    validation_results["warnings"].append(
                        f"Número de clases inusual: {num_classes}"
                    )
            
            # Verificar arquitectura CNN
            has_conv_layers = any('Conv' in type(layer).__name__ 
                                 for layer in self.model.layers)
            if not has_conv_layers:
                validation_results["warnings"].append(
                    "No se encontraron capas convolucionales"
                )
            
            # Recomendaciones
            if not self._has_dropout():
                validation_results["recommendations"].append(
                    "Considerar agregar capas Dropout para regularización"
                )
            
            if not self._has_batch_normalization():
                validation_results["recommendations"].append(
                    "Considerar agregar normalización por lotes"
                )
            
        except Exception as e:
            validation_results["errors"].append(f"Error en validación: {e}")
            validation_results["is_valid"] = False
        
        return validation_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información completa del modelo.
        
        Returns:
            dict: Información detallada del modelo.
        """
        return self.model_info.copy() if self.model_info else {"error": "Modelo no cargado"}
    
    def get_conv_layer_names(self) -> list:
        """
        Obtiene nombres de las capas convolucionales para Grad-CAM.
        
        Returns:
            list: Lista de nombres de capas convolucionales.
        """
        if self.model is None:
            return []
        
        conv_layers = []
        for layer in self.model.layers:
            layer_type = type(layer).__name__
            if 'Conv' in layer_type and 'D' in layer_type:  # Conv2D, Conv1D, etc.
                conv_layers.append(layer.name)
        
        return conv_layers
    
    def get_last_conv_layer_name(self) -> Optional[str]:
        """
        Obtiene el nombre de la última capa convolucional.
        
        Returns:
            str: Nombre de la última capa convolucional o None si no hay.
        """
        conv_layers = self.get_conv_layer_names()
        return conv_layers[-1] if conv_layers else None


# Función de conveniencia para carga rápida
def load_pneumonia_model(model_path: str) -> Model:
    """
    Función de conveniencia para cargar el modelo de neumonía.
    
    Args:
        model_path (str): Ruta al archivo del modelo.
        
    Returns:
        Model: Modelo cargado.
    """
    loader = ModelLoader(model_path)
    return loader.load_model()


# Función para obtener ruta por defecto del modelo
def get_default_model_path() -> str:
    """
    Obtiene la ruta por defecto del modelo.
    
    Returns:
        str: Ruta por defecto al modelo conv_MLP_84.h5.
    """
    # Buscar en el directorio raíz del proyecto
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    
    model_candidates = [
        project_root / "conv_MLP_84.h5",
        project_root / "models" / "conv_MLP_84.h5",
        project_root / "data-science-project" / "models" / "conv_MLP_84.h5"
    ]
    
    for candidate in model_candidates:
        if candidate.exists():
            return str(candidate)
    
    # Si no se encuentra, retornar la ruta esperada
    return str(project_root / "conv_MLP_84.h5")


if __name__ == "__main__":
    # Ejemplo de uso y testing
    print("🤖 Cargador de Modelo de Detección de Neumonía")
    
    # Intentar cargar modelo por defecto
    default_model_path = get_default_model_path()
    print(f"📍 Ruta por defecto: {default_model_path}")
    
    if os.path.exists(default_model_path):
        try:
            loader = ModelLoader(default_model_path)
            model = loader.load_model()
            
            # Mostrar información
            print(f"\n📊 Información del modelo:")
            info = loader.get_model_info()
            for key, value in info.items():
                print(f"   {key}: {value}")
            
            # Validar arquitectura
            print(f"\n✅ Validación de arquitectura:")
            validation = loader.validate_model_architecture()
            print(f"   Válido: {validation['is_valid']}")
            if validation['warnings']:
                print(f"   Advertencias: {validation['warnings']}")
            if validation['recommendations']:
                print(f"   Recomendaciones: {validation['recommendations']}")
            
            # Capas convolucionales
            conv_layers = loader.get_conv_layer_names()
            print(f"\n🔍 Capas convolucionales encontradas: {len(conv_layers)}")
            for layer_name in conv_layers:
                print(f"   - {layer_name}")
            
            last_conv = loader.get_last_conv_layer_name()
            print(f"🎯 Última capa conv (para Grad-CAM): {last_conv}")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
    else:
        print(f"⚠️ Modelo no encontrado en: {default_model_path}")
        print("   Coloque el archivo conv_MLP_84.h5 en el directorio raíz del proyecto")
    
    print("\n✅ Módulo de carga de modelo listo")
