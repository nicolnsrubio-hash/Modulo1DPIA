# 🌫️ Detector de Neumonía con IA

<div align="center">

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)

**Herramienta de Deep Learning para la detección rápida de neumonía en imágenes radiográficas**

[Instalación](#instalación) • [Uso](#uso) • [Docker](#docker) • [Arquitectura](#arquitectura) • [Licencia](#licencia)

</div>

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Características](#características)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Uso](#uso)
- [Arquitectura del Proyecto](#arquitectura-del-proyecto)
- [Estructura de Carpetas](#estructura-de-carpetas)
- [Docker](#docker)
- [Pruebas](#pruebas)
- [Contribución](#contribución)
- [Licencia](#licencia)
- [Autores](#autores)

## 📝 Descripción

Sistema de detección de neumonía basado en Deep Learning que procesa imágenes radiográficas de tórax en formato DICOM y las clasifica en 3 categorías:

- 🦠 **Neumonía Bacteriana**
- 🦠 **Neumonía Viral** 
- ✅ **Sin Neumonía (Normal)**

Utiliza **Grad-CAM** para generar mapas de calor que resaltan las regiones de interés médico, proporcionando explicabilidad a las predicciones del modelo.

## ✨ Características

- 🎨 **Interfaz gráfica intuitiva** con Tkinter
- 🔍 **Visualización Grad-CAM** para explicabilidad de IA
- 📊 **Exportación de reportes** en PDF y CSV
- 📦 **Soporte múltiples formatos**: DICOM, JPEG, PNG
- 🐳 **Containerización Docker** incluida
- 🧪 **Pruebas unitarias** con pytest
- 📐 **Código PEP8 compliant** con docstrings

## 🔧 Requisitos del Sistema

### Versiones de Software
- **Python**: 3.9+
- **TensorFlow**: 2.x
- **Sistema Operativo**: Windows 10/11, macOS, Linux

### Dependencias Principales
```
tensorflow>=2.8.0
opencv-python>=4.5.0
pydicom>=2.3.0
pillow>=9.0.0
numpy>=1.21.0
```

## 🚀 Instalación

### Opción 1: Anaconda (Recomendada)

```bash
# Crear entorno virtual
conda create -n pneumonia-detector python=3.9
conda activate pneumonia-detector

# Clonar repositorio
git clone https://github.com/tu-usuario/UAO-Neumonia.git
cd UAO-Neumonia

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 2: venv

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Clonar repositorio
git clone https://github.com/tu-usuario/UAO-Neumonia.git
cd UAO-Neumonia

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 3: uv (Más rápido)

```bash
# Instalar uv
pip install uv

# Clonar repositorio
git clone https://github.com/tu-usuario/UAO-Neumonia.git
cd UAO-Neumonia

# Instalación con uv
uv sync
```

## 📖 Uso

### Interfaz Gráfica

```bash
# Activar entorno virtual
conda activate pneumonia-detector

# Ejecutar aplicación
python detector_neumonia.py
```

#### Pasos para usar la interfaz:

1. **Cargar imagen**: Botón "Cargar Imagen" → Seleccionar archivo DICOM/JPEG/PNG
2. **Ingresar datos**: Cédula del paciente en el campo de texto
3. **Predecir**: Botón "Predecir" → Esperar resultados
4. **Guardar**: Botón "Guardar" → Exportar a CSV
5. **Generar PDF**: Botón "PDF" → Crear reporte
6. **Limpiar**: Botón "Borrar" → Nueva predicción

### API Programática

```python
from data_science_project.src.integrator import PneumoniaDetector

# Inicializar detector
detector = PneumoniaDetector()
detector.load_model()

# Predecir desde archivo
clase, probabilidad, heatmap = detector.predict_pneumonia("imagen.dcm")

# Predecir desde array
import numpy as np
img_array = np.load("imagen.npy")
clase, probabilidad, heatmap = detector.predict_from_array(img_array)
```

## 🏗️ Arquitectura del Proyecto

### Módulos Principales

#### `integrator.py`
Módulo principal que coordina todos los componentes:
- Integra lectura, preprocesamiento, predicción y visualización
- Retorna clase, probabilidad y mapa de calor Grad-CAM

#### `read_img.py` 
Script de lectura de imágenes:
- Soporte DICOM, JPEG, PNG
- Conversión a arrays NumPy para procesamiento

#### `preprocess_img.py`
Pipeline de preprocesamiento:
- Resize a 512x512
- Conversión a escala de grises
- Ecualización CLAHE
- Normalización [0,1]
- Formato batch tensor

#### `load_model.py`
Gestor del modelo de ML:
- Carga modelo `conv_MLP_84.h5`
- Validación de arquitectura
- Información del modelo

#### `grad_cam.py`
Generación de mapas de atención:
- Implementación Grad-CAM
- Visualización de regiones relevantes
- Overlay colorido sobre imagen original

## 📁 Estructura de Carpetas

```
UAO-Neumonia/
├── 📁 data-science-project/
│   ├── 📁 data/
│   │   ├── 📁 raw/          # Datasets sin procesar
│   │   ├── 📁 processed/    # Datasets limpios
│   │   └── 📁 external/     # Datos externos
│   ├── 📁 notebooks/
│   │   ├── 📄 01-data-exploration.ipynb
│   │   ├── 📄 02-feature-engineering.ipynb
│   │   ├── 📄 03-model-training.ipynb
│   │   └── 📄 04-evaluation.ipynb
│   ├── 📁 src/
│   │   ├── 📁 data/
│   │   │   ├── 📄 read_img.py
│   │   │   └── 📄 preprocess_img.py
│   │   ├── 📁 models/
│   │   │   ├── 📄 load_model.py
│   │   │   └── 📄 grad_cam.py
│   │   ├── 📁 visualizations/
│   │   │   └── 📄 heatmap_overlay.py
│   │   └── 📄 integrator.py
│   ├── 📁 tests/
│   │   ├── 📄 test_data_processing.py
│   │   └── 📄 test_models.py
│   ├── 📁 reports/
│   │   └── 📁 figures/
│   └── 📁 docs/
├── 📄 detector_neumonia.py    # Interfaz gráfica principal
├── 📄 requirements.txt
├── 📄 Dockerfile
├── 📄 README.md
└── 📄 LICENSE
```

## 🐳 Docker

### Construcción

```bash
# Construir imagen
docker build -t pneumonia-detector .

# Ejecutar contenedor
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  pneumonia-detector
```

### Docker Compose

```yaml
version: '3.8'
services:
  pneumonia-detector:
    build: .
    environment:
      - DISPLAY=${DISPLAY}
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
      - ./data:/app/data
```

## 🧪 Pruebas

```bash
# Ejecutar todas las pruebas
pytest

# Pruebas con cobertura
pytest --cov=src --cov-report=html

# Pruebas específicas
pytest tests/test_models.py -v
```

## 📊 Modelo de ML

### Arquitectura
Red Neuronal Convolucional basada en el trabajo de F. Pasa et al.:

- **5 bloques convolucionales** con conexiones skip
- **Filtros**: 16, 32, 48, 64, 80 (3x3)
- **Regularización**: Dropout 20%
- **Clasificador**: 3 capas Dense (1024, 1024, 3)

### Grad-CAM
Técnica de explicabilidad que resalta regiones importantes:
- Cálculo de gradientes de la clase objetivo
- Combinación lineal con mapas de activación
- Visualización como mapa de calor

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

### Estándares de Código
- **PEP 8** compliance
- **Docstrings** en todas las funciones
- **Type hints** donde sea apropiado
- **Pruebas unitarias** para nueva funcionalidad

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2025 Universidad Autónoma de Occidente

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

## 👥 Autores

### Equipo Original
- **Isabella Torres Revelo** - [@isa-tr](https://github.com/isa-tr)
- **Nicolas Diaz Salazar** - [@nicolasdiazsalazar](https://github.com/nicolasdiazsalazar)

### Contribuidores Módulo 1
- **Tu Nombre** - Refactorización PEP8, Docker, tests unitarias

## 📞 Soporte

- 📧 Email: soporte@uao.edu.co
- 📱 Issues: [GitHub Issues](https://github.com/tu-usuario/UAO-Neumonia/issues)
- 📖 Documentación: [Wiki del proyecto](https://github.com/tu-usuario/UAO-Neumonia/wiki)

## 🔗 Enlaces Útiles

- [Imágenes de prueba](https://drive.google.com/drive/folders/1WOuL0wdVC6aojy8IfssHcqZ4Up14dy0g?usp=drive_link)
- [Paper original](https://link-to-paper.com)
- [Documentación TensorFlow](https://tensorflow.org)
- [Grad-CAM explicación](https://arxiv.org/abs/1610.02391)

---

<div align="center">
Hecho con ❤️ para la detección temprana de neumonía
</div>
