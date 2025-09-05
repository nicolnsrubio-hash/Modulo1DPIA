# Dockerfile para proyecto de detección de neumonía
# Universidad Autónoma de Occidente - Módulo 1

# Usar imagen base de Python
FROM python:3.10-slim

# Establecer información del mantenedor
LABEL maintainer="Universidad Autónoma de Occidente"
LABEL version="1.0.0"
LABEL description="Detector de neumonía usando Deep Learning"

# Establecer variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_CPP_MIN_LOG_LEVEL=2

# Instalar dependencias básicas del sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/data/models /app/logs /app/temp

# Configurar PYTHONPATH
ENV PYTHONPATH="/app"

# Exponer puerto para aplicación web (opcional)
EXPOSE 5000

# Verificar instalación
RUN python -c "import tensorflow as tf; import cv2; import numpy as np; print('Dependencias instaladas correctamente')"

# Comando por defecto - ejecutar pruebas
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Para ejecutar la aplicación principal (comentado por defecto)
# CMD ["python", "src/integrator.py"]

# Para modo interactivo de desarrollo
# CMD ["bash"]
