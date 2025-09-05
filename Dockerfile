# Dockerfile para Detector de Neumonía con IA
# Universidad Autónoma de Occidente - Módulo 1
# Soporte completo para interfaz gráfica con X11 forwarding

# Usar imagen base de Ubuntu con Python 3.9
FROM ubuntu:20.04

# Configurar variables de entorno para evitar interacciones durante instalación
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Bogota
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Variables para GUI
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1
ENV XAUTHORITY=/tmp/.docker.xauth

# Información del contenedor
LABEL maintainer="Universidad Autónoma de Occidente"
LABEL description="Detector de Neumonía con IA - Interfaz gráfica con Grad-CAM"
LABEL version="1.0.0"

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    # Python y herramientas básicas
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    # Dependencias para GUI
    python3-tk \
    x11-apps \
    xvfb \
    x11vnc \
    # Dependencias para OpenCV y procesamiento de imágenes
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Dependencias para DICOM
    dcmtk \
    # Fuentes y herramientas adicionales
    fontconfig \
    fonts-dejavu-core \
    # Utilidades de sistema
    wget \
    curl \
    unzip \
    git \
    gosu \
    # Limpieza
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 -s /bin/bash neumonia && \
    echo "neumonia ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de configuración primero (para aprovechar cache de Docker)
COPY requirements.txt /app/

# Actualizar pip y instalar dependencias Python
RUN python3.9 -m pip install --upgrade pip setuptools wheel && \
    python3.9 -m pip install --no-cache-dir -r requirements.txt

# Copiar código fuente del proyecto
COPY . /app/

# Crear directorios necesarios
RUN mkdir -p /app/data/raw /app/data/processed /app/data/external \
             /app/reports/figures /app/logs /app/models \
             /tmp/.X11-unix /home/neumonia/.config

# Script de entrada para configurar X11
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash

# Configurar X11 forwarding
if [ ! -z "$DISPLAY" ]; then
    echo "Configurando X11 forwarding para DISPLAY=$DISPLAY"
    
    # Crear archivo de autorización X11 si no existe
    if [ ! -f $XAUTHORITY ]; then
        touch $XAUTHORITY
        chmod 600 $XAUTHORITY
    fi
    
    # Verificar conectividad X11
    if command -v xdpyinfo >/dev/null 2>&1; then
        if timeout 5 xdpyinfo >/dev/null 2>&1; then
            echo "✓ Conexión X11 establecida correctamente"
        else
            echo "⚠ Advertencia: No se puede conectar al servidor X11"
            echo "  Asegúrate de ejecutar 'xhost +local:docker' en el host"
        fi
    fi
fi

# Configurar timezone
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Verificar instalación de dependencias críticas
echo "Verificando dependencias críticas..."
python3.9 -c "import tensorflow as tf; print(f'✓ TensorFlow: {tf.__version__}')" 2>/dev/null || echo "❌ TensorFlow no disponible"
python3.9 -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}')" 2>/dev/null || echo "❌ OpenCV no disponible"
python3.9 -c "import tkinter; print('✓ Tkinter disponible')" 2>/dev/null || echo "❌ Tkinter no disponible"
python3.9 -c "import pydicom; print('✓ PyDICOM disponible')" 2>/dev/null || echo "❌ PyDICOM no disponible"

# Ejecutar comando con usuario apropiado
if [ "$(id -u)" -eq 0 ]; then
    # Si somos root, cambiar a usuario neumonia
    exec gosu neumonia "$@"
else
    # Si ya somos usuario normal, ejecutar directamente
    exec "$@"
fi
EOF

# Configurar permisos
RUN chown -R neumonia:neumonia /app /home/neumonia /tmp/.X11-unix && \
    chmod -R 755 /app && \
    chmod +x /app/entrypoint.sh && \
    chmod +x /app/detector_neumonia.py 2>/dev/null || echo "Main script permissions set"

# Puerto para aplicaciones web (si se añade en el futuro)
EXPOSE 8080

# Volúmenes recomendados
VOLUME ["/app/data", "/app/reports", "/app/models"]

# Comando por defecto
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3.9", "detector_neumonia.py"]

# Comandos de uso:
# docker build -t neumonia-detector .
# docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix neumonia-detector
# docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd)/data:/app/data neumonia-detector
