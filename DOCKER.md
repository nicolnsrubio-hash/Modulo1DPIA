# 🐳 Guía de Docker - Detector de Neumonía con IA

Esta guía explica cómo usar Docker para ejecutar el sistema de detección de neumonía con soporte completo para interfaz gráfica.

## 📋 Prerrequisitos

### En Linux (Ubuntu/Debian)
```bash
# Instalar Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo usermod -aG docker $USER

# Habilitar X11 forwarding
xhost +local:docker
```

### En Windows
- Instalar [Docker Desktop](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe)
- Instalar [VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/)
- Configurar VcXsrv con: "Disable access control" ✓

### En macOS
- Instalar [Docker Desktop](https://desktop.docker.com/mac/main/amd64/Docker.dmg)
- Instalar XQuartz: `brew install --cask xquartz`
- Configurar: `xhost +localhost`

## 🏗️ Construcción

### Opción 1: Docker Compose (Recomendada)
```bash
# Construir y ejecutar
docker-compose up --build

# Solo construir
docker-compose build
```

### Opción 2: Docker directo
```bash
# Construir imagen
docker build -t neumonia-detector .

# Verificar construcción
docker images | grep neumonia-detector
```

## 🚀 Ejecución

### Interfaz Gráfica Completa

#### Linux
```bash
# Habilitar X11
xhost +local:docker

# Ejecutar con GUI
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/data:/app/data \
  neumonia-detector
```

#### Windows
```bash
# Ejecutar VcXsrv primero, luego:
docker run -it --rm \
  -e DISPLAY=host.docker.internal:0.0 \
  -v "%cd%/data:/app/data" \
  neumonia-detector
```

#### macOS
```bash
# Ejecutar XQuartz primero
xhost +localhost
docker run -it --rm \
  -e DISPLAY=host.docker.internal:0 \
  -v $(pwd)/data:/app/data \
  neumonia-detector
```

### Con Docker Compose

```bash
# Ejecutar aplicación principal
docker-compose up neumonia-detector

# Modo desarrollo (shell interactivo)
docker-compose --profile dev run --rm neumonia-dev

# Ejecutar tests
docker-compose --profile test run --rm neumonia-test

# Jupyter Lab (análisis de datos)
docker-compose --profile jupyter up neumonia-jupyter
```

## 📁 Volúmenes y Persistencia

### Directorios Importantes
```
./data/          → /app/data/          # Datos de entrada
./reports/       → /app/reports/       # Reportes generados
./models/        → /app/models/        # Modelos ML
./data-science-project/notebooks/ → /app/notebooks/ # Jupyter notebooks
```

### Ejemplo de Uso con Volúmenes
```bash
# Crear estructura de directorios
mkdir -p data/{raw,processed,external} reports models

# Ejecutar con volúmenes específicos
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/data:/app/data:rw \
  -v $(pwd)/reports:/app/reports:rw \
  -v $(pwd)/models:/app/models:rw \
  neumonia-detector
```

## 🧪 Testing y Desarrollo

### Ejecutar Pruebas Unitarias
```bash
# Con Docker Compose
docker-compose --profile test run --rm neumonia-test

# Con Docker directo
docker run --rm -v $(pwd):/app neumonia-detector \
  python3.9 -m pytest data-science-project/tests/ -v
```

### Modo Desarrollo
```bash
# Shell interactivo
docker-compose --profile dev run --rm neumonia-dev

# O con Docker directo
docker run -it --rm \
  -v $(pwd):/app \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  neumonia-detector bash
```

### Jupyter Lab
```bash
# Ejecutar Jupyter
docker-compose --profile jupyter up neumonia-jupyter

# Acceder en: http://localhost:8888
# Token se muestra en logs del contenedor
```

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
# Configurar timezone
-e TZ=America/Bogota

# Configurar logging
-e PYTHONUNBUFFERED=1

# GPU support (si disponible)
--gpus all
```

### Recursos del Sistema
```bash
# Limitar memoria y CPU
docker run --rm \
  --memory=2g \
  --cpus="1.5" \
  neumonia-detector
```

### Networking
```bash
# Usar red del host (Linux)
--network host

# Exponer puertos específicos
-p 8080:8080 -p 8888:8888
```

## 🐛 Solución de Problemas

### Error: "cannot connect to X server"
```bash
# Linux: Verificar X11
echo $DISPLAY
xhost +local:docker

# Windows: Verificar VcXsrv está ejecutándose
# macOS: Verificar XQuartz está ejecutándose
```

### Error: "Permission denied"
```bash
# Verificar permisos de volúmenes
ls -la data/ reports/ models/

# Cambiar propietario si es necesario
sudo chown -R $USER:$USER data/ reports/ models/
```

### Error: "Module not found"
```bash
# Reconstruir imagen
docker-compose build --no-cache neumonia-detector

# Verificar instalación
docker run --rm neumonia-detector python3.9 -c "import tensorflow, cv2, pydicom; print('OK')"
```

### Performance Issues
```bash
# Verificar recursos disponibles
docker stats

# Aumentar límites
docker run --memory=4g --cpus="2.0" neumonia-detector
```

## 📝 Comandos Útiles

### Gestión de Contenedores
```bash
# Listar contenedores
docker ps -a

# Logs de contenedor
docker-compose logs neumonia-detector

# Limpiar contenedores parados
docker container prune

# Entrar a contenedor ejecutándose
docker exec -it neumonia-detector bash
```

### Gestión de Imágenes
```bash
# Listar imágenes
docker images

# Limpiar imágenes no usadas
docker image prune

# Eliminar imagen específica
docker rmi neumonia-detector
```

### Backups y Exportación
```bash
# Exportar imagen
docker save neumonia-detector > neumonia-detector.tar

# Importar imagen
docker load < neumonia-detector.tar

# Backup de datos
docker run --rm -v $(pwd)/data:/backup ubuntu tar czf /backup/data-backup.tar.gz /app/data
```

## 🚀 Producción

### Docker en Servidor
```bash
# Ejecutar como daemon
docker-compose up -d

# Monitoreo
docker-compose logs -f

# Actualizar
docker-compose pull
docker-compose up -d --build
```

### Con systemd (Linux)
```ini
# /etc/systemd/system/neumonia-detector.service
[Unit]
Description=Detector de Neumonía con IA
Requires=docker.service
After=docker.service

[Service]
Type=forking
RemainAfterExit=yes
WorkingDirectory=/path/to/UAO-Neumonia
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

```bash
# Habilitar servicio
sudo systemctl enable neumonia-detector.service
sudo systemctl start neumonia-detector.service
```

## 📞 Soporte

Si encuentras problemas:

1. Verificar logs: `docker-compose logs neumonia-detector`
2. Verificar recursos: `docker stats`
3. Probar en modo desarrollo: `docker-compose --profile dev run --rm neumonia-dev`
4. Reportar issue con logs completos

---

**Universidad Autónoma de Occidente - Módulo 1**  
Sistema de Detección de Neumonía con IA
