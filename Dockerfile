# --------------------------
# Etapa 1: Construcción
# --------------------------

# Utiliza una imagen base oficial de Python para construir las dependencias
FROM python:3.9-slim AS builder

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar requirements.txt para aprovechar el caché de Docker
COPY requirements.txt .

# Instalar dependencias en un entorno aislado
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --------------------------
# Etapa 2: Imagen Final
# --------------------------

# Utiliza una imagen base limpia y ligera para la ejecución
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

COPY requirements.txt .
# Copiar dependencias instaladas desde la etapa de construcción
COPY --from=builder /root/.cache /root/.cache

# Copiar todo el directorio app/ en /app/app
COPY app/ app/

# Copiar datos en el directorio /app/data
COPY data/ data/

# Instalar dependencias
RUN pip3 install -r requirements.txt

# Declarar el volumen para las salidas del modelo
VOLUME /app/data

# Exponer el puerto de FastAPI
EXPOSE 8080

# Ejecutar el orquestador con la ruta completa
CMD ["python3", "app/orchestrator.py"]


