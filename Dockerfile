# Usa una imagen base de Python 3.13 slim
FROM python:3

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requisitos al contenedor
COPY requirements.txt ./

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código al contenedor
COPY . .

# Comando para ejecutar la aplicación
CMD ["python", "main.py"]
