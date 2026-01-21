
FROM python:3.10-slim

# Crear un usuario para no correr como root (seguridad de Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# Copiar dependencias e instalar
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copiar el código del proyecto
COPY --chown=user . .

# Hugging Face Spaces usa el puerto 7860 por defecto
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]