FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data logs

# Expose both — docker-compose controls which is published per service
EXPOSE 8000
EXPOSE 8501

# Default to backend; frontend overrides this in docker-compose
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
