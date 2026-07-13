# AeroAgent application container.
# Ollama and ChromaDB run as separate services (see docker-compose.yml) - this image only contains
# the Python app + API.

FROM python:3.13-slim AS base

WORKDIR /app

# System deps needed by some Python packages (e.g. build tools for torch/chromadb wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential\
    curl\
    && rm -rf /var/lib/apt/lists/*

# Install deps first for better layer caching
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# App code
COPY main.py ./

# Runtime env vars - overridden by docker-compose.yml for inter-service networking
ENV OLLAMA_BASE_URL="http://ollama:11434/v1"
ENV AEROAGENT_MODEL="llama3.2:3b"

EXPOSE 8000

CMD ["uvicorn", "aeroagent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]