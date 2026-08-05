# Safety Monitoring System -- API service (FastAPI + YOLO + rule engine + LLM)
#
# Build context is the repository root (see docker-compose.yml), because
# this image needs the existing YOLO/ and LLM/ packages alongside API/.
#
# Requires an NVIDIA GPU + nvidia-container-toolkit on the host at runtime,
# since LLM/llm_client.py hard-requires CUDA.

FROM python:3.14-slim

# System libraries needed by opencv-python-headless and video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached across rebuilds
COPY API/requirements.txt /app/API/requirements.txt
RUN pip install --no-cache-dir -r /app/API/requirements.txt

# Copy the existing project packages (read-only dependencies for the API)
# plus the API code itself.
COPY YOLO/ /app/YOLO/
COPY LLM/ /app/LLM/
COPY API/ /app/API/

ENV PYTHONPATH=/app \
    YOLO_MODEL_PATH=/app/YOLO/models/best.onnx \
    LLM_MODEL_PATH=/app/LLM/model \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "8000"]
