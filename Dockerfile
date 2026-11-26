# Stage 1: Build React + TypeScript + Vite Frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python FastAPI + TensorFlow + Ultralytics YOLO11 Inference Backend
FROM python:3.12-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/matplotlib_cache \
    YOLO_CONFIG_DIR=/tmp/ultralytics_config \
    TF_CPP_MIN_LOG_LEVEL=2

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir opencv-python-headless pyyaml psutil requests tqdm polars ultralytics-thop && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps ultralytics

COPY backend/ ./backend/
COPY ml/ ./ml/
COPY yolo11/outputs/ ./yolo11/outputs/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
