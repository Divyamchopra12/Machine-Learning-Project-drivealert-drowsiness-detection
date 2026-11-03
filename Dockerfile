# DriveAlert – web app container (FastAPI backend + frontend)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# OpenCV / MediaPipe need these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# pygame is only for the desktop app – skip it on a server
RUN grep -v pygame requirements.txt > req-server.txt && pip install -r req-server.txt

COPY . .
RUN mkdir -p data models

EXPOSE 8000
# Render / Railway / Cloud Run pass the port in $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
