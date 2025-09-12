FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/
COPY data/ ./data/

ENV PYTHONPATH="/app/src"

ENV PORT=8000
EXPOSE $PORT

RUN printf '#!/bin/bash\nexec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}\n' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1


CMD ["/entrypoint.sh"]
# CMD uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
