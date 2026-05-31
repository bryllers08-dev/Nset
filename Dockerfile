# ── Use the official slim Python image (pip included, guaranteed) ──────────
FROM python:3.11-slim

# ── System deps for TensorFlow (libstdc++, etc.) ───────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ───────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (layer-cached unless requirements changes)
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir

# ── Copy bot source ─────────────────────────────────────────────────────────
COPY deriv_lstm_bot.py .

# ── DATA_DIR defaults to /app/data; override with Railway env var ───────────
ENV DATA_DIR=/app/data \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# ── Create data directory ────────────────────────────────────────────────────
RUN mkdir -p /app/data

CMD ["python3", "deriv_lstm_bot.py"]
