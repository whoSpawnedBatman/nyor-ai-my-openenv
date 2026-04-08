# ── Pharma B2B Quotation Hub — Dockerfile ───────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Nyor AI Team"
LABEL description="Nyor AI - Pharma B2B Quotation Hub benchmark server"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first (layer-cache friendly)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Make env package importable from /app
ENV PYTHONPATH=/app

# Expose FastAPI port (HF Spaces default)
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "server/app.py"]
