# ── Hospital Quotation OpenEnv — Dockerfile ───────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Nyor AI"
LABEL description="Nyor AI - Hospital Quotation OpenEnv benchmark server"

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
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
