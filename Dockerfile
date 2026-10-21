# Base Python 3.12 image
FROM python:3.12-slim

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CHROMA_PERSIST_DIRECTORY=/app/data/chroma \
    REGISTRY_PATH=/app/data/registry.json

# Install minimal OS dependencies for healthchecks and HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Upgrade pip and install CPU-only PyTorch wheel first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Copy and install application requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configuration
COPY src/ /app/src/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
COPY pytest.ini /app/pytest.ini
COPY README.md /app/README.md

# Make entrypoint executable, set up persistent data directory and ownership
RUN chmod +x /app/docker-entrypoint.sh && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose FastAPI backend (8000) and Streamlit frontend (8501)
EXPOSE 8000 8501

# Configure entrypoint and default command
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["api"]
