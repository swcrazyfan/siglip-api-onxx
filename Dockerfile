FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Add labels to link the package to your repository
LABEL org.opencontainers.image.source=https://github.com/swcrazyfan/siglip-api-onxx
LABEL org.opencontainers.image.description="SigLIP 2 API with PyTorch"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory and cache directory
WORKDIR /app
RUN mkdir -p /app/cache

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY download_model.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8001

# Health check - increased start period since model downloads at startup
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Create startup script to download model then start the application
RUN echo '#!/bin/bash\necho "Downloading model on startup..."\npython download_model.py\necho "Starting application..."\nexec uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1' > /app/start.sh && \
    chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]
