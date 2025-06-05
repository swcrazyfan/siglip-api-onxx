# Use Python 3.13 slim image for better performance
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for both CPU and GPU scenarios
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Environment variables with Docker-optimized defaults
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Auto-scaling enabled by default (will detect available resources)
ENV AUTO_SCALE_WORKERS=true
ENV CPU_UTILIZATION_FACTOR=0.85

# Image embedding optimized defaults
ENV BATCH_SIZE=auto
ENV BATCH_WAIT_TIME_MS=auto
ENV MAX_QUEUE_SIZE=1000

# Model configuration
ENV SIGLIP_MODEL=google/siglip2-base-patch16-512
ENV SIGLIP_VERBOSE_OUTPUT=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
