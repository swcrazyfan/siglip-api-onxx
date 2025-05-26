# Multi-stage build for smaller final image
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd -m -u 1000 user

# Create and set permissions for the base cache directory before switching to user.
# The Hugging Face library will create subdirectories (like 'huggingface/hub') within this.
RUN mkdir -p /home/user/.cache && \
    chown -R user:user /home/user/.cache

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:/opt/venv/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
    # TRANSFORMERS_CACHE is deprecated and removed, HF_HOME is preferred.

WORKDIR $HOME/app

# Copy application code
COPY --chown=user:user app.py .

# Pre-download the model during build for faster startup (optional)
# Uncomment the following lines if you want to include the model in the image
# This will increase image size but reduce startup time
# RUN python -c "from huggingface_hub import hf_hub_download; \
#     hf_hub_download('pulsejet/siglip-base-patch16-256-multilingual-onnx', 'onnx/model_quantized.onnx')"

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
