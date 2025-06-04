# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a SigLIP2 API project that provides OpenAI-compatible multimodal embeddings using Google's SigLIP2 model with PyTorch. The API supports both text and image inputs with automatic detection.

## Common Development Commands

### Running the Application

```bash
# Local development
uvicorn app:app --host 0.0.0.0 --port 8001 --reload

# Docker compose (recommended)
docker-compose up -d
docker-compose logs -f
docker-compose down

# Docker build and run
docker build -t siglip2-api .
docker run -p 8001:8001 siglip2-api
docker run --gpus all -p 8001:8001 siglip2-api  # With GPU support
```

### Dependency Management

```bash
# Install dependencies
pip install -r requirements.txt
```

## Architecture Overview

The application is a FastAPI-based REST API that:

1. **Model Loading**: Uses PyTorch and Hugging Face transformers to load SigLIP2 models (`google/siglip2-base-patch16-512` by default)
2. **Input Processing**: Automatically detects input types (text, URL, base64 image, file path) and processes accordingly
3. **Embedding Generation**: Provides unified embeddings for text and images in the same vector space
4. **OpenAI Compatibility**: Implements OpenAI's embedding API format for drop-in replacement

Key architectural decisions:
- Async model loading with locks to prevent race conditions
- Automatic device selection (MPS > CUDA > CPU)
- Model caching using Docker volumes
- Environment-based model selection via `SIGLIP_MODEL`

## API Endpoints

- `/v1/embeddings` - OpenAI-compatible embeddings endpoint
- `/v1/embeddings/image` - File upload support for embeddings
- `/v1/rank` - Rank candidates by similarity
- `/v1/classify` - Zero-shot image classification
- `/v1/models` - List available models
- `/health` - Health check endpoint

## Environment Variables

- `SIGLIP_MODEL` - Model selection (default: `google/siglip2-base-patch16-512`)
- `LOG_LEVEL` - Logging level (default: INFO)
- `SIGLIP_VERBOSE_OUTPUT` - Enable verbose debug logging (default: true)
- `HF_HOME` - Hugging Face cache directory
- `CUDA_VISIBLE_DEVICES` - GPU device selection