# SigLIP2 API

A lightweight, fast, and OpenAI-compatible API for multimodal embeddings using Google's SigLIP2 model with PyTorch. Features intelligent batch processing, automatic hardware detection, and optimizations for image embeddings.

## 🚀 Features

- **OpenAI-compatible**: Drop-in replacement for OpenAI embeddings API
- **Multimodal**: Supports both text and image embeddings in the same vector space
- **🧠 Intelligent Batch Processing**: Automatic batching with queue management for optimal performance
- **🔧 Auto-Scaling**: Automatically detects and scales to available CPU cores and GPU
- **⚡ Hardware Optimization**: Optimized configurations for GPU vs CPU, with special tuning for image embeddings
- **Fast**: Uses PyTorch with optimized inference and efficient batching
- **Automatic Input Detection**: Automatically detects if input is text, URL, base64 image, or file path
- **File Upload Support**: Direct image file uploads via multipart/form-data
- **Zero-shot Classification**: Built-in endpoint for image classification
- **Ranking/Similarity**: Compare and rank texts/images by similarity
- **Docker Ready**: Easy deployment with automatic resource detection
- **GPU Support**: Automatic CUDA acceleration when available (RTX 3060, etc.)

## 📋 Requirements

- Docker and docker-compose
- (Optional) NVIDIA GPU with CUDA support

## 🛠️ Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/swcrazyfan/siglip2-api.git
cd siglip2-api

# Start the service (automatically detects available hardware)
docker-compose up -d

# Check logs to see detected hardware (GPU or CPU)
docker-compose logs -f

# Stop the service
docker-compose down
```

### Using Docker

```bash
# Build the image
docker build -t siglip2-api .

# Run CPU-only (auto-detects and uses available CPU cores)
docker run -p 8000:8000 siglip2-api

# Run with GPU support (auto-detects GPU if available)
docker run --gpus all -p 8000:8000 siglip2-api

# Limit CPU cores for shared systems
docker run --cpus="6" -p 8000:8000 siglip2-api

# Combine GPU + CPU limits
docker run --gpus all --cpus="6" -p 8000:8000 siglip2-api
```

### Performance Monitoring

```bash
# Check hardware detection and performance estimates
curl http://localhost:8000/health

# Example response shows your detected hardware:
{
  "status": "healthy",
  "hardware": {
    "cpu_cores": 16,
    "has_gpu": true,
    "gpu_name": "NVIDIA GeForce RTX 3060",
    "gpu_memory_gb": 12,
    "device_type": "cuda"
  },
  "batch_processing": {
    "enabled": true,
    "batch_size": 16,
    "max_concurrent_batches": 2,
    "auto_scaling": true
  },
  "performance_estimates": {
    "images_per_second": "30.0",
    "note": "GPU acceleration active"
  }
}
```

## ⚡ Performance Features

### Intelligent Batch Processing
- **Automatic batching**: Groups individual requests for efficient processing
- **Queue management**: Handles traffic spikes without failures
- **Mixed workloads**: Optimizes text, image, and joint embeddings separately

### Hardware Auto-Detection
- **CPU scaling**: Automatically uses available CPU cores (great for your AMD Ryzen 7 5700X)
- **GPU priority**: Automatically detects and prioritizes GPU when available (RTX 3060, etc.)
- **Memory optimization**: Adjusts batch sizes based on available GPU/system memory
- **Docker awareness**: Works perfectly in containerized environments

### Image Embedding Optimizations
- **GPU acceleration**: 10-50x faster image processing with CUDA
- **Batch optimization**: Larger batches for GPU, smaller responsive batches for CPU
- **Memory efficiency**: Smart memory management for different hardware configurations

### Expected Performance

| Hardware | Images/Second | Configuration |
|----------|---------------|---------------|
| RTX 3060 12GB | 20-50 | Auto: 2 workers, batch size 16 |
| AMD Ryzen 7 5700X (CPU) | 2-8 | Auto: 6 workers, batch size 6 |
| 4 vCPUs (cloud) | 1-3 | Auto: 3 workers, batch size 4 |

## 📡 API Endpoints

### 1. List Available Models (OpenAI Compatible)

```bash
GET /v1/models
```

Returns the available model(s) in OpenAI format. Required for LiteLLM integration.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "siglip2-large-patch16-512",
      "object": "model",
      "owned_by": "google"
    }
  ]
}
```

### 2. Generate Embeddings (OpenAI Compatible)

```bash
POST /v1/embeddings
```

**Request:**
```json
{
  "input": "a photo of a cat",
  "model": "siglip2-large-patch16-512"
}
```

**Examples:**

```bash
# Text embedding
curl -X POST "http://localhost:8001/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input": "a beautiful sunset"}'

# Image URL embedding
curl -X POST "http://localhost:8001/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input": "https://example.com/image.jpg"}'

# Base64 image embedding
curl -X POST "http://localhost:8001/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input": "data:image/jpeg;base64,/9j/4AAQ..."}'

# Multiple inputs
curl -X POST "http://localhost:8001/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "a cat",
      "https://example.com/cat.jpg",
      "a dog"
    ]
  }'
```

### 3. Generate Embeddings with File Upload

```bash
POST /v1/embeddings/image
```

**Upload an image file:**
```bash
# Upload image file
curl -X POST "http://localhost:8001/v1/embeddings/image" \
  -F "file=@/path/to/image.jpg"

# With form data (text or URL)
curl -X POST "http://localhost:8001/v1/embeddings/image" \
  -F "input=a photo of a cat"
```

**Python example:**
```python
import requests

# Upload file
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8001/v1/embeddings/image",
        files={"file": ("image.jpg", f, "image/jpeg")}
    )
```

### 4. Rank/Similarity

```bash
POST /v1/rank
```

**Request:**
```json
{
  "query": "https://example.com/image.jpg",
  "candidates": [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird"
  ]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8001/v1/rank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cute animals",
    "candidates": [
      "https://example.com/cat.jpg",
      "https://example.com/dog.jpg",
      "https://example.com/fish.jpg"
    ]
  }'
```

### 5. Zero-shot Classification

```bash
POST /v1/classify
```

**Request:**
```json
{
  "image": "https://example.com/image.jpg",
  "labels": ["cat", "dog", "bird", "fish"]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8001/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",
    "labels": ["cat", "dog", "hamster", "rabbit"]
  }'
```

### 6. Zero-shot Classification with File Upload

```bash
POST /v1/classify/image
```

**Upload image and classify:**
```bash
# Upload and classify
curl -X POST "http://localhost:8001/v1/classify/image" \
  -F "file=@/path/to/image.jpg" \
  -F "labels=cat,dog,bird,fish"
```

**Python example:**
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8001/v1/classify/image",
        files={"file": ("image.jpg", f, "image/jpeg")},
        data={"labels": "cat,dog,bird,fish"}
    )
```

### 7. Health Check

```bash
GET /health
```

## 🎯 Use Cases

### Image Search
Find similar images based on text queries:
```python
import requests
import numpy as np

# Get embeddings for query and images
query_resp = requests.post("http://localhost:8001/v1/embeddings", 
    json={"input": "sunset over mountains"})
query_embedding = np.array(query_resp.json()["data"][0]["embedding"])

# Get embeddings for your image database
image_urls = ["image1.jpg", "image2.jpg", "image3.jpg"]
images_resp = requests.post("http://localhost:8001/v1/embeddings",
    json={"input": image_urls})

# Calculate similarities
image_embeddings = [np.array(d["embedding"]) for d in images_resp.json()["data"]]
similarities = [np.dot(query_embedding, img_emb) for img_emb in image_embeddings]
```

### File Upload Workflow
Process local images without base64 encoding:
```python
import requests

# Upload and get embedding
with open("local_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8001/v1/embeddings/image",
        files={"file": f}
    )
    embedding = response.json()["data"][0]["embedding"]

# Upload and classify
with open("local_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8001/v1/classify/image",
        files={"file": f},
        data={"labels": "cat,dog,bird"}
    )
    classifications = response.json()["classifications"]
```

### Cross-modal Retrieval
Find images that match text descriptions or vice versa:
```python
# Use the ranking endpoint
response = requests.post("http://localhost:8001/v1/rank",
    json={
        "query": "a red sports car",
        "candidates": image_urls
    })
rankings = response.json()["results"][0]["rankings"]
```

## ⚠️ Limitations

- **Text Length**: Maximum 64 tokens (~10-15 words). Longer text is automatically truncated
- **Image Size**: Images are resized to 256x256 pixels
- **Languages**: Optimized for multilingual text but trained primarily on English image-text pairs

## 📄 Configuration

### Automatic Configuration (Recommended)
The API automatically detects your hardware and configures optimal settings:

**Docker with GPU:**
```bash
# Automatically detects RTX 3060 and optimizes for GPU
docker run --gpus all -p 8000:8000 siglip2-api
```

**Docker CPU-only:**
```bash
# Automatically detects AMD Ryzen cores and optimizes for CPU
docker run --cpus="6" -p 8000:8000 siglip2-api
```

### Manual Configuration (Advanced)
Override auto-detection with environment variables:

```bash
# Custom worker count (overrides auto-detection)
docker run -e MAX_CONCURRENT_BATCHES=4 -p 8000:8000 siglip2-api

# Custom batch size for your workload
docker run -e BATCH_SIZE=12 -p 8000:8000 siglip2-api

# Disable auto-scaling (use fixed settings)
docker run -e AUTO_SCALE_WORKERS=false -e MAX_CONCURRENT_BATCHES=2 -p 8000:8000 siglip2-api
```

### Environment Variables

**Hardware Detection:**
- `AUTO_SCALE_WORKERS`: Enable automatic scaling (default: true)
- `CPU_UTILIZATION_FACTOR`: CPU usage factor (default: 0.85 for Docker)
- `MAX_CONCURRENT_BATCHES`: Override auto-detected worker count
- `BATCH_SIZE`: Override auto-detected batch size

**Performance Tuning:**
- `BATCH_WAIT_TIME_MS`: Max wait time to fill batches (auto: 75ms GPU, 35ms CPU)
- `MAX_QUEUE_SIZE`: Maximum queued requests (default: 1000)
- `QUEUE_OVERFLOW_ACTION`: "reject" or "wait" when queue full

**Model Configuration:**
- `SIGLIP_MODEL`: Model to use (default: `google/siglip2-base-patch16-512`)
- `SIGLIP_VERBOSE_OUTPUT`: Enable verbose logging (default: true)

### Model Selection

Choose between different SigLIP2 models:

**Available Models:**
- `google/siglip2-base-patch16-512` (default) - Smaller, faster, 768-dim embeddings
- `google/siglip2-large-patch16-512` - Larger, more accurate, 1152-dim embeddings

```bash
# Use large model with auto-scaling
docker run --gpus all -e SIGLIP_MODEL=google/siglip2-large-patch16-512 -p 8000:8000 siglip2-api
```

## 🚀 Performance Tips

1. **GPU First**: Use `--gpus all` for 10-50x speedup on image embeddings
2. **CPU Optimization**: Let auto-scaling detect your CPU cores (works great with Ryzen 7 5700X)
3. **Batch Requests**: Send multiple images in one request for better throughput
4. **Monitor Health**: Check `/health` endpoint for real-time performance stats
5. **Resource Limits**: Use Docker `--cpus` to limit CPU usage in shared environments
6. **Queue Monitoring**: Watch queue size in health check to detect bottlenecks

## 🔍 Monitoring & Troubleshooting

### Real-time Monitoring
```bash
# Check current performance and queue status
curl http://localhost:8000/health | jq .

# Monitor logs for batch processing info
docker logs siglip2-api -f | grep "Batch\|Worker\|GPU"
```

### Performance Testing
```bash
# Test concurrent batch processing
python test_batch.py

# Expected output shows batching working:
# ✅ Request 0: A beautiful sunset... -> 768 dims
# ✅ Completed 8/8 requests
# ⏱️  Total time: 1.23 seconds (much faster than 8x individual)
```

### Common Solutions

**GPU Not Detected:**
```bash
# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check Docker GPU support
docker run --gpus all -e SIGLIP_VERBOSE_OUTPUT=true siglip2-api
```

**High CPU Usage:**
```bash
# Limit CPU cores for shared systems
docker run --cpus="6" -p 8000:8000 siglip2-api

# Or manually set worker count
docker run -e MAX_CONCURRENT_BATCHES=4 -p 8000:8000 siglip2-api
```

**Slow Performance:**
- Check `/health` endpoint for hardware detection
- Ensure GPU is detected if available
- Monitor queue size - high queue indicates need for more workers
- Use larger batch sizes for GPU workloads

## 📊 Model Information

### Base Model (Default)
- **Model**: `google/siglip2-base-patch16-512`
- **Parameters**: ~400M
- **Embedding Dimension**: 768
- **Speed**: Faster inference
- **Memory**: Lower memory usage

### Large Model
- **Model**: `google/siglip2-large-patch16-512`
- **Parameters**: ~1.1B
- **Embedding Dimension**: 1152 (estimated)
- **Speed**: Slower inference
- **Memory**: Higher memory usage
- **Accuracy**: Better performance

### Common Specifications
- **Max Text Length**: 64 tokens
- **Image Resolution**: 512x512  
- **Languages**: Multilingual (best performance on English)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Research for the SigLIP2 model
- Hugging Face for model hosting and transformers library
- PyTorch team for the deep learning framework
