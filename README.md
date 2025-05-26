# SigLIP ONNX API

A lightweight, fast, and OpenAI-compatible API for multimodal embeddings using Google's SigLIP model with ONNX runtime. Supports both text and image inputs with automatic detection, including direct file uploads.

## 🚀 Features

- **OpenAI-compatible**: Drop-in replacement for OpenAI embeddings API
- **Multimodal**: Supports both text and image embeddings in the same vector space
- **Fast**: Uses ONNX runtime for optimized inference
- **Automatic Input Detection**: Automatically detects if input is text, URL, base64 image, or file path
- **File Upload Support**: Direct image file uploads via multipart/form-data
- **Zero-shot Classification**: Built-in endpoint for image classification
- **Ranking/Similarity**: Compare and rank texts/images by similarity
- **Docker Ready**: Easy deployment with Docker and docker-compose
- **GPU Support**: Optional CUDA acceleration

## 📋 Requirements

- Docker and docker-compose
- (Optional) NVIDIA GPU with CUDA support

## 🛠️ Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/swcrazyfan/siglip-api-onxx.git
cd siglip-api-onxx

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Using Docker

```bash
# Build the image
docker build -t siglip-api .

# Run the container
docker run -p 8000:8000 siglip-api

# With GPU support
docker run --gpus all -p 8000:8000 siglip-api
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

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
      "id": "siglip-base-patch16-256-multilingual",
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
  "model": "siglip-base-patch16-256-multilingual"
}
```

**Examples:**

```bash
# Text embedding
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input": "a beautiful sunset"}'

# Image URL embedding
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input": "https://example.com/image.jpg"}'

# Base64 image embedding
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input": "data:image/jpeg;base64,/9j/4AAQ..."}'

# Multiple inputs
curl -X POST "http://localhost:8000/v1/embeddings" \
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
curl -X POST "http://localhost:8000/v1/embeddings/image" \
  -F "file=@/path/to/image.jpg"

# With form data (text or URL)
curl -X POST "http://localhost:8000/v1/embeddings/image" \
  -F "input=a photo of a cat"
```

**Python example:**
```python
import requests

# Upload file
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/embeddings/image",
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
curl -X POST "http://localhost:8000/v1/rank" \
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
curl -X POST "http://localhost:8000/v1/classify" \
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
curl -X POST "http://localhost:8000/v1/classify/image" \
  -F "file=@/path/to/image.jpg" \
  -F "labels=cat,dog,bird,fish"
```

**Python example:**
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/classify/image",
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
query_resp = requests.post("http://localhost:8000/v1/embeddings", 
    json={"input": "sunset over mountains"})
query_embedding = np.array(query_resp.json()["data"][0]["embedding"])

# Get embeddings for your image database
image_urls = ["image1.jpg", "image2.jpg", "image3.jpg"]
images_resp = requests.post("http://localhost:8000/v1/embeddings",
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
        "http://localhost:8000/v1/embeddings/image",
        files={"file": f}
    )
    embedding = response.json()["data"][0]["embedding"]

# Upload and classify
with open("local_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/classify/image",
        files={"file": f},
        data={"labels": "cat,dog,bird"}
    )
    classifications = response.json()["classifications"]
```

### Cross-modal Retrieval
Find images that match text descriptions or vice versa:
```python
# Use the ranking endpoint
response = requests.post("http://localhost:8000/v1/rank",
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

## 🔧 Configuration

### Environment Variables

- `LOG_LEVEL`: Logging level (default: INFO)
- `CUDA_VISIBLE_DEVICES`: GPU device ID for CUDA acceleration

### Docker Compose Volumes

- `siglip-model-cache`: Caches downloaded models to avoid re-downloading
- Mount local directories for image access (see docker-compose.yml)

## 🚀 Performance Tips

1. **GPU Acceleration**: Use NVIDIA GPU for faster inference
2. **Batch Processing**: Send multiple inputs in a single request
3. **Model Caching**: The model is cached after first download
4. **Connection Pooling**: Reuse HTTP connections for multiple requests
5. **File Uploads**: Use file uploads for local images instead of base64 encoding

## 🐛 Troubleshooting

### Model Download Issues
If the model fails to download:
```bash
# Clear the cache
docker-compose down -v
docker-compose up -d
```

### Memory Issues
For large batches, you may need to increase Docker memory limits.

### GPU Not Detected
Ensure NVIDIA Docker runtime is installed:
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 📊 Model Information

- **Model**: SigLIP base patch16 256 multilingual
- **Parameters**: ~400M
- **Embedding Dimension**: 768
- **Max Text Length**: 64 tokens
- **Image Resolution**: 256x256
- **Languages**: Multilingual (best performance on English)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Research for the SigLIP model
- Hugging Face for model hosting and libraries
- ONNX Runtime team for optimization tools
