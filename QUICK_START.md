# Quick Start Guide for Your Hardware

**Your Setup**: AMD Ryzen 7 5700X (8 cores/16 threads) + RTX 3060 12GB

## 🚀 Super Simple Usage

### One Image, Works Everywhere
```bash
# CPU-only (auto-detects your Ryzen 7 5700X)
docker-compose up -d

# With GPU (auto-detects your RTX 3060)
docker run --gpus all -p 8000:8000 siglip2-api

# Check what hardware was detected
curl http://localhost:8000/health | jq .hardware
```

**The image automatically detects and optimizes for:**
- ✅ **RTX 3060**: 2 workers, batch size 16, 20-50 images/sec
- ✅ **Ryzen CPU**: 6 workers, batch size 6, 2-8 images/sec
- ✅ **No configuration needed!**

## 🔍 Verify Everything Works

```bash
# Start with GPU
docker run --gpus all -p 8000:8000 siglip2-api

# Check hardware detection
curl http://localhost:8000/health

# Should show:
# "gpu_name": "NVIDIA GeForce RTX 3060"
# "gpu_memory_gb": 12
# "device_type": "cuda"

# Test batch processing
python test_batch.py
```

## ⚙️ Fine-Tuning Options

### Maximize GPU Performance
```bash
# Larger batches for your RTX 3060
docker run --gpus all -e BATCH_SIZE=20 -p 8000:8000 siglip2-api
```

### Share Resources with Other Apps
```bash
# Limit to 6 cores + GPU
docker run --gpus all --cpus="6" -p 8000:8000 siglip2-api
```

### CPU-Only with Limits
```bash
# Use only 4 CPU cores
docker run --cpus="4" -e MAX_CONCURRENT_BATCHES=3 -p 8000:8000 siglip2-api
```

## 📊 Expected Performance on Your Hardware

| Mode | Images/Sec | Response Time | Memory Usage |
|------|------------|---------------|--------------|
| **RTX 3060** | **20-50** | **50-200ms** | **~4-6GB GPU** |
| **Ryzen CPU** | **2-8** | **500-2000ms** | **System RAM** |

## 🎯 Recommended Commands

```bash
# Best performance (auto-detects everything)
docker run --gpus all -p 8000:8000 siglip2-api

# With resource limits for shared usage
docker run --gpus all --cpus="6" --memory="8g" -p 8000:8000 siglip2-api

# Quick test
docker-compose up -d && curl http://localhost:8000/health
```

**That's it!** No profiles, no complexity - just works! 🚀 