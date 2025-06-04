# Batch Processing & Auto-Scaling

Your SigLIP API now includes intelligent batch processing and auto-scaling features that dramatically improve performance when handling multiple concurrent requests.

## 🚀 Key Features

### ✅ Intelligent Batching
- **Automatic batching**: Individual requests are automatically grouped into batches for efficient processing
- **Mixed request types**: Text, image, and joint image-text requests are processed in separate optimized batches
- **Smart timing**: Batches are processed either when full or after a brief wait time (50ms default)

### ✅ Auto-Scaling
- **CPU-aware**: Automatically scales batch workers based on your CPU count
- **Smart defaults**: Uses 75% of available CPUs with memory-conscious limits
- **GPU optimization**: Reduces workers when GPU is the bottleneck

### ✅ Robust Queue System
- **No failures**: Requests queue instead of failing during traffic spikes
- **Overflow protection**: Configurable behavior when queue fills up
- **Timeout handling**: Prevents hanging requests with configurable timeouts

## 📊 Performance Benefits

### Before (Single Processing)
- 8 concurrent text requests: ~8 × single_request_time
- High resource waste during low traffic
- Fails under high load

### After (Batch Processing)
- 8 concurrent text requests: ~1 × batch_time (much faster!)
- Efficient resource utilization
- Graceful degradation under load

## ⚙️ Configuration

### Environment Variables

```bash
# Batch Processing
BATCH_SIZE=8                    # Items per batch (default: 8)
BATCH_WAIT_TIME_MS=50          # Max wait to fill batch (default: 50ms)

# Auto-Scaling  
AUTO_SCALE_WORKERS=true        # Enable auto-scaling (default: true)
CPU_UTILIZATION_FACTOR=0.75    # Use 75% of CPUs (default: 0.75)
MAX_CONCURRENT_BATCHES=6       # Override auto-detected worker count

# Queue Management
MAX_QUEUE_SIZE=1000            # Max queued requests (default: 1000)
REQUEST_TIMEOUT=300            # Max request lifetime in seconds (default: 300)
QUEUE_OVERFLOW_ACTION=reject   # "reject" or "wait" (default: reject)
```

### Scaling Examples

Your API automatically scales based on your hardware:

| Hardware | Auto-Detected Workers | Reasoning |
|----------|----------------------|-----------|
| 2 vCPUs  | 1-2 workers         | Conservative for small instances |
| 4 vCPUs  | 3 workers           | 75% utilization |
| 8 vCPUs  | 6 workers           | Balanced performance |
| 16+ vCPUs| 8 workers (capped)  | Memory efficiency limit |

## 🔍 Monitoring

### Health Check Endpoint

```bash
curl http://localhost:8000/health
```

Response includes batch processing stats:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps",
  "batch_processing": {
    "enabled": true,
    "batch_size": 8,
    "batch_wait_time_ms": 50,
    "max_concurrent_batches": 6,
    "auto_scaling": true,
    "cpu_count": 8,
    "queue_size": 12,
    "max_queue_size": 1000,
    "active_workers": 6,
    "total_workers": 6
  }
}
```

### Key Metrics
- **queue_size**: Current requests in queue
- **active_workers**: Number of running batch workers
- **batch_size**: Items processed per batch
- **cpu_count**: Detected CPU cores

## 🧪 Testing

Use the included test script to verify batch processing:

```bash
# Install test dependencies first
pip install aiohttp

# Run test suite
python test_batch.py
```

The test will:
1. Check health and configuration
2. Send 8 concurrent requests
3. Measure total time vs individual request time
4. Verify batching is working

Expected output:
```
🧪 Testing concurrent batch processing...
✅ Request 0: A beautiful sunset over the mo... -> 768 dims
✅ Request 1: A cat sitting on a windowsill... -> 768 dims
...
✅ Completed 8/8 requests
⏱️  Total time: 1.23 seconds
📊 Average time per request: 0.154 seconds
```

## 🔧 Advanced Configuration

### Fine-Tuning for Your Use Case

**High Throughput (Many Concurrent Users)**
```bash
BATCH_SIZE=16
BATCH_WAIT_TIME_MS=25
MAX_QUEUE_SIZE=2000
```

**Low Latency (Real-time Applications)**
```bash
BATCH_SIZE=4
BATCH_WAIT_TIME_MS=10
MAX_CONCURRENT_BATCHES=8
```

**Memory Constrained**
```bash
MAX_CONCURRENT_BATCHES=2
BATCH_SIZE=4
AUTO_SCALE_WORKERS=false
```

### GPU vs CPU Considerations

**With GPU (MPS/CUDA)**
- GPU is usually the bottleneck
- Fewer workers (2-4) often optimal
- Larger batch sizes help GPU utilization

**CPU Only**
- CPU processing can parallelize better
- More workers beneficial
- Batch size less critical

## 📈 Performance Tips

1. **Monitor your queue**: If `queue_size` stays high, consider increasing workers
2. **Watch for timeouts**: High `REQUEST_TIMEOUT` values indicate overload
3. **Batch size tuning**: Larger batches = better GPU utilization but higher latency
4. **Memory usage**: More workers = more memory usage

## 🚨 Troubleshooting

### High Queue Size
- **Cause**: More requests than processing capacity
- **Solution**: Increase `MAX_CONCURRENT_BATCHES` or `BATCH_SIZE`

### Request Timeouts
- **Cause**: System overloaded
- **Solution**: Scale up hardware or reduce `REQUEST_TIMEOUT`

### High Memory Usage
- **Cause**: Too many concurrent workers
- **Solution**: Reduce `MAX_CONCURRENT_BATCHES`

### Slow Response Times
- **Cause**: Batch wait time too high
- **Solution**: Reduce `BATCH_WAIT_TIME_MS`

## 🎯 API Compatibility

**100% Compatible**: All existing API endpoints work exactly the same. Batching happens transparently under the hood.

- ✅ `/v1/embeddings` - Now batched automatically
- ✅ `/v1/rank` - Benefits from batching
- ✅ `/v1/classify` - Improved performance
- ✅ File uploads - Same interface, better performance
- ✅ OpenAI compatibility - Maintained 