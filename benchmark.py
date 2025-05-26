#!/usr/bin/env python3
"""
Benchmark script for SigLIP ONNX API
Measures latency and throughput for different input types
"""

import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


def benchmark_single_request(url: str, payload: dict) -> float:
    """Time a single request"""
    start = time.time()
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return time.time() - start


def benchmark_endpoint(name: str, url: str, payload: dict, iterations: int = 20):
    """Benchmark an endpoint with multiple iterations"""
    print(f"\n📊 Benchmarking {name}")
    print("=" * 50)
    
    # Warmup
    print("Warming up...", end=" ")
    for _ in range(3):
        benchmark_single_request(url, payload)
    print("Done")
    
    # Benchmark
    times = []
    print(f"Running {iterations} iterations...", end=" ")
    
    for _ in range(iterations):
        duration = benchmark_single_request(url, payload)
        times.append(duration)
    
    print("Done")
    
    # Calculate statistics
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0
    
    print(f"Average latency: {avg_time*1000:.2f} ms")
    print(f"Min latency: {min_time*1000:.2f} ms")
    print(f"Max latency: {max_time*1000:.2f} ms")
    print(f"Std deviation: {std_dev*1000:.2f} ms")
    print(f"Throughput: {1/avg_time:.2f} requests/second")
    
    return times


def benchmark_concurrent(name: str, url: str, payload: dict, workers: int = 5, total_requests: int = 50):
    """Benchmark with concurrent requests"""
    print(f"\n🚀 Concurrent Benchmark: {name}")
    print(f"Workers: {workers}, Total requests: {total_requests}")
    print("=" * 50)
    
    completed_requests = 0
    errors = 0
    times = []
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for _ in range(total_requests):
            future = executor.submit(benchmark_single_request, url, payload)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                duration = future.result()
                times.append(duration)
                completed_requests += 1
            except Exception as e:
                errors += 1
    
    total_time = time.time() - start_time
    
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Completed requests: {completed_requests}")
    print(f"Failed requests: {errors}")
    print(f"Overall throughput: {completed_requests/total_time:.2f} requests/second")
    
    if times:
        print(f"Average latency: {statistics.mean(times)*1000:.2f} ms")


def main():
    base_url = "http://localhost:8001"
    embeddings_url = f"{base_url}/v1/embeddings"
    
    print("🧪 SigLIP ONNX API Benchmark")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        print("✅ API is running")
    except:
        print("❌ API is not running. Please start it first.")
        return
    
    # Benchmark different input types
    
    # 1. Short text
    benchmark_endpoint(
        "Short Text (10 words)",
        embeddings_url,
        {"input": "The quick brown fox jumps over the lazy dog today"}
    )
    
    # 2. Maximum text (will be truncated)
    long_text = " ".join(["word"] * 100)
    benchmark_endpoint(
        "Long Text (100 words, truncated)",
        embeddings_url,
        {"input": long_text}
    )
    
    # 3. Image URL
    benchmark_endpoint(
        "Image URL",
        embeddings_url,
        {"input": "https://via.placeholder.com/256"}
    )
    
    # 4. Batch processing
    benchmark_endpoint(
        "Batch (5 texts)",
        embeddings_url,
        {"input": ["text 1", "text 2", "text 3", "text 4", "text 5"]}
    )
    
    # 5. Mixed batch
    benchmark_endpoint(
        "Mixed Batch (3 texts + 2 images)",
        embeddings_url,
        {"input": [
            "a cat",
            "https://via.placeholder.com/256",
            "a dog",
            "https://via.placeholder.com/256",
            "a bird"
        ]}
    )
    
    # Concurrent benchmarks
    print("\n" + "="*50)
    print("CONCURRENT BENCHMARKS")
    print("="*50)
    
    # Single text, concurrent
    benchmark_concurrent(
        "Text Embeddings",
        embeddings_url,
        {"input": "concurrent test"},
        workers=5,
        total_requests=50
    )
    
    # Batch, concurrent
    benchmark_concurrent(
        "Batch Embeddings",
        embeddings_url,
        {"input": ["text 1", "text 2", "text 3"]},
        workers=3,
        total_requests=30
    )
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
