#!/usr/bin/env python3
"""
Simple test script to verify batch processing functionality
"""

import asyncio
import time
import aiohttp
import json
from concurrent.futures import ThreadPoolExecutor
import requests

API_BASE = "http://localhost:8000"

async def test_concurrent_requests():
    """Test concurrent requests to verify batching"""
    print("🧪 Testing concurrent batch processing...")
    
    # Test data
    test_texts = [
        "A beautiful sunset over the mountains",
        "A cat sitting on a windowsill", 
        "Modern architecture with glass buildings",
        "Fresh fruits on a wooden table",
        "An astronaut floating in space",
        "Ocean waves crashing on the shore",
        "A forest path in autumn",
        "City lights at night"
    ]
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # Send all requests concurrently
        tasks = []
        for i, text in enumerate(test_texts):
            task = send_embedding_request(session, text, i)
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Print results
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        print(f"✅ Completed {len(successful)}/{len(test_texts)} requests")
        print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
        print(f"📊 Average time per request: {(end_time - start_time) / len(test_texts):.3f} seconds")
        
        if failed:
            print(f"❌ Failed requests: {len(failed)}")
            for error in failed:
                print(f"   - {error}")
        
        # Print first embedding shape for verification
        if successful:
            print(f"🔢 Embedding shape: {len(successful[0]['data'][0]['embedding'])}")


async def send_embedding_request(session, text, idx):
    """Send a single embedding request"""
    payload = {
        "model": "siglip2-base-patch16-512",
        "input": text
    }
    
    try:
        async with session.post(f"{API_BASE}/v1/embeddings", json=payload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Request {idx}: {text[:30]}... -> {len(result['data'][0]['embedding'])} dims")
                return result
            else:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")
    except Exception as e:
        print(f"❌ Request {idx} failed: {e}")
        raise


def test_health_check():
    """Test health check endpoint"""
    print("\n🏥 Testing health check...")
    
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print("✅ Health check successful")
            print(f"   Model loaded: {health.get('model_loaded')}")
            print(f"   Device: {health.get('device')}")
            
            if 'batch_processing' in health:
                bp = health['batch_processing']
                print(f"   Batch processing enabled: {bp.get('enabled')}")
                print(f"   Batch size: {bp.get('batch_size')}")
                print(f"   Workers: {bp.get('active_workers', 0)}/{bp.get('total_workers', 0)}")
                print(f"   Queue: {bp.get('queue_size', 0)}/{bp.get('max_queue_size', 0)}")
                print(f"   Auto-scaling: {bp.get('auto_scaling')}")
                print(f"   CPU count: {bp.get('cpu_count')}")
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")


def test_root_endpoint():
    """Test root endpoint"""
    print("\n📋 Testing root endpoint...")
    
    try:
        response = requests.get(f"{API_BASE}/", timeout=10)
        if response.status_code == 200:
            root = response.json()
            print("✅ Root endpoint successful")
            print(f"   API: {root.get('name')} v{root.get('version')}")
            print(f"   Model: {root.get('model')}")
            
            if 'features' in root:
                print("   Features:")
                for feature in root['features']:
                    print(f"     • {feature}")
            
            if 'batch_config' in root:
                bc = root['batch_config']
                print(f"   Batch config: {bc}")
        else:
            print(f"❌ Root endpoint failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")


async def main():
    """Main test function"""
    print("🚀 SigLIP Batch Processing Test Suite")
    print("=" * 50)
    
    # Test basic endpoints first
    test_health_check()
    test_root_endpoint()
    
    print("\n" + "=" * 50)
    
    # Test concurrent batch processing
    await test_concurrent_requests()
    
    print("\n🎉 Test suite completed!")


if __name__ == "__main__":
    asyncio.run(main()) 