#!/usr/bin/env python3
"""
Test script to verify model selection via environment variable
Run this locally to test different models
"""

import os
import subprocess
import time
import requests
import signal
import sys

def test_model_loading(model_name):
    """Test loading a specific model"""
    print(f"\n🧪 Testing {model_name}...")
    
    # Set environment variable
    env = os.environ.copy()
    env['SIGLIP_MODEL'] = model_name
    env['SIGLIP_VERBOSE_OUTPUT'] = 'false'  # Reduce noise
    
    # Start the app
    print("   Starting API server...")
    process = subprocess.Popen(
        ['uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8002'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for startup
        print("   Waiting for server to start...")
        time.sleep(10)  # Give it time to load the model
        
        # Test health endpoint
        for attempt in range(5):
            try:
                response = requests.get("http://localhost:8002/health", timeout=5)
                if response.status_code == 200:
                    print("   ✅ Server is healthy")
                    break
            except requests.RequestException:
                if attempt == 4:
                    print("   ❌ Server failed to start")
                    return False
                time.sleep(2)
        
        # Test model info
        response = requests.get("http://localhost:8002/v1/models")
        if response.status_code == 200:
            model_data = response.json()
            loaded_model = model_data["data"][0]["id"]
            print(f"   📋 Loaded model: {loaded_model}")
        
        # Test embedding
        response = requests.post(
            "http://localhost:8002/v1/embeddings",
            json={"input": "test text"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            embedding_dim = len(data["data"][0]["embedding"])
            print(f"   📐 Embedding dimension: {embedding_dim}")
            print("   ✅ Model working correctly")
            return True
        else:
            print(f"   ❌ Embedding test failed: {response.text}")
            return False
            
    finally:
        # Cleanup
        print("   🧹 Stopping server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

def main():
    """Test both models"""
    print("🔬 Model Selection Test Suite")
    print("=" * 50)
    
    models_to_test = [
        "google/siglip2-base-patch16-512",
        "google/siglip2-large-patch16-512"
    ]
    
    results = {}
    
    for model in models_to_test:
        success = test_model_loading(model)
        results[model] = success
    
    print("\n📊 Test Results:")
    print("=" * 50)
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model}: {status}")
    
    if all(results.values()):
        print("\n🎉 All tests passed! Model selection is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the logs above.")

if __name__ == "__main__":
    # Check if uvicorn is available
    try:
        subprocess.run(['uvicorn', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uvicorn not found. Please install with: pip install uvicorn")
        sys.exit(1)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Test interrupted by user")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    main() 