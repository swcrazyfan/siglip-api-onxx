#!/usr/bin/env python3
"""
Usage examples for SigLIP 2 API
Shows all the different ways to interact with the API
"""

import requests
import json
import base64
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8001"

def test_health():
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_openai_text_embedding():
    """Test OpenAI-compatible text embedding"""
    print("\n📝 Testing OpenAI-compatible text embedding...")
    
    payload = {
        "input": "a beautiful sunset over the ocean",
        "model": "siglip2-base-patch16-512"
    }
    
    response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        embedding = data["data"][0]["embedding"]
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        return True
    else:
        print(f"   Error: {response.text}")
        return False

def test_openai_batch_text_embedding():
    """Test OpenAI-compatible batch text embedding"""
    print("\n📝 Testing batch text embeddings...")
    
    payload = {
        "input": [
            "a photo of a cat", 
            "a photo of a dog", 
            "a red sports car"
        ],
        "model": "siglip2-base-patch16-512"
    }
    
    response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Number of embeddings: {len(data['data'])}")
        print(f"   Each embedding dimension: {len(data['data'][0]['embedding'])}")
        return True
    else:
        print(f"   Error: {response.text}")
        return False

def main():
    """Run all usage examples"""
    print("🚀 SigLIP 2 API Usage Examples\n")
    
    # Test health first
    if not test_health():
        print("❌ Health check failed. Make sure the API is running!")
        return
    
    # Test text embeddings
    test_openai_text_embedding()
    test_openai_batch_text_embedding()
    
    print("\n✅ All basic tests completed!")
    print("\n📖 Additional endpoints available:")
    print("   GET  /v1/models - List available models")
    print("   POST /v1/embeddings/image - Upload image files")
    print("   POST /v1/rank - Rank candidates by similarity") 
    print("   POST /v1/classify - Zero-shot image classification")
    print("   GET  /docs - Interactive API documentation")

if __name__ == "__main__":
    main()
