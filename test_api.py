#!/usr/bin/env python3
"""
Test script for SigLIP ONNX API
Run this after starting the API to verify it's working correctly
"""

import requests
import time
import sys


def test_endpoint(name: str, method: str, url: str, json_data=None, expected_status=200):
    """Test a single endpoint"""
    print(f"Testing {name}...", end=" ")
    try:
        if method == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, json=json_data)
        
        if response.status_code == expected_status:
            print("✅ PASSED")
            return True, response.json()
        else:
            print(f"❌ FAILED (Status: {response.status_code})")
            print(f"   Response: {response.text}")
            return False, None
    except Exception as e:
        print(f"❌ FAILED (Error: {e})")
        return False, None


def main():
    base_url = "http://localhost:8000"
    
    print("🧪 SigLIP API Test Suite")
    print("=" * 50)
    
    # Wait for API to be ready
    print("Waiting for API to be ready...", end=" ")
    for i in range(30):
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✅ Ready!")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("❌ Timeout waiting for API")
        sys.exit(1)
    
    print()
    
    # Run tests
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Health endpoint
    tests_total += 1
    passed, _ = test_endpoint(
        "Health Check",
        "GET",
        f"{base_url}/health"
    )
    if passed:
        tests_passed += 1
    
    # Test 2: Root endpoint
    tests_total += 1
    passed, _ = test_endpoint(
        "Root Endpoint",
        "GET",
        f"{base_url}/"
    )
    if passed:
        tests_passed += 1
    
    # Test 3: Text embedding
    tests_total += 1
    passed, data = test_endpoint(
        "Text Embedding",
        "POST",
        f"{base_url}/v1/embeddings",
        {"input": "a photo of a cat"}
    )
    if passed:
        tests_passed += 1
        if data and len(data['data'][0]['embedding']) == 768:
            print("   ✓ Correct embedding dimension (768)")
    
    # Test 4: Image URL embedding
    tests_total += 1
    passed, _ = test_endpoint(
        "Image URL Embedding",
        "POST",
        f"{base_url}/v1/embeddings",
        {"input": "https://via.placeholder.com/256"}
    )
    if passed:
        tests_passed += 1
    
    # Test 5: Multiple inputs
    tests_total += 1
    passed, _ = test_endpoint(
        "Multiple Inputs",
        "POST",
        f"{base_url}/v1/embeddings",
        {"input": ["text 1", "text 2", "text 3"]}
    )
    if passed:
        tests_passed += 1
    
    # Test 6: Ranking
    tests_total += 1
    passed, _ = test_endpoint(
        "Ranking Endpoint",
        "POST",
        f"{base_url}/v1/rank",
        {
            "query": "a cat",
            "candidates": ["a dog", "a feline", "a car"]
        }
    )
    if passed:
        tests_passed += 1
    
    # Test 7: Classification
    tests_total += 1
    passed, _ = test_endpoint(
        "Classification Endpoint",
        "POST",
        f"{base_url}/v1/classify",
        {
            "image": "https://via.placeholder.com/256",
            "labels": ["cat", "dog", "car"]
        }
    )
    if passed:
        tests_passed += 1
    
    # Test 8: Invalid input handling
    tests_total += 1
    passed, _ = test_endpoint(
        "Invalid Input Handling",
        "POST",
        f"{base_url}/v1/embeddings",
        {"input": ""},
        expected_status=500
    )
    if passed:
        tests_passed += 1
    
    # Summary
    print()
    print("=" * 50)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
