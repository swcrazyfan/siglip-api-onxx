#!/usr/bin/env python3
"""
Example client for SigLIP ONNX API
Demonstrates various use cases and features including file uploads
"""

import requests
import numpy as np
import base64
from typing import List, Dict, Any
import json
from pathlib import Path


class SigLIPClient:
    """Client for interacting with SigLIP API"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def get_embeddings(self, inputs: List[str] | str) -> Dict[str, Any]:
        """Get embeddings for text or images"""
        if isinstance(inputs, str):
            inputs = [inputs]
        
        response = self.session.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": inputs}
        )
        response.raise_for_status()
        return response.json()
    
    def get_embeddings_from_file(self, file_path: str) -> Dict[str, Any]:
        """Get embeddings by uploading a file"""
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'image/jpeg')}
            response = self.session.post(
                f"{self.base_url}/v1/embeddings/image",
                files=files
            )
        response.raise_for_status()
        return response.json()
    
    def get_embeddings_with_form(self, text_or_url: str) -> Dict[str, Any]:
        """Get embeddings using form data (alternative to JSON)"""
        response = self.session.post(
            f"{self.base_url}/v1/embeddings/image",
            data={"input": text_or_url}
        )
        response.raise_for_status()
        return response.json()
    
    def rank(self, query: str | List[str], candidates: List[str]) -> Dict[str, Any]:
        """Rank candidates by similarity to query"""
        response = self.session.post(
            f"{self.base_url}/v1/rank",
            json={
                "query": query,
                "candidates": candidates
            }
        )
        response.raise_for_status()
        return response.json()
    
    def classify(self, image: str, labels: List[str]) -> Dict[str, Any]:
        """Classify an image with given labels"""
        response = self.session.post(
            f"{self.base_url}/v1/classify",
            json={
                "image": image,
                "labels": labels
            }
        )
        response.raise_for_status()
        return response.json()
    
    def classify_file(self, file_path: str, labels: List[str]) -> Dict[str, Any]:
        """Classify an uploaded image file"""
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'image/jpeg')}
            data = {'labels': ','.join(labels)}
            response = self.session.post(
                f"{self.base_url}/v1/classify/image",
                files=files,
                data=data
            )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


def image_to_base64(image_path: str) -> str:
    """Convert local image to base64 data URI"""
    with open(image_path, "rb") as f:
        image_data = f.read()
    base64_str = base64.b64encode(image_data).decode('utf-8')
    
    # Detect image type from extension
    if image_path.lower().endswith('.png'):
        mime_type = "image/png"
    elif image_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = "image/jpeg"
    elif image_path.lower().endswith('.gif'):
        mime_type = "image/gif"
    elif image_path.lower().endswith('.webp'):
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"  # default
    
    return f"data:{mime_type};base64,{base64_str}"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def save_test_image(url: str, filename: str = "test_image.jpg"):
    """Download and save a test image"""
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(response.content)
    return filename


def main():
    # Initialize client
    client = SigLIPClient()
    
    # Check health
    print("🏥 Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    print()
    
    # Example 1: Text embeddings (standard)
    print("📝 Example 1: Text Embeddings")
    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a beautiful sunset"
    ]
    response = client.get_embeddings(texts)
    print(f"Generated {len(response['data'])} embeddings")
    print(f"Embedding dimension: {len(response['data'][0]['embedding'])}")
    print()
    
    # Example 2: Image embedding from URL (standard)
    print("🌐 Example 2: Image Embedding from URL")
    image_url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
    response = client.get_embeddings(image_url)
    print(f"Generated embedding for image URL")
    print(f"Usage: {response['usage']}")
    print()
    
    # Example 3: File upload (new feature!)
    print("📤 Example 3: Image File Upload")
    # Download a test image
    test_image_path = save_test_image(image_url, "test_cat.jpg")
    
    try:
        response = client.get_embeddings_from_file(test_image_path)
        print(f"Generated embedding for uploaded file: {test_image_path}")
        print(f"Embedding dimension: {len(response['data'][0]['embedding'])}")
    except Exception as e:
        print(f"Error uploading file: {e}")
    print()
    
    # Example 4: Form-based input (alternative to JSON)
    print("📋 Example 4: Form-based Input")
    response = client.get_embeddings_with_form("a fluffy kitten")
    print(f"Generated embedding using form data")
    print()
    
    # Example 5: Mixed inputs
    print("🎨 Example 5: Mixed Text and Image Inputs")
    mixed_inputs = [
        "a cute kitten",
        image_url,
        "a playful puppy"
    ]
    response = client.get_embeddings(mixed_inputs)
    
    # Calculate similarities
    embeddings = [np.array(d['embedding']) for d in response['data']]
    text1_img_sim = cosine_similarity(embeddings[0], embeddings[1])
    text2_img_sim = cosine_similarity(embeddings[2], embeddings[1])
    
    print(f"Similarity between 'a cute kitten' and image: {text1_img_sim:.3f}")
    print(f"Similarity between 'a playful puppy' and image: {text2_img_sim:.3f}")
    print()
    
    # Example 6: Zero-shot classification with URL
    print("🎯 Example 6: Zero-shot Image Classification (URL)")
    labels = ["cat", "dog", "bird", "fish", "hamster"]
    classification = client.classify(image_url, labels)
    
    print("Classifications:")
    for item in classification['classifications']:
        print(f"  {item['label']}: {item['score']:.3f}")
    print()
    
    # Example 7: Zero-shot classification with file upload
    print("📁 Example 7: Zero-shot Classification with File Upload")
    try:
        classification = client.classify_file(test_image_path, labels)
        print(f"Classifications for {classification['filename']}:")
        for item in classification['classifications']:
            print(f"  {item['label']}: {item['score']:.3f}")
    except Exception as e:
        print(f"Error classifying file: {e}")
    print()
    
    # Example 8: Image search
    print("🔍 Example 8: Image Search (Text -> Images)")
    image_urls = [
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",  # cat
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # person
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400"   # mountains
    ]
    
    ranking = client.rank("a cute cat", image_urls)
    
    print("Search results for 'a cute cat':")
    for i, result in enumerate(ranking['results'][0]['rankings'], 1):
        print(f"  {i}. Score: {result['score']:.3f} - {result['candidate']}")
    print()
    
    # Example 9: Handling base64 images
    print("🖼️ Example 9: Base64 Image Input")
    if Path(test_image_path).exists():
        base64_image = image_to_base64(test_image_path)
        response = client.get_embeddings(base64_image)
        print(f"Generated embedding for base64 image")
        print(f"Base64 string length: {len(base64_image)} characters")
    print()
    
    # Example 10: Demonstrating text truncation
    print("✂️ Example 10: Text Truncation Warning")
    long_text = " ".join(["word"] * 100)  # 100 words, will be truncated
    response = client.get_embeddings(long_text)
    print("Check server logs for truncation warning")
    print()
    
    # Cleanup
    if Path(test_image_path).exists():
        Path(test_image_path).unlink()
        print(f"🧹 Cleaned up test image: {test_image_path}")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Make sure the server is running on http://localhost:8001")
    except Exception as e:
        print(f"❌ Error: {e}")
