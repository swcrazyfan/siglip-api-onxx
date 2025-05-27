#!/usr/bin/env python3
"""
Quick test script for SigLIP 2 API
Tests basic functionality without running the full server
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_model_loading():
    """Test that models can be loaded"""
    try:
        print("Testing SigLIP 2 model loading...")
        
        # Import the load_models function
        from app import load_models, device, model, processor
        
        # Load models
        await load_models()
        
        print(f"✅ Models loaded successfully!")
        print(f"   Device: {device}")
        print(f"   Model loaded: {model is not None}")
        print(f"   Processor loaded: {processor is not None}")
        
        # Test basic text embedding
        if model and processor:
            from app import get_text_embedding
            print("\n📝 Testing text embedding...")
            
            text_embedding, metadata = await get_text_embedding("a cat sitting on a chair")
            print(f"   Text embedding shape: {text_embedding.shape}")
            print(f"   Metadata: {metadata}")
            
            print("✅ Text embedding test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting SigLIP 2 API tests...\n")
    
    success = await test_model_loading()
    
    if success:
        print("\n🎉 All tests passed! The API should work correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
