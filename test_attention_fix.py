#!/usr/bin/env python3
"""
Test script to validate the attention_mask fix
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_attention_mask_fix():
    """Test that the attention_mask fix works"""
    try:
        print("🔧 Testing attention_mask fix...")
        
        # Import functions
        from app import load_models, get_text_embedding, get_image_embedding, device, model, processor
        
        # Load models
        await load_models()
        print(f"✅ Models loaded on device: {device}")
        
        # Test text embedding (this was the main source of attention_mask errors)
        print("\n📝 Testing text embedding...")
        text_embedding, metadata = await get_text_embedding("a cat sitting on a chair")
        print(f"   ✅ Text embedding successful! Shape: {text_embedding.shape}")
        print(f"   Metadata: {metadata}")
        
        # Test that we can create embeddings without errors
        print(f"   First 5 values: {text_embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ attention_mask test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🚀 Testing SigLIP 2 attention_mask fixes...\n")
    
    success = await test_attention_mask_fix()
    
    if success:
        print("\n🎉 attention_mask fix successful! No more ONNX input errors.")
    else:
        print("\n💥 attention_mask test failed.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
