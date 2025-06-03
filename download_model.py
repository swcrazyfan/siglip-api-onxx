#!/usr/bin/env python3
"""
Script to pre-download SigLIP 2 model during Docker build
"""

from transformers import AutoModel, AutoProcessor
import torch
import os

print('Pre-downloading SigLIP 2 model...')
model_name = 'google/siglip2-large-patch16-512'
cache_dir = '/app/cache'

try:
    print('Downloading model...')
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    print('Downloading processor...')
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    print('Model pre-download completed successfully!')
except Exception as e:
    print(f'Warning: Model pre-download failed: {e}')
    print('Model will be downloaded at runtime instead.')
