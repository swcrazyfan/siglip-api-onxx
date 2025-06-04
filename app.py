"""
SigLIP 2 API - Fast multimodal embeddings with OpenAI-compatible endpoints
Uses configurable SigLIP 2 model with PyTorch backend
Supports text and image inputs with automatic detection
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
import requests
from io import BytesIO
import logging
import base64
import re
from pathlib import Path
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager
import json
import os
from asyncio import Queue, Semaphore
import time
from dataclasses import dataclass
from typing import Callable
import uuid
import multiprocessing

# Verbose output toggle
# Controlled by environment variable SIGLIP_VERBOSE_OUTPUT (e.g., "true" or "false")
# Defaults to true (verbose output enabled).
VERBOSE_OUTPUT_ENV = os.getenv("SIGLIP_VERBOSE_OUTPUT", "true").lower()
VERBOSE_OUTPUT = VERBOSE_OUTPUT_ENV == "true"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if VERBOSE_OUTPUT else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
if VERBOSE_OUTPUT:
    logger.info("Verbose output enabled (DEBUG level logging). To disable, set environment variable SIGLIP_VERBOSE_OUTPUT=false.")
else:
    logger.info("Verbose output disabled (INFO level logging). To enable, set environment variable SIGLIP_VERBOSE_OUTPUT=true.")

# Global variables for models
_model_load_lock = asyncio.Lock()  # Lock for synchronizing model loading
model = None
processor = None
device = None

# Model configuration
MODEL_NAME = os.getenv("SIGLIP_MODEL", "google/siglip2-base-patch16-512")
MAX_TEXT_TOKENS = 64

# Log which model is being used
logger.info(f"Using model: {MODEL_NAME}")

# Auto-scaling configuration
cpu_count = multiprocessing.cpu_count()
AUTO_SCALE_WORKERS = os.getenv("AUTO_SCALE_WORKERS", "true").lower() == "true"
CPU_UTILIZATION_FACTOR = float(os.getenv("CPU_UTILIZATION_FACTOR", "0.75"))

# Enhanced hardware detection for Docker and GPU environments
def detect_hardware_config():
    """Detect available hardware and return optimal configuration"""
    config = {
        "cpu_count": cpu_count,
        "has_gpu": False,
        "gpu_memory": 0,
        "gpu_name": "none",
        "device_type": "cpu",
        "recommended_workers": 2,
        "recommended_batch_size": 8
    }
    
    # Try to detect GPU first (prioritize GPU if available)
    try:
        import torch
        if torch.cuda.is_available():
            config["has_gpu"] = True
            config["device_type"] = "cuda"
            config["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # GB
            config["gpu_name"] = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {config['gpu_name']} with {config['gpu_memory']}GB")
            
            # GPU-optimized settings (fewer workers, larger batches)
            if config["gpu_memory"] >= 8:  # 8GB+ GPU (like RTX 3060 12GB)
                config["recommended_workers"] = 2  # GPU is the bottleneck, not CPU
                config["recommended_batch_size"] = 16  # Larger batches for GPU efficiency
            else:  # Smaller GPU
                config["recommended_workers"] = 1
                config["recommended_batch_size"] = 8
                
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config["has_gpu"] = True
            config["device_type"] = "mps"
            config["gpu_name"] = "Apple Silicon"
            logger.info("Apple Silicon MPS detected")
            
            # MPS-optimized settings
            config["recommended_workers"] = 2
            config["recommended_batch_size"] = 12
            
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")
    
    # CPU-only configuration (especially for image embeddings)
    if not config["has_gpu"]:
        config["device_type"] = "cpu"
        logger.info(f"No GPU detected, using CPU-only with {cpu_count} cores")
        
        # For image embeddings on CPU, we need more workers but smaller batches
        # Since images are much slower than text
        if cpu_count <= 4:
            config["recommended_workers"] = max(1, cpu_count - 1)  # Leave 1 core for system
            config["recommended_batch_size"] = 4
        elif cpu_count <= 8:
            config["recommended_workers"] = max(2, int(cpu_count * 0.75))
            config["recommended_batch_size"] = 6
        else:  # 8+ cores
            config["recommended_workers"] = max(4, min(int(cpu_count * 0.75), 8))  # Cap at 8 for memory
            config["recommended_batch_size"] = 8
    
    return config

# Get hardware configuration
hardware_config = detect_hardware_config()

if AUTO_SCALE_WORKERS:
    # Use hardware-aware defaults
    default_workers = hardware_config["recommended_workers"]
    default_batch_size = hardware_config["recommended_batch_size"]
else:
    # Conservative defaults
    default_workers = 2
    default_batch_size = 8

# Batch processing configuration with hardware-aware defaults
BATCH_SIZE = int(os.getenv("BATCH_SIZE", str(default_batch_size)))
BATCH_WAIT_TIME_MS = int(os.getenv("BATCH_WAIT_TIME_MS", "50"))
MAX_CONCURRENT_BATCHES = int(os.getenv("MAX_CONCURRENT_BATCHES", str(default_workers)))

# Adjust batch wait time based on hardware
if hardware_config["has_gpu"]:
    # GPU can handle larger batches efficiently, so wait a bit longer to fill them
    BATCH_WAIT_TIME_MS = int(os.getenv("BATCH_WAIT_TIME_MS", "75"))
else:
    # CPU benefits from faster response, shorter wait times
    BATCH_WAIT_TIME_MS = int(os.getenv("BATCH_WAIT_TIME_MS", "35"))

# Queue configuration  
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "1000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes default
QUEUE_OVERFLOW_ACTION = os.getenv("QUEUE_OVERFLOW_ACTION", "reject")  # or "wait"

logger.info(f"Hardware configuration detected:")
logger.info(f"  CPU cores: {hardware_config['cpu_count']}")
logger.info(f"  GPU available: {hardware_config['has_gpu']}")
if hardware_config["has_gpu"]:
    logger.info(f"  GPU: {hardware_config['gpu_name']}")
    if hardware_config["gpu_memory"] > 0:
        logger.info(f"  GPU memory: {hardware_config['gpu_memory']}GB")
logger.info(f"  Device type: {hardware_config['device_type']}")
logger.info(f"Batch processing configuration:")
logger.info(f"  Auto-scaling: {AUTO_SCALE_WORKERS}")
logger.info(f"  Batch workers: {MAX_CONCURRENT_BATCHES}")
logger.info(f"  Batch size: {BATCH_SIZE}")
logger.info(f"  Batch wait time: {BATCH_WAIT_TIME_MS}ms")
logger.info(f"  Max queue size: {MAX_QUEUE_SIZE}")

# Special logging for image embedding workloads
if not hardware_config["has_gpu"]:
    logger.info(f"🖼️  Image embedding performance estimate:")
    logger.info(f"    Expected throughput: {MAX_CONCURRENT_BATCHES * 0.3:.1f}-{MAX_CONCURRENT_BATCHES * 1.0:.1f} images/sec")
    logger.info(f"    Recommendation: Enable GPU with --gpus all for 10-50x speedup")
else:
    logger.info(f"🚀 GPU acceleration enabled - expect 10-50x faster image embeddings")

# Queue system
request_queue: Optional[Queue] = None
processing_semaphore: Optional[Semaphore] = None
batch_workers: List[asyncio.Task] = []

@dataclass
class QueuedRequest:
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    future: asyncio.Future
    created_at: float


class JointInputItem(BaseModel):
    image: str = Field(..., description="Image URL, base64 image, or file path for the image part of the joint input.")
    text: str = Field(..., description="Text to associate with the image for the joint input.")
    # embedding_preference: Optional[str] = Field(default="text_embeds", description="Which embedding to return for joint: 'text_embeds' or 'image_embeds'")

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    model: str = Field(default=MODEL_NAME.split("/")[-1])  # Use just the model name part
    input: Union[str, List[Union[str, JointInputItem]]] = Field(..., description="A single string (text/image URL/base64/file path), or a list containing strings and/or joint image-text objects.")
    encoding_format: Optional[str] = Field(default="float", description="Format of the embeddings")


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]


class RankRequest(BaseModel):
    """Request model for ranking/similarity"""
    model: str = Field(default=MODEL_NAME.split("/")[-1])  # Use just the model name part
    query: Union[str, List[str]] = Field(..., description="Query text or image(s)")
    candidates: List[str] = Field(..., description="Candidates to rank")
    return_scores: Optional[bool] = Field(default=True, description="Return similarity scores")


class RankResponse(BaseModel):
    """Response model for ranking results"""
    results: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]


class ClassifyRequest(BaseModel):
    """Request model for zero-shot classification"""
    image: str = Field(..., description="Image URL, base64, or file path")
    labels: List[str] = Field(..., description="Classification labels")
    model: str = Field(default=MODEL_NAME.split("/")[-1])  # Use just the model name part


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    await load_models()
    yield
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="SigLIP 2 Embeddings API",
    description=f"Fast multimodal embeddings with OpenAI-compatible endpoints using {MODEL_NAME}",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def load_models():
    """Load SigLIP 2 model and processor"""
    global model, processor, device
    
    async with _model_load_lock:  # Ensure only one coroutine loads models at a time
        # Check if models are already loaded after acquiring the lock
        if all(component is not None for component in [model, processor, device]):
            logger.debug("Models already loaded by another coroutine")
            return
            
        logger.info("Loading SigLIP 2 model and processor...")
        
        try:
            # Determine device with proper order: MPS > CUDA > CPU
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS (Apple Silicon) device")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
            
            # Load model with appropriate settings for the device
            if device.type == "cuda":
                model = AutoModel.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16,  # Use half precision for GPU
                    device_map="auto"
                )
            else:
                model = AutoModel.from_pretrained(MODEL_NAME)
                model.to(device)
            
            # Load processor
            processor = AutoProcessor.from_pretrained(MODEL_NAME)
            
            # Set model to evaluation mode
            model.eval()
            
            logger.info(f"Successfully loaded SigLIP 2 model: {MODEL_NAME}")
            logger.info(f"Model device: {next(model.parameters()).device}")
            logger.info(f"Model dtype: {next(model.parameters()).dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise


async def ensure_models_loaded():
    """Ensure all required model components are loaded"""
    global model, processor, device
    
    if any(component is None for component in [model, processor, device]):
        logger.info("One or more model components not loaded, initializing...")
        await load_models()
    else:
        logger.debug("All model components already loaded")


def detect_input_type(input_str: str) -> str:
    """
    Detect whether input is a URL, base64 image, file path, or text.
    Returns: 'url', 'base64', 'file', or 'text'
    """
    # Check if it's a URL
    if input_str.startswith(('http://', 'https://')):
        return 'url'
    
    # Check if it's a base64 image
    if is_base64_image(input_str):
        return 'base64'
    
    # Check if it's a file path
    if len(input_str) < 500:  # Reasonable path length
        try:
            path = Path(input_str)
            if path.exists() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}:
                return 'file'
        except:
            pass
    
    # Default to text
    return 'text'


def is_base64_image(s: str) -> bool:
    """Check if string is a base64-encoded image"""
    # Check for data URI format
    if s.startswith('data:image/'):
        return True
    
    # Check if it's likely raw base64
    if len(s) > 200:  # Images are rarely < 200 chars in base64
        s_clean = s.strip()
        base64_pattern = re.compile(r'^[A-Za-z0-9+/]+={0,2}$')
        
        if base64_pattern.match(s_clean):
            try:
                # Decode first few bytes to check magic numbers
                sample = base64.b64decode(s_clean[:64] + '==', validate=True)
                
                # Check for common image magic numbers
                if sample.startswith((
                    b'\xff\xd8\xff',  # JPEG
                    b'\x89PNG',       # PNG
                    b'GIF87a',        # GIF
                    b'GIF89a',        # GIF
                    b'RIFF',          # WEBP
                    b'BM',            # BMP
                )):
                    return True
            except:
                pass
    
    return False


def decode_base64_image(base64_str: str) -> bytes:
    """Decode base64 image string"""
    if base64_str.startswith('data:image/'):
        base64_str = base64_str.split(',', 1)[1]
    return base64.b64decode(base64_str)


async def load_image_from_input(input_str: str) -> Image.Image:
    """Load image from URL, base64, or file path"""
    input_type = detect_input_type(input_str)
    
    if input_type == 'url':
        response = await asyncio.to_thread(requests.get, input_str, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    
    elif input_type == 'base64':
        image_bytes = decode_base64_image(input_str)
        return Image.open(BytesIO(image_bytes)).convert('RGB')
    
    elif input_type == 'file':
        return await asyncio.to_thread(lambda p: Image.open(p).convert('RGB'), input_str)
    
    else:
        raise ValueError(f"Input appears to be text, not an image")


async def load_image_from_upload(file: UploadFile) -> Image.Image:
    """Load image from uploaded file"""
    contents = await file.read()
    return Image.open(BytesIO(contents)).convert('RGB')


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length"""
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / (norm + 1e-12)


def preprocess_texts(texts: List[str]) -> List[str]:
    """Preprocess text queries using SigLIP 2 prompt template"""
    return [f'This is a photo of {text}.' for text in texts]


async def get_embedding(input_str: str) -> tuple[np.ndarray, dict]:
    """Get embedding for any input type"""
    await ensure_models_loaded()
    
    input_type = detect_input_type(input_str)
    
    if input_type in ['url', 'base64', 'file']:
        try:
            return await get_image_embedding(input_str)
        except Exception as e:
            logger.warning(f"Failed to process as image: {e}, falling back to text")
            return await get_text_embedding(input_str)
    else:
        return await get_text_embedding(input_str)


async def get_image_embedding(image_input: Union[str, Image.Image]) -> tuple[np.ndarray, dict]:
    """Get embedding for an image with metadata"""
    await ensure_models_loaded()
    
    if isinstance(image_input, str):
        image = await load_image_from_input(image_input)
    else:
        image = image_input
    
    # Process image with SigLIP 2 processor
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get image features using SigLIP 2 - only pass expected arguments
    with torch.no_grad():
        image_features = model.get_image_features(
            pixel_values=inputs.get('pixel_values')
        )
        embedding = image_features[0].cpu().numpy()
    
    # Normalize embedding
    embedding = normalize_embeddings(embedding).squeeze()
    
    logger.debug(f"Image embedding shape: {embedding.shape}")
    
    return embedding, {"input_type": "image", "truncated": False}


async def get_joint_embedding(image_input_str: str, text_input_str: str, warn_text_truncation: bool = True) -> tuple[np.ndarray, dict]:
    """Get a joint embedding for an image and text pair."""
    await ensure_models_loaded()

    # 1. Process Image
    if isinstance(image_input_str, str):
        image = await load_image_from_input(image_input_str)
    else:
        logger.warning("get_joint_embedding received non-string image_input_str, attempting to use as PIL.Image")
        image = image_input_str
        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image_input_str type for joint embedding if not a string.")

    # 2. Preprocess text with SigLIP 2 prompt template
    processed_text = preprocess_texts([text_input_str])

    # 3. Use processor with BOTH image and text - SigLIP 2 approach
    inputs = processor(
        text=processed_text,
        images=image,
        padding="max_length",
        max_length=MAX_TEXT_TOKENS,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    logger.debug(f"Joint processor returned keys: {list(inputs.keys())}")
    
    # Get processed token count for metadata
    processed_text_token_count = inputs['input_ids'].shape[1] if 'input_ids' in inputs else MAX_TEXT_TOKENS
    
    # Check for text truncation (simple estimation)
    original_length = len(text_input_str.split())  # Simple word count
    text_metadata = {"truncated": original_length > MAX_TEXT_TOKENS}
    
    if text_metadata["truncated"] and warn_text_truncation:
        logger.warning(f"Joint input: Text part may have been truncated. Original ~{original_length} words.")

    # 4. Run SigLIP 2 inference to get both embeddings
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.get('input_ids'),
            attention_mask=inputs.get('attention_mask'),
            pixel_values=inputs.get('pixel_values')
        )
        
        # Get text embeddings (conditioned on image)
        text_embeds = outputs.text_embeds[0].cpu().numpy()
    
    # Normalize embedding
    embedding = normalize_embeddings(text_embeds).squeeze()

    # 5. Prepare final metadata
    final_metadata = {
        "input_type": "joint_image_text",
        "text_truncated": text_metadata["truncated"],
        "processed_text_tokens": processed_text_token_count,
        "image_tokens": 1,  # SigLIP treats image as single token conceptually
        "total_processed_tokens": processed_text_token_count + 1
    }
    
    return embedding, final_metadata


async def get_text_embedding(text: str, warn_truncation: bool = True) -> tuple[np.ndarray, dict]:
    """Get embedding for text with truncation warning"""
    await ensure_models_loaded()
    
    metadata = {"input_type": "text", "truncated": False}
    
    # Preprocess text with SigLIP 2 prompt template
    processed_text = preprocess_texts([text])
    
    # Process text with SigLIP 2 - IMPORTANT: padding="max_length" and max_length=64
    inputs = processor(
        text=processed_text,
        padding="max_length",
        max_length=MAX_TEXT_TOKENS,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Update metadata
    processed_token_count = inputs['input_ids'].shape[1] if 'input_ids' in inputs else MAX_TEXT_TOKENS
    metadata["processed_tokens"] = processed_token_count

    # Simple truncation check (word-based approximation)
    original_length = len(text.split())
    if original_length > MAX_TEXT_TOKENS and warn_truncation:
        metadata["truncated"] = True
        logger.warning(f"Text input may have been truncated. Original ~{original_length} words.")
    
    logger.debug(f"Sending to SigLIP 2 for text embedding, input keys: {list(inputs.keys())}")
    
    # Get text features using SigLIP 2 - only pass the expected arguments
    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=inputs.get('input_ids'),
            attention_mask=inputs.get('attention_mask')
        )
        embedding = text_features[0].cpu().numpy()

    # Normalize embedding
    embedding = normalize_embeddings(embedding).squeeze()
    
    logger.debug(f"Text embedding shape: {embedding.shape}")
    
    return embedding, metadata


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings - OpenAI compatible endpoint"""
    try:
        inputs = [request.input] if isinstance(request.input, str) else request.input
        embeddings = []
        total_tokens = 0
        
        for idx, item in enumerate(inputs):
            if isinstance(item, str):
                embedding, metadata = await get_embedding(item)
                # Token counting for simple string (text or image)
                if metadata["input_type"] == "text":
                    # Use MAX_TEXT_TOKENS as default if "processed_tokens" is somehow missing after our fixes
                    total_tokens += metadata.get("processed_tokens", MAX_TEXT_TOKENS)
                else: # image
                    total_tokens += 1 # Image counted as 1 token, as decided
            elif isinstance(item, JointInputItem): # Pydantic automatically converts dict to JointInputItem
                embedding, metadata = await get_joint_embedding(item.image, item.text)
                # For joint input, sum image tokens (1) and processed text tokens
                total_tokens += metadata.get("total_processed_tokens", MAX_TEXT_TOKENS + 1) # Default if somehow missing
            else:
                # This case should ideally not be reached due to Pydantic validation
                # on EmbeddingRequest.input which now includes JointInputItem
                logger.error(f"Unexpected item type in input list: {type(item)} - {item}")
                raise HTTPException(status_code=400, detail=f"Invalid input item type: {type(item)}")

            embeddings.append({
                "object": "embedding",
                "index": idx,
                "embedding": embedding.tolist()
            })
        
        return EmbeddingResponse(
            object="list",
            data=embeddings,
            model=request.model,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"Error in create_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings/image", response_model=EmbeddingResponse)
async def create_embeddings_with_image(
    file: Optional[UploadFile] = File(None, description="Image file to get embedding for"),
    input: Optional[str] = Form(None, description="Text input or image URL/base64"),
    model: str = Form(MODEL_NAME.split("/")[-1], description="Model to use"),
    encoding_format: str = Form("float", description="Format of the embeddings")
):
    """
    Generate embeddings with file upload support.
    Accepts either:
    - An uploaded image file
    - A text/URL/base64 input via form field
    - Both (will process the uploaded file)
    """
    try:
        embeddings = []
        total_tokens = 0
        
        if file:
            # Process uploaded file
            logger.info(f"Processing uploaded file: {file.filename}")
            
            # Validate file type
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file must be an image")
            
            # Load and process image
            image = await load_image_from_upload(file)
            embedding, metadata = await get_image_embedding(image)
            
            embeddings.append({
                "object": "embedding",
                "index": 0,
                "embedding": embedding.tolist()
            })
            total_tokens = 1
            
        elif input:
            # Process text/URL/base64 input
            embedding, metadata = await get_embedding(input)
            
            if metadata["input_type"] == "text":
                total_tokens = metadata.get("processed_tokens", 1)
            else:
                total_tokens = 1
            
            embeddings.append({
                "object": "embedding",
                "index": 0,
                "embedding": embedding.tolist()
            })
            
        else:
            raise HTTPException(status_code=400, detail="Either 'file' or 'input' must be provided")
        
        return EmbeddingResponse(
            object="list",
            data=embeddings,
            model=model,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create_embeddings_with_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rank", response_model=RankResponse)
async def rank_candidates(request: RankRequest):
    """Rank candidates by similarity to query"""
    try:
        # Get query embeddings
        queries = [request.query] if isinstance(request.query, str) else request.query
        query_embeddings = []
        
        for q in queries:
            embedding, _ = await get_embedding(q)
            query_embeddings.append(embedding)
        
        query_embeddings = np.array(query_embeddings)
        
        # Get candidate embeddings
        candidate_embeddings = []
        for c in request.candidates:
            embedding, _ = await get_embedding(c)
            candidate_embeddings.append(embedding)
        
        candidate_embeddings = np.array(candidate_embeddings)
        
        # Compute cosine similarities
        similarities = np.dot(query_embeddings, candidate_embeddings.T)
        
        # Apply sigmoid for SigLIP 2 (no temperature scaling needed)
        probabilities = 1 / (1 + np.exp(-similarities))
        
        # Format results
        results = []
        for i, query in enumerate(queries):
            query_results = []
            for j, candidate in enumerate(request.candidates):
                result = {
                    "candidate": candidate,
                    "score": float(probabilities[i, j]),
                    "similarity": float(similarities[i, j])
                }
                query_results.append(result)
            
            # Sort by score
            query_results.sort(key=lambda x: x['score'], reverse=True)
            
            results.append({
                "query": query,
                "rankings": query_results
            })
        
        return RankResponse(
            results=results,
            model=request.model,
            usage={
                "prompt_tokens": len(queries) + len(request.candidates),
                "total_tokens": len(queries) + len(request.candidates)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in rank_candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/classify")
async def classify_image(request: ClassifyRequest):
    """Zero-shot image classification"""
    rank_request = RankRequest(
        model=request.model,
        query=request.image,
        candidates=request.labels
    )
    
    rank_response = await rank_candidates(rank_request)
    
    # Simplify response for classification
    classifications = []
    for result in rank_response.results[0]["rankings"]:
        classifications.append({
            "label": result["candidate"],
            "score": result["score"]
        })
    
    return {
        "image": request.image,
        "classifications": classifications,
        "model": request.model
    }


@app.post("/v1/classify/image")
async def classify_uploaded_image(
    file: UploadFile = File(..., description="Image file to classify"),
    labels: str = Form(..., description="Comma-separated list of labels"),
    model: str = Form(MODEL_NAME.split("/")[-1], description="Model to use")
):
    """Zero-shot image classification with file upload"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
        # Parse labels
        label_list = [label.strip() for label in labels.split(",") if label.strip()]
        if not label_list:
            raise HTTPException(status_code=400, detail="At least one label must be provided")
        
        # Load image
        image = await load_image_from_upload(file)
        
        # Get image embedding
        image_embedding, _ = await get_image_embedding(image)
        
        # Get label embeddings
        label_embeddings = []
        for label in label_list:
            embedding, _ = await get_text_embedding(label)
            label_embeddings.append(embedding)
        
        label_embeddings = np.array(label_embeddings)
        
        # Compute similarities
        similarities = np.dot(image_embedding.reshape(1, -1), label_embeddings.T).squeeze()
        
        # Apply sigmoid for SigLIP 2
        probabilities = 1 / (1 + np.exp(-similarities))
        
        # Format results
        classifications = []
        for i, label in enumerate(label_list):
            classifications.append({
                "label": label,
                "score": float(probabilities[i])
            })
        
        # Sort by score
        classifications.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            "filename": file.filename,
            "classifications": classifications,
            "model": model
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in classify_uploaded_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    await ensure_models_loaded()
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME.split("/")[-1],  # Use just the model name part
                "object": "model",
                "owned_by": "google",
                "root": MODEL_NAME.split("/")[-1],
                "permission": []
            }
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with batch processing info"""
    queue_info = {}
    if request_queue is not None:
        queue_info = {
            "queue_size": request_queue.qsize(),
            "max_queue_size": MAX_QUEUE_SIZE,
            "active_workers": len([w for w in batch_workers if not w.done()]),
            "total_workers": len(batch_workers)
        }
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "version": "2.0.0",
        "hardware": {
            "cpu_cores": hardware_config["cpu_count"],
            "has_gpu": hardware_config["has_gpu"],
            "gpu_name": hardware_config["gpu_name"],
            "gpu_memory_gb": hardware_config["gpu_memory"],
            "device_type": hardware_config["device_type"],
            "optimized_for": "image_embeddings"
        },
        "batch_processing": {
            "enabled": True,
            "batch_size": BATCH_SIZE,
            "batch_wait_time_ms": BATCH_WAIT_TIME_MS,
            "max_concurrent_batches": MAX_CONCURRENT_BATCHES,
            "auto_scaling": AUTO_SCALE_WORKERS,
            **queue_info
        },
        "performance_estimates": {
            "images_per_second": f"{MAX_CONCURRENT_BATCHES * (15 if hardware_config['has_gpu'] else 0.5):.1f}" if hardware_config["has_gpu"] else f"{MAX_CONCURRENT_BATCHES * 0.3:.1f}-{MAX_CONCURRENT_BATCHES * 1.0:.1f}",
            "note": "GPU acceleration provides 10-50x speedup for image embeddings" if not hardware_config["has_gpu"] else "GPU acceleration active"
        }
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SigLIP 2 Embeddings API",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "device": str(device) if device else "unknown",
        "endpoints": {
            "/v1/models": "List available models (OpenAI compatible)",
            "/v1/embeddings": "Generate embeddings (OpenAI compatible)",
            "/v1/embeddings/image": "Generate embeddings with file upload",
            "/v1/rank": "Rank candidates by similarity",
            "/v1/classify": "Zero-shot image classification",
            "/v1/classify/image": "Zero-shot classification with file upload",
            "/health": "Health check",
            "/docs": "Interactive API documentation"
        }
    }
