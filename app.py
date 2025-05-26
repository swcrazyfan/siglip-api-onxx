"""
SigLIP ONNX API - Fast multimodal embeddings with OpenAI-compatible endpoints
Supports text and image inputs with automatic detection
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import onnxruntime as ort
from transformers import AutoTokenizer, AutoProcessor
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for models
vision_session = None
text_session = None
processor = None
tokenizer = None
logit_scale = 100.0  # SigLIP's temperature parameter

# Model configuration
MODEL_NAME = "pulsejet/siglip-base-patch16-256-multilingual-onnx"
MAX_TEXT_TOKENS = 64
IMAGE_SIZE = 256


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    model: str = Field(default="siglip-base-patch16-256-multilingual")
    input: Union[str, List[str]] = Field(..., description="Text, image URL, base64 image, or file path")
    encoding_format: Optional[str] = Field(default="float", description="Format of the embeddings")


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]


class RankRequest(BaseModel):
    """Request model for ranking/similarity"""
    model: str = Field(default="siglip-base-patch16-256-multilingual")
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
    model: str = Field(default="siglip-base-patch16-256-multilingual")


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
    title="SigLIP ONNX Embeddings API",
    description="Fast multimodal embeddings with OpenAI-compatible endpoints",
    version="1.0.0",
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
    """Load ONNX models and processors"""
    global vision_session, text_session, processor, tokenizer
    
    logger.info("Loading SigLIP ONNX models...")
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Download ONNX model
        vision_model_path = hf_hub_download(
            repo_id=MODEL_NAME,
            filename="onnx/model_quantized.onnx"
        )
        
        # Create ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Detect available providers
        available_providers = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
            logger.info("Using CUDA for acceleration")
        providers.append('CPUExecutionProvider')
        
        vision_session = ort.InferenceSession(vision_model_path, session_options, providers=providers)
        
        # For text, we'll use the same model (SigLIP uses shared architecture)
        text_session = vision_session
        
        # Load processor and tokenizer
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-multilingual")
        tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-256-multilingual")
        
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


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
        async with asyncio.to_thread(requests.get, input_str, timeout=10) as response:
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
    
    elif input_type == 'base64':
        image_bytes = decode_base64_image(input_str)
        return Image.open(BytesIO(image_bytes)).convert('RGB')
    
    elif input_type == 'file':
        return await asyncio.to_thread(Image.open, input_str)
    
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


async def get_image_embedding(image_input: Union[str, Image.Image]) -> tuple[np.ndarray, dict]:
    """Get embedding for an image with metadata"""
    if isinstance(image_input, str):
        image = await load_image_from_input(image_input)
    else:
        image = image_input
    
    # Process image
    image_inputs = processor(images=image, return_tensors="np")
    
    # Run inference
    outputs = await asyncio.to_thread(
        vision_session.run,
        None,
        {vision_session.get_inputs()[0].name: image_inputs['pixel_values']}
    )
    
    # Extract and normalize embedding
    embedding = outputs[0].squeeze()  # Adjust index based on model output
    embedding = normalize_embeddings(embedding).squeeze()
    
    return embedding, {"input_type": "image", "truncated": False}


async def get_text_embedding(text: str, warn_truncation: bool = True) -> tuple[np.ndarray, dict]:
    """Get embedding for text with truncation warning"""
    metadata = {"input_type": "text", "truncated": False}
    
    # Check token length
    test_tokens = tokenizer(text, return_tensors="np", truncation=False)
    actual_length = len(test_tokens['input_ids'][0])
    
    if actual_length > MAX_TEXT_TOKENS:
        metadata["truncated"] = True
        metadata["original_tokens"] = actual_length
        metadata["processed_tokens"] = MAX_TEXT_TOKENS
        
        if warn_truncation:
            truncated_ratio = ((actual_length - MAX_TEXT_TOKENS) / actual_length) * 100
            logger.warning(
                f"Text has {actual_length} tokens but SigLIP only processes {MAX_TEXT_TOKENS}. "
                f"Truncating {truncated_ratio:.1f}% of input!"
            )
    
    # Process with truncation
    text_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_TOKENS
    )
    
    # Run inference
    input_feed = {}
    for i, input_name in enumerate(text_session.get_inputs()):
        if input_name.name == "input_ids":
            input_feed[input_name.name] = text_inputs['input_ids']
        elif input_name.name == "attention_mask":
            input_feed[input_name.name] = text_inputs['attention_mask']
    
    outputs = await asyncio.to_thread(text_session.run, None, input_feed)
    
    # Extract and normalize embedding
    embedding = outputs[0].squeeze()  # Adjust based on model output
    embedding = normalize_embeddings(embedding).squeeze()
    
    return embedding, metadata


async def get_embedding(input_str: str) -> tuple[np.ndarray, dict]:
    """Get embedding for any input type"""
    input_type = detect_input_type(input_str)
    
    if input_type in ['url', 'base64', 'file']:
        try:
            return await get_image_embedding(input_str)
        except Exception as e:
            logger.warning(f"Failed to process as image: {e}, falling back to text")
            return await get_text_embedding(input_str)
    else:
        return await get_text_embedding(input_str)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings - OpenAI compatible endpoint"""
    try:
        inputs = [request.input] if isinstance(request.input, str) else request.input
        embeddings = []
        total_tokens = 0
        
        for idx, inp in enumerate(inputs):
            embedding, metadata = await get_embedding(inp)
            
            # Track tokens for text inputs
            if metadata["input_type"] == "text":
                total_tokens += metadata.get("processed_tokens", 1)
            else:
                total_tokens += 1  # Count images as 1 token
            
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
    model: str = Form("siglip-base-patch16-256-multilingual", description="Model to use"),
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
        
        # Apply temperature scaling
        scaled_similarities = logit_scale * similarities
        
        # Compute probabilities using sigmoid
        probabilities = 1 / (1 + np.exp(-scaled_similarities))
        
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
    model: str = Form("siglip-base-patch16-256-multilingual", description="Model to use")
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
        
        # Apply temperature scaling and sigmoid
        scaled_similarities = logit_scale * similarities
        probabilities = 1 / (1 + np.exp(-scaled_similarities))
        
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
    """List available models - OpenAI compatible endpoint"""
    return {
        "object": "list",
        "data": [
            {
                "id": "siglip-base-patch16-256-multilingual",
                "object": "model",
                "created": 1677610602,  # Arbitrary timestamp
                "owned_by": "google",
                "permission": [],
                "root": "siglip-base-patch16-256-multilingual",
                "parent": None,
            }
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": vision_session is not None,
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SigLIP ONNX Embeddings API",
        "version": "1.0.0",
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
