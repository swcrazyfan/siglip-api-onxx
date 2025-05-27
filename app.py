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
_model_load_lock = asyncio.Lock()  # Lock for synchronizing model loading
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
    
    async with _model_load_lock:  # Ensure only one coroutine loads models at a time
        # Check if models are already loaded after acquiring the lock
        if all(component is not None for component in [vision_session, text_session, processor, tokenizer]):
            logger.debug("Models already loaded by another coroutine")
            return
            
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


async def ensure_models_loaded():
    """Ensure all required model components are loaded"""
    global vision_session, text_session, processor, tokenizer
    
    if any(component is None for component in [vision_session, text_session, processor, tokenizer]):
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
    
    # Process image
    image_inputs = processor(images=image, return_tensors="np")
    
    # Prepare input feed
    input_feed = {}
    batch_size = image_inputs['pixel_values'].shape[0]

    for model_input_node in vision_session.get_inputs():
        node_name = model_input_node.name
        if node_name == "pixel_values":
            input_feed[node_name] = image_inputs['pixel_values']
        elif node_name == "input_ids":
            # Provide dummy input_ids for image-only inference
            # e.g., a single padding token or a generic token
            dummy_input_ids = np.array([[tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0]] * batch_size, dtype=np.int64)
            input_feed[node_name] = dummy_input_ids
        # elif node_name == "attention_mask": # Optional: provide dummy attention_mask if model requires it
            # dummy_attention_mask = np.ones_like(dummy_input_ids, dtype=np.int64)
            # input_feed[node_name] = dummy_attention_mask
        # else:
            # logger.warning(f"Model expects an input '{node_name}' not handled in image_embedding.")

    # Run inference
    outputs = await asyncio.to_thread(vision_session.run, None, input_feed)

    # Log raw output shapes and names for debugging
    if vision_session and outputs:
        output_node_names = [node.name for node in vision_session.get_outputs()]
        logger.info(f"ONNX session output node names (image path): {output_node_names}")
        logger.info(f"Raw outputs shapes from ONNX session (image path): {[o.shape for o in outputs]}")

        image_embedding_array = None
        possible_image_embed_names = ["image_embeds", "image_embeddings", "image_features", "vision_features"]
        
        for i, name in enumerate(output_node_names):
            if name in possible_image_embed_names:
                image_embedding_array = outputs[i]
                logger.info(f"Found image embedding output '{name}' at index {i} with shape {image_embedding_array.shape}")
                break
        
        if image_embedding_array is None:
            if outputs and len(outputs) > 3: # Fallback to outputs[3] if no named output found and enough outputs exist
                logger.warning(f"Could not find a known image embedding output name in {output_node_names}. Falling back to outputs[3].")
                image_embedding_array = outputs[3]
            elif outputs: # Fallback to outputs[0] if not enough outputs for index 3
                 logger.warning(f"Could not find a known image embedding output name or index 3. Falling back to outputs[0]. This might be incorrect.")
                 image_embedding_array = outputs[0]
            else:
                logger.error("ONNX session returned no outputs (image path).")
                raise ValueError("ONNX session returned no outputs for image embedding.")
    elif not outputs:
        logger.error("ONNX session returned no outputs (image path) (outputs list is empty or None).")
        raise ValueError("ONNX session returned no outputs for image embedding.")
    else: # Should not happen
        logger.error("vision_session is None, cannot process outputs (image path).")
        raise ValueError("vision_session is not loaded.")

    # Extract and normalize embedding
    if image_embedding_array.ndim == 0:
        logger.error(f"Image embedding array is a scalar: {image_embedding_array}. Cannot process.")
        raise ValueError("Image embedding output from ONNX model is a scalar.")
        
    embedding = image_embedding_array.squeeze()
    logger.info(f"Shape of image embedding after squeeze: {embedding.shape}")

    embedding = normalize_embeddings(embedding).squeeze()
    logger.info(f"Shape of image embedding after normalize_embeddings and final squeeze: {embedding.shape}")
    
    return embedding, {"input_type": "image", "truncated": False}


async def get_text_embedding(text: str, warn_truncation: bool = True) -> tuple[np.ndarray, dict]:
    """Get embedding for text with truncation warning"""
    await ensure_models_loaded()
    
    metadata = {"input_type": "text", "truncated": False}
    
    # Check token length
    test_tokens = tokenizer(text, return_tensors="np", truncation=False)
    actual_length = len(test_tokens['input_ids'][0])
    
    # Process with truncation and padding
    text_inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length", # Pad to max_length to ensure consistent input shape
        truncation=True,
        max_length=MAX_TEXT_TOKENS
    )
    
    # The number of tokens actually processed and fed to the model
    processed_token_count = text_inputs['input_ids'].shape[1]
    metadata["processed_tokens"] = processed_token_count

    if actual_length > MAX_TEXT_TOKENS:
        metadata["truncated"] = True
        metadata["original_tokens"] = actual_length
        # metadata["processed_tokens"] is already set by now to MAX_TEXT_TOKENS due to truncation and padding="max_length"
        
        if warn_truncation:
            truncated_ratio = ((actual_length - MAX_TEXT_TOKENS) / actual_length) * 100
            logger.warning(
                f"Text input was {actual_length} tokens long and has been truncated/padded to {processed_token_count} tokens. "
                f"Original text truncated by {truncated_ratio:.1f}%."
            )
    elif actual_length < processed_token_count: # Check if padding occurred (and no truncation)
        logger.info(f"Text input was {actual_length} tokens long and has been padded to {processed_token_count} tokens.")
    # If actual_length == processed_token_count and not > MAX_TEXT_TOKENS, no truncation or significant padding occurred.
    
    # Run inference
    input_feed = {}
    # Determine batch size from the actual text inputs
    batch_size = text_inputs['input_ids'].shape[0]

    # Ensure all model inputs defined in the ONNX graph are provided
    for model_input_node in text_session.get_inputs():
        node_name = model_input_node.name
        if node_name == "input_ids":
            input_feed[node_name] = text_inputs['input_ids']
        elif node_name == "attention_mask":
            input_feed[node_name] = text_inputs['attention_mask']
        elif node_name == "pixel_values":
            # Provide dummy pixel_values for text-only inference
            # Expected shape: (batch_size, num_channels, height, width)
            # Expected dtype: float32 (common for image models)
            dummy_pixel_values = np.zeros(
                (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE), # IMAGE_SIZE is a global constant
                dtype=np.float32
            )
            input_feed[node_name] = dummy_pixel_values
        # else:
            # logger.warning(f"Model expects an input '{node_name}' not handled in text_embedding.")
    
    outputs = await asyncio.to_thread(text_session.run, None, input_feed)

    # Log raw output shapes and names for debugging
    if text_session and outputs:
        output_node_names = [node.name for node in text_session.get_outputs()]
        logger.info(f"ONNX session output node names: {output_node_names}")
        logger.info(f"Raw outputs shapes from ONNX session: {[o.shape for o in outputs]}")

        # Attempt to find the text embedding output by name
        text_embedding_array = None
        # Common names for text embeddings output
        possible_text_embed_names = ["text_embeds", "text_embeddings", "text_features"]
        
        for i, name in enumerate(output_node_names):
            if name in possible_text_embed_names:
                text_embedding_array = outputs[i]
                logger.info(f"Found text embedding output '{name}' at index {i} with shape {text_embedding_array.shape}")
                break
        
        if text_embedding_array is None:
            if outputs: # Fallback to outputs[0] if no named output found
                logger.warning(f"Could not find a known text embedding output name in {output_node_names}. Falling back to outputs[0].")
                text_embedding_array = outputs[0]
            else:
                logger.error("ONNX session returned no outputs.")
                raise ValueError("ONNX session returned no outputs for text embedding.")
    elif not outputs:
        logger.error("ONNX session returned no outputs (outputs list is empty or None).")
        raise ValueError("ONNX session returned no outputs for text embedding.")
    else: # Should not happen if ensure_models_loaded worked
        logger.error("text_session is None, cannot process outputs.")
        raise ValueError("text_session is not loaded.")

    # Extract and normalize embedding
    if text_embedding_array.ndim == 0: # Check if it's a scalar
        logger.error(f"Text embedding array is a scalar: {text_embedding_array}. Cannot process.")
        raise ValueError("Text embedding output from ONNX model is a scalar.")

    embedding = text_embedding_array.squeeze()
    logger.info(f"Shape of embedding after squeeze: {embedding.shape}")
    
    if embedding.ndim == 0: # Check if squeeze resulted in a scalar
         logger.warning(f"Embedding became scalar after squeeze: {embedding}. This might lead to errors in normalization if not handled.")
         # If it's scalar, normalization might not be meaningful or possible in the current way.
         # For now, let normalize_embeddings handle it, but this is a point of concern.

    embedding = normalize_embeddings(embedding).squeeze()
    logger.info(f"Shape of embedding after normalize_embeddings and final squeeze: {embedding.shape}")
    
    return embedding, metadata


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
