def generate_thumbnail_embedding(
    image_path: str,
    description: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None, # api_key is not used by this server
    logger=None
) -> Optional[List[float]]:
    """
    Generate embedding for image using the SigLIP model via local API.
    
    Args:
        image_path: Path to the thumbnail image
        description: Short text description of the thumbnail (not used in current API implementation but kept for future compatibility)
        api_base: API base URL (default: http://localhost:8001)
        api_key: API key (not required for local server)
        logger: Optional logger
        
    Returns:
        Optional[List[float]]: 768-dimensional embedding vector or None on failure
    """
    if logger is None:
        