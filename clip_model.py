"""CLIP Model Loader and Initialization Module.

Academic Project: Zero-Shot Image Classification for Rare Species
Team Member 1 Responsibility: Model Initialization, Device Management, and Model Metadata
Backbone: openai/clip-vit-base-patch32
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import torch
from transformers import CLIPModel, CLIPProcessor
from config import DEFAULT_MODEL_NAME

logger = logging.getLogger(__name__)

# Known local snapshot cache path if cached
HF_SNAPSHOT_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--openai--clip-vit-base-patch32"
    / "snapshots"
    / "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
)

# Global cache for standalone usage outside of Streamlit if needed
_CACHED_MODEL: Optional[CLIPModel] = None
_CACHED_PROCESSOR: Optional[CLIPProcessor] = None
_CURRENT_DEVICE: Optional[torch.device] = None


def get_optimal_device() -> torch.device:
    """Detect and return the best available compute device (CUDA GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Running on CPU.")
    return device


def load_clip_model_and_processor(
    model_name: str = DEFAULT_MODEL_NAME,
    force_reload: bool = False
) -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    """Load and cache the pretrained CLIP model and processor.
    
    Places the model in evaluation mode (model.eval()) and disables gradient tracking.
    
    Args:
        model_name: Hugging Face model identifier (default: openai/clip-vit-base-patch32).
        force_reload: Whether to bypass memory cache and reload.
        
    Returns:
        Tuple of (model, processor, device).
        
    Raises:
        RuntimeError: If model weights cannot be loaded due to network or corrupt cache.
    """
    global _CACHED_MODEL, _CACHED_PROCESSOR, _CURRENT_DEVICE

    if not force_reload and _CACHED_MODEL is not None and _CACHED_PROCESSOR is not None:
        return _CACHED_MODEL, _CACHED_PROCESSOR, _CURRENT_DEVICE

    device = get_optimal_device()

    # Determine optimal load target: prefer verified local offline snapshot if present
    load_source = model_name
    use_local_only = False
    if HF_SNAPSHOT_PATH.exists() and (HF_SNAPSHOT_PATH / "model.safetensors").exists():
        load_source = str(HF_SNAPSHOT_PATH)
        use_local_only = True
        logger.info(f"Using verified offline model snapshot from: {load_source}")

    try:
        logger.info(f"Loading CLIP processor from '{load_source}'...")
        processor = CLIPProcessor.from_pretrained(
            load_source,
            local_files_only=use_local_only
        )

        logger.info(f"Loading CLIP model from '{load_source}'...")
        model = CLIPModel.from_pretrained(
            load_source,
            local_files_only=use_local_only
        )

        # Move to target device and switch to evaluation mode
        model.to(device)
        model.eval()

        # Turn off gradients for all parameters to ensure pure inference
        for param in model.parameters():
            param.requires_grad = False

        _CACHED_MODEL = model
        _CACHED_PROCESSOR = processor
        _CURRENT_DEVICE = device

        logger.info(f"Successfully loaded CLIP model on {device}")
        return model, processor, device

    except Exception as exc:
        # If offline snapshot failed for any reason, try online model_name as fallback
        if use_local_only:
            try:
                logger.info(f"Local snapshot fallback: attempting default hub load for {model_name}...")
                processor = CLIPProcessor.from_pretrained(model_name)
                model = CLIPModel.from_pretrained(model_name)
                model.to(device)
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                _CACHED_MODEL = model
                _CACHED_PROCESSOR = processor
                _CURRENT_DEVICE = device
                return model, processor, device
            except Exception as fallback_exc:
                pass

        err_msg = (
            f"Failed to load CLIP model '{model_name}'. "
            f"Please verify internet connectivity for initial download or Hugging Face cache. "
            f"Error details: {str(exc)}"
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg) from exc


def get_model_metadata(model: CLIPModel, device: torch.device) -> Dict[str, Any]:
    """Extract academic and architecture metadata from the loaded CLIP model.
    
    Returns:
        Dictionary with parameter counts, embedding dimension, vision and text configurations.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    vision_cfg = getattr(model.config, "vision_config", None)
    text_cfg = getattr(model.config, "text_config", None)

    return {
        "model_name": DEFAULT_MODEL_NAME,
        "device": str(device).upper(),
        "total_parameters": f"{total_params / 1e6:.1f} Million ({total_params:,} params)",
        "trainable_parameters": trainable_params,
        "projection_dim": getattr(model.config, "projection_dim", 512),
        "vision_backbone": "Vision Transformer (ViT-B/32)",
        "vision_image_size": f"{vision_cfg.image_size}x{vision_cfg.image_size}" if vision_cfg else "224x224",
        "vision_patch_size": f"{vision_cfg.patch_size}x{vision_cfg.patch_size}" if vision_cfg else "32x32",
        "vision_hidden_size": vision_cfg.hidden_size if vision_cfg else 768,
        "vision_layers": vision_cfg.num_hidden_layers if vision_cfg else 12,
        "vision_heads": vision_cfg.num_attention_heads if vision_cfg else 12,
        "text_vocab_size": text_cfg.vocab_size if text_cfg else 49408,
        "text_max_position": text_cfg.max_position_embeddings if text_cfg else 77,
        "text_hidden_size": text_cfg.hidden_size if text_cfg else 512,
        "text_layers": text_cfg.num_hidden_layers if text_cfg else 12,
    }
