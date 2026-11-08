"""Zero-Shot Classification and Feature Extraction Module using CLIP.

Academic Project: Zero-Shot Image Classification for Rare Species
Team Member 2 Responsibility: Embedding Generation, Cosine Similarity, Softmax Normalization & Ranking
Model: openai/clip-vit-base-patch32
"""

import time
from typing import List, Dict, Any, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def encode_image(
    image: Image.Image,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device
) -> torch.Tensor:
    """Preprocess an input PIL image and generate normalized 512-dim CLIP image embeddings.
    
    Args:
        image: PIL Image object in RGB mode.
        processor: Pretrained CLIPProcessor.
        model: Pretrained CLIPModel in eval mode.
        device: torch.device (cpu or cuda).
        
    Returns:
        torch.Tensor of shape (1, 512) normalized with L2 norm.
    """
    # Preprocess image (resizes to 224x224, normalizes with ImageNet stats, converts to tensor)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        # Pass through Vision Transformer backbone to get projected image features
        outputs = model.get_image_features(**inputs)
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            image_features = outputs.image_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            image_features = outputs.pooler_output
        elif isinstance(outputs, torch.Tensor):
            image_features = outputs
        elif hasattr(outputs, "last_hidden_state"):
            image_features = outputs.last_hidden_state
        else:
            image_features = outputs[0]

        # Normalize to unit vector on the hypersphere
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    return image_features


def encode_text_prompts(
    text_prompts: List[str],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device
) -> torch.Tensor:
    """Tokenize and generate normalized 512-dim CLIP text embeddings for candidate descriptions.
    
    Args:
        text_prompts: List of descriptive text strings.
        processor: Pretrained CLIPProcessor.
        model: Pretrained CLIPModel in eval mode.
        device: torch.device (cpu or cuda).
        
    Returns:
        torch.Tensor of shape (N, 512) normalized with L2 norm where N is number of prompts.
    """
    if not text_prompts:
        raise ValueError("Candidate text prompts list cannot be empty.")

    # Tokenize text with truncation and padding up to max_length (77 tokens)
    inputs = processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        # Pass through Text Transformer backbone to get projected text features
        outputs = model.get_text_features(**inputs)
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            text_features = outputs.text_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            text_features = outputs.pooler_output
        elif isinstance(outputs, torch.Tensor):
            text_features = outputs
        elif hasattr(outputs, "last_hidden_state"):
            text_features = outputs.last_hidden_state
        else:
            text_features = outputs[0]

        # Normalize to unit vector on the hypersphere
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    return text_features


def compute_zero_shot_classification(
    image: Image.Image,
    candidate_prompts: List[str],
    category_labels: List[str],
    metadata_lookup: Dict[str, Dict[str, Any]],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    top_k: int = 5
) -> Dict[str, Any]:
    """Execute the end-to-end Zero-Shot Image Classification pipeline.
    
    Mathematical Workflow:
      1. Image vector I in R^{1x512}, normalized ||I||_2 = 1.
      2. Text vectors T in R^{Nx512}, normalized ||T_i||_2 = 1.
      3. Cosine similarity matrix S = I * T^T in R^{1xN}.
      4. Logits scaling: logits = 100.0 * S (CLIP default logit temperature).
      5. Softmax distribution: P_i = exp(logits_i) / sum_j exp(logits_j).
      6. Rank descending by similarity and softmax probability.
    
    Args:
        image: PIL Image object.
        candidate_prompts: List of text prompts passed to CLIP text encoder.
        category_labels: User-facing category display names corresponding to prompts.
        metadata_lookup: Dictionary mapping label names to extra metadata (scientific name, status, etc.).
        processor: Pretrained CLIPProcessor.
        model: Pretrained CLIPModel.
        device: torch.device.
        top_k: Number of highest matching classes to return.
        
    Returns:
        Dictionary containing:
          - predictions: List of dicts for top_k matches.
          - all_results: List of dicts for all evaluated candidates.
          - elapsed_time_ms: Total inference time in milliseconds.
          - image_embedding_norm: L2 norm of the raw feature vector before unit normalization.
          - embedding_dim: Feature dimension (512).
    """
    start_time = time.perf_counter()

    # Step 1: Generate image embedding
    image_embeds = encode_image(image, processor, model, device)

    # Step 2: Generate text embeddings
    text_embeds = encode_text_prompts(candidate_prompts, processor, model, device)

    # Step 3: Compute raw cosine similarities: shape (1, N)
    # Since both vectors are unit normalized, cosine similarity is the dot product.
    cosine_sim = torch.matmul(image_embeds, text_embeds.t()).squeeze(0)  # shape (N,)

    # Step 4: Scale similarities using CLIP's learned logit_scale parameter
    # CLIP learns an exp(logit_scale) factor (~100.0) to sharpen the distribution for classification
    logit_scale = model.logit_scale.exp().to(device)
    scaled_logits = cosine_sim * logit_scale

    # Step 5: Softmax probability distribution over the evaluated candidate set
    probabilities = F.softmax(scaled_logits, dim=-1)

    # Convert tensors to CPU numpy / Python primitives
    cosine_sim_np = cosine_sim.cpu().numpy()
    probabilities_np = probabilities.cpu().numpy()

    # Step 6: Rank candidates descending
    sorted_indices = np.argsort(-cosine_sim_np)

    all_results = []
    for rank_idx, idx in enumerate(sorted_indices):
        label = category_labels[idx]
        prompt = candidate_prompts[idx]
        cos_score = float(cosine_sim_np[idx])
        prob_pct = float(probabilities_np[idx] * 100.0)

        meta = metadata_lookup.get(label, {})
        all_results.append({
            "rank": rank_idx + 1,
            "label": label,
            "prompt": prompt,
            "cosine_similarity": cos_score,
            "similarity_percentage": round(prob_pct, 2),
            "scientific_name": meta.get("scientific_name", "N/A"),
            "conservation_status": meta.get("conservation_status", "Unspecified / Custom"),
            "description": meta.get("description", "")
        })

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    top_predictions = all_results[:top_k]

    return {
        "predictions": top_predictions,
        "all_results": all_results,
        "elapsed_time_ms": round(elapsed_ms, 2),
        "embedding_dim": image_embeds.shape[-1],
        "total_categories_evaluated": len(candidate_prompts),
        "top_category": top_predictions[0] if top_predictions else None
    }

