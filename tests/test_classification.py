"""Unit and Integration Tests for RareVision AI.

Academic Project: Zero-Shot Image Classification for Rare Species
Team Member 4 Responsibility: Testing, Validation, and Robustness Verification
"""

import io
import os
import sys
from pathlib import Path
import pytest
from PIL import Image
import numpy as np

# Ensure parent directory is in python path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_SPECIES_CATEGORIES,
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_FILE_SIZE_BYTES,
    DEFAULT_MODEL_NAME
)
from image_processor import (
    validate_image_file,
    load_image_safely,
    create_thumbnail,
    ImageProcessingError
)


class MockUploadedFile:
    """Mock Streamlit UploadedFile object for unit testing."""
    def __init__(self, name: str, size: int, buffer: io.BytesIO = None):
        self.name = name
        self.size = size
        self.buffer = buffer or io.BytesIO(b"fake image bytes")

    def read(self):
        return self.buffer.getvalue()

    def seek(self, pos):
        self.buffer.seek(pos)


# =========================================================================
# 1. Config & Data Integrity Tests
# =========================================================================

def test_config_categories_structure():
    """Verify predefined sample categories have the required academic schema."""
    assert "Animals" in DEFAULT_SPECIES_CATEGORIES
    assert "Birds" in DEFAULT_SPECIES_CATEGORIES
    assert "Plants" in DEFAULT_SPECIES_CATEGORIES

    for kingdom, species_list in DEFAULT_SPECIES_CATEGORIES.items():
        assert len(species_list) > 0, f"{kingdom} should contain species entries"
        for item in species_list:
            assert "name" in item, "Each entry must have a name"
            assert "scientific_name" in item, "Each entry must have a scientific name"
            assert "conservation_status" in item, "Each entry must have an IUCN/conservation note"
            assert "prompt_template" in item, "Each entry must have a prompt template"
            assert len(item["name"]) > 0


# =========================================================================
# 2. Image Preprocessing & Validation Tests
# =========================================================================

def test_validate_image_valid_formats():
    """Test validator accepts legitimate image extensions."""
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        mock_file = MockUploadedFile(f"test_wildlife{ext}", 1024 * 500)
        is_valid, error = validate_image_file(mock_file)
        assert is_valid is True
        assert error is None


def test_validate_image_invalid_formats():
    """Test validator rejects unsupported files (pdf, exe, txt)."""
    for ext in [".pdf", ".exe", ".txt", ".mp4", ".gif"]:
        mock_file = MockUploadedFile(f"malicious{ext}", 1024 * 100)
        is_valid, error = validate_image_file(mock_file)
        assert is_valid is False
        assert "Unsupported file format" in error


def test_validate_image_oversized():
    """Test validator rejects images exceeding max allowed file size."""
    oversized_bytes = MAX_IMAGE_FILE_SIZE_BYTES + 1024 * 1024
    mock_file = MockUploadedFile("large_raw_photo.png", oversized_bytes)
    is_valid, error = validate_image_file(mock_file)
    assert is_valid is False
    assert "exceeds maximum allowed limit" in error


def test_load_image_safely_synthetic():
    """Create a synthetic PIL image in memory and test load_image_safely converts to RGB."""
    # Create an RGBA image
    img = Image.new("RGBA", (100, 100), color=(255, 120, 0, 200))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    loaded = load_image_safely(buffer)
    assert loaded.mode == "RGB"
    assert loaded.size == (100, 100)


def test_load_image_corrupted_bytes():
    """Test error handling when reading corrupt image bytes."""
    corrupted_buffer = io.BytesIO(b"Not an image stream definitely corrupt")
    with pytest.raises(ImageProcessingError):
        load_image_safely(corrupted_buffer)


def test_create_thumbnail():
    """Test thumbnail creation preserves aspect ratio and conforms to max dimensions."""
    img = Image.new("RGB", (800, 400), color=(50, 150, 250))
    thumb = create_thumbnail(img, size=(120, 120))
    assert thumb.width <= 120
    assert thumb.height <= 120
    # Original aspect ratio 2:1 -> thumb size should be (120, 60)
    assert thumb.size == (120, 60)


# =========================================================================
# 3. Model & Zero-Shot Classification Tests
# =========================================================================

def test_classification_logic_with_clip():
    """Test actual CLIP embedding and classification pipeline."""
    try:
        from clip_model import load_clip_model_and_processor, get_model_metadata
        from classification import (
            encode_image,
            encode_text_prompts,
            compute_zero_shot_classification
        )
    except Exception as exc:
        pytest.skip(f"CLIP dependencies or weights not yet ready: {exc}")

    try:
        model, processor, device = load_clip_model_and_processor(DEFAULT_MODEL_NAME)
    except Exception as exc:
        pytest.skip(f"Could not load CLIP model in test environment: {exc}")

    # Check model metadata
    metadata = get_model_metadata(model, device)
    assert metadata["projection_dim"] == 512
    assert "ViT-B/32" in metadata["vision_backbone"]

    # Create synthetic test image (e.g. orange stripes resembling tiger or green plant)
    synthetic_img = Image.new("RGB", (224, 224), color=(220, 100, 30))

    # Test image encoder
    img_embed = encode_image(synthetic_img, processor, model, device)
    assert img_embed.shape == (1, 512)
    # Norm should be approximately 1.0 (unit vector)
    norm_val = float(img_embed.norm(p=2).cpu().item())
    assert abs(norm_val - 1.0) < 1e-4

    # Test text encoder
    test_prompts = [
        "a photograph of a Bengal Tiger",
        "a photograph of a Pitcher Plant",
        "a photograph of an Asian Elephant"
    ]
    test_labels = ["Bengal Tiger", "Pitcher Plant", "Asian Elephant"]
    text_embeds = encode_text_prompts(test_prompts, processor, model, device)
    assert text_embeds.shape == (3, 512)

    # Test complete zero-shot pipeline
    meta_lookup = {
        "Bengal Tiger": {"scientific_name": "Panthera tigris", "conservation_status": "Endangered"},
        "Pitcher Plant": {"scientific_name": "Nepenthes khasiana", "conservation_status": "Endangered"},
        "Asian Elephant": {"scientific_name": "Elephas maximus", "conservation_status": "Endangered"}
    }
    results = compute_zero_shot_classification(
        image=synthetic_img,
        candidate_prompts=test_prompts,
        category_labels=test_labels,
        metadata_lookup=meta_lookup,
        processor=processor,
        model=model,
        device=device,
        top_k=3
    )

    assert "predictions" in results
    assert len(results["predictions"]) == 3
    assert results["elapsed_time_ms"] > 0
    assert results["embedding_dim"] == 512

    # Check that probabilities sum to approximately 100%
    total_pct = sum(item["similarity_percentage"] for item in results["predictions"])
    assert 99.0 <= total_pct <= 101.0

