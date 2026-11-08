"""Image validation, loading, and preprocessing module for RareVision AI.

Academic Project: Zero-Shot Image Classification for Rare Species
Team Member 1 & 4 Responsibility: Preprocessing & Input Validation
"""

import io
from typing import Tuple, Optional
from PIL import Image, ImageOps, UnidentifiedImageError
from config import ALLOWED_IMAGE_EXTENSIONS, MAX_IMAGE_FILE_SIZE_BYTES, MAX_IMAGE_FILE_SIZE_MB


class ImageProcessingError(Exception):
    """Custom exception raised when image processing or validation fails."""
    pass


def validate_image_file(uploaded_file) -> Tuple[bool, Optional[str]]:
    """Validate uploaded file type and file size.
    
    Args:
        uploaded_file: Streamlit UploadedFile or file-like object with .name and .size.
        
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    if uploaded_file is None:
        return False, "No file uploaded. Please select or drag-and-drop an image."

    file_name = getattr(uploaded_file, "name", "unknown")
    file_size = getattr(uploaded_file, "size", None)

    # Check extension
    extension = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return False, (
            f"Unsupported file format '{extension}'. "
            f"Allowed formats: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}."
        )

    # Check file size
    if file_size is not None and file_size > MAX_IMAGE_FILE_SIZE_BYTES:
        return False, (
            f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed limit "
            f"of {MAX_IMAGE_FILE_SIZE_MB} MB."
        )

    return True, None


def load_image_safely(file_or_path) -> Image.Image:
    """Safely open, transpose EXIF orientation, and convert an image to RGB.
    
    Args:
        file_or_path: File path, file-like buffer, or BytesIO.
        
    Returns:
        PIL.Image.Image in RGB mode.
        
    Raises:
        ImageProcessingError: If image is corrupted or cannot be decoded.
    """
    try:
        if hasattr(file_or_path, "read"):
            # Ensure seek is at start
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            raw_bytes = file_or_path.read()
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            img = Image.open(io.BytesIO(raw_bytes))
        else:
            img = Image.open(file_or_path)

        # Correct orientation from camera EXIF tags
        img = ImageOps.exif_transpose(img)

        # Convert palette/grayscale/RGBA to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Verify integrity
        img.load()
        return img

    except UnidentifiedImageError:
        raise ImageProcessingError(
            "The provided file is not a valid image or the file header is corrupted."
        )
    except Exception as exc:
        raise ImageProcessingError(f"Failed to process image: {str(exc)}")


def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (150, 150)) -> Image.Image:
    """Generate a thumbnail for session history display while preserving aspect ratio.
    
    Args:
        image: Source PIL Image.
        size: Target thumbnail box size (width, height).
        
    Returns:
        Resized PIL Image thumbnail.
    """
    thumb = image.copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    return thumb

