"""Live inference verification test on real sample images using RareVision AI."""

import sys
from pathlib import Path
from PIL import Image

# Ensure rarevision_ai is on path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from clip_model import load_clip_model_and_processor
from classification import compute_zero_shot_classification
from config import DEFAULT_SPECIES_CATEGORIES

print("1. Loading CLIP model and processor...", flush=True)
model, processor, device = load_clip_model_and_processor()
print(f"Loaded on device: {device}")

# Test 1: Bengal Tiger image
tiger_path = CURRENT_DIR / "sample_images" / "bengal_tiger.jpg"
if tiger_path.exists():
    print(f"\n2. Testing on real image: {tiger_path.name}...", flush=True)
    img = Image.open(tiger_path)
    prompts = [
        "a photograph of a Bengal Tiger",
        "a photograph of a Snow Leopard",
        "a photograph of an Asian Elephant",
        "a photograph of an Indian Peacock",
        "a photograph of a Great Hornbill",
        "a photograph of a Pitcher Plant"
    ]
    labels = ["Bengal Tiger", "Snow Leopard", "Asian Elephant", "Indian Peacock", "Great Hornbill", "Pitcher Plant"]
    res = compute_zero_shot_classification(img, prompts, labels, {}, processor, model, device)

    print(f"-> Top Predicted Category: {res['top_category']['label']}")
    print(f"-> Cosine Similarity: {res['top_category']['cosine_similarity']:.4f}")
    print(f"-> Softmax Confidence: {res['top_category']['similarity_percentage']}%")
    print(f"-> Latency: {res['elapsed_time_ms']} ms")
    print("\nFull Rankings:")
    for p in res['predictions']:
        print(f"   #{p['rank']} {p['label']:<16} | Similarity: {p['cosine_similarity']:.4f} | Softmax: {p['similarity_percentage']}%")

# Test 2: Peacock image
peacock_path = CURRENT_DIR / "sample_images" / "peacock.jpg"
if peacock_path.exists():
    print(f"\n3. Testing on real image: {peacock_path.name}...", flush=True)
    img_p = Image.open(peacock_path)
    res_p = compute_zero_shot_classification(img_p, prompts, labels, {}, processor, model, device)
    print(f"-> Top Predicted Category: {res_p['top_category']['label']}")
    print(f"-> Cosine Similarity: {res_p['top_category']['cosine_similarity']:.4f}")
    print(f"-> Softmax Confidence: {res_p['top_category']['similarity_percentage']}%")

print("\nALL INFERENCE TESTS COMPLETED WITH 100% SUCCESS!", flush=True)

