"""Configuration and constants for RareVision AI.

Academic Project: Zero-Shot Image Classification for Rare Species
Team: B.Tech Computer Science and Engineering (3rd Year)
Model: openai/clip-vit-base-patch32
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"
ASSETS_DIR = BASE_DIR / "assets"
TESTS_DIR = BASE_DIR / "tests"

# Hugging Face CLIP Model Identifier
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"

# Image Processing Limits
ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
MAX_IMAGE_FILE_SIZE_MB = 15
MAX_IMAGE_FILE_SIZE_BYTES = MAX_IMAGE_FILE_SIZE_MB * 1024 * 1024

# Predefined Species Categories
# NOTE: Predefined sample species categories are curated for demonstration and educational testing.
# They are not all officially endangered; their conservation status is listed for academic context.
DEFAULT_SPECIES_CATEGORIES = {
    "Animals": [
        {
            "name": "Bengal Tiger",
            "scientific_name": "Panthera tigris tigris",
            "conservation_status": "Endangered (IUCN)",
            "description": "A majestic apex predator with reddish-orange coat and black vertical stripes, native to the Indian subcontinent.",
            "prompt_template": "a high quality photograph of a Bengal Tiger (Panthera tigris)"
        },
        {
            "name": "Snow Leopard",
            "scientific_name": "Panthera uncia",
            "conservation_status": "Vulnerable (IUCN)",
            "description": "Elusive high-altitude feline with thick smoky-gray fur patterned with dark rosettes, inhabiting the Himalayas and Central Asia.",
            "prompt_template": "a high quality photograph of a Snow Leopard in snow (Panthera uncia)"
        },
        {
            "name": "Red Panda",
            "scientific_name": "Ailurus fulgens",
            "conservation_status": "Endangered (IUCN)",
            "description": "Small arboreal mammal with reddish-brown fur, a long shaggy ringed tail, and white markings on its muzzle, native to eastern Himalayas.",
            "prompt_template": "a photograph of a Red Panda on a tree branch (Ailurus fulgens)"
        },
        {
            "name": "Asian Elephant",
            "scientific_name": "Elephas maximus",
            "conservation_status": "Endangered (IUCN)",
            "description": "Largest terrestrial mammal in Asia, characterized by smaller ears than African elephants, arched back, and twin-domed head.",
            "prompt_template": "a photograph of an Asian Elephant in the forest (Elephas maximus)"
        },
        {
            "name": "Indian Pangolin",
            "scientific_name": "Manis crassicaudata",
            "conservation_status": "Endangered (IUCN)",
            "description": "Unique nocturnal scaly anteater covered in overlapping keratin scales, known to roll into a tight protective ball when threatened.",
            "prompt_template": "a close-up photograph of an Indian Pangolin with protective scales (Manis crassicaudata)"
        },
        {
            "name": "Black Panther",
            "scientific_name": "Panthera pardus (melanistic)",
            "conservation_status": "Vulnerable / Melanistic morph",
            "description": "Melanistic color variant of the leopard with deep black coat and ghost rosettes, found in dense tropical rain forests.",
            "prompt_template": "a photograph of a Black Panther (melanistic leopard) in the jungle"
        },
        {
            "name": "Great Indian Bustard",
            "scientific_name": "Ardeotis nigriceps",
            "conservation_status": "Critically Endangered (IUCN)",
            "description": "One of the heaviest flying birds in the world, with horizontal body, long bare legs, and black crown, native to arid Indian grasslands.",
            "prompt_template": "a photograph of a Great Indian Bustard bird in dry grassland (Ardeotis nigriceps)"
        }
    ],
    "Birds": [
        {
            "name": "Peacock",
            "scientific_name": "Pavo cristatus",
            "conservation_status": "Least Concern (National Bird of India)",
            "description": "Large vibrant bird famous for the male's iridescent blue-green plumage and spectacular fan-shaped covert train.",
            "prompt_template": "a vibrant photograph of an Indian Peacock displaying plumage (Pavo cristatus)"
        },
        {
            "name": "Great Hornbill",
            "scientific_name": "Buceros bicornis",
            "conservation_status": "Vulnerable (IUCN)",
            "description": "Grand canopy bird with large black-and-yellow downcurved bill topped by a prominent hollow casque, indigenous to Indian forests.",
            "prompt_template": "a photograph of a Great Hornbill with a yellow casque (Buceros bicornis)"
        },
        {
            "name": "Indian Vulture",
            "scientific_name": "Gyps indicus",
            "conservation_status": "Critically Endangered (IUCN)",
            "description": "Critically endangered scavenger raptor with pale plumage, broad wings, and a bare dark head and neck.",
            "prompt_template": "a photograph of an Indian Vulture perched on rocks (Gyps indicus)"
        },
        {
            "name": "Flamingo",
            "scientific_name": "Phoenicopterus roseus",
            "conservation_status": "Least Concern (IUCN)",
            "description": "Tall wading bird recognized by slender pink-and-white feathers, stilt-like legs, and downward-bent feeding bill.",
            "prompt_template": "a photograph of a Greater Flamingo wading in shallow water (Phoenicopterus roseus)"
        },
        {
            "name": "Siberian Crane",
            "scientific_name": "Leucogeranus leucogeranus",
            "conservation_status": "Critically Endangered (IUCN)",
            "description": "Magnificent snow-white migratory crane with dark red mask and black flight feathers visible in flight.",
            "prompt_template": "a photograph of an endangered Siberian Crane in wetland (Leucogeranus leucogeranus)"
        }
    ],
    "Plants": [
        {
            "name": "Blue Vanda Orchid",
            "scientific_name": "Vanda coerulea",
            "conservation_status": "CITES Appendix II / Threatened",
            "description": "Exotic epiphytic orchid noted for unique bluish-purple patterned flowers, native to the northeastern Himalayas and Western Ghats.",
            "prompt_template": "a close-up photograph of a Blue Vanda Orchid flower (Vanda coerulea)"
        },
        {
            "name": "Neelakurinji",
            "scientific_name": "Strobilanthes kunthiana",
            "conservation_status": "Endemic / Protected",
            "description": "Shroob native to the shola grasslands of the Western Ghats famous for blooming en masse once every 12 years in purplish-blue.",
            "prompt_template": "a photograph of purplish-blue Neelakurinji flowers blooming on hills (Strobilanthes kunthiana)"
        },
        {
            "name": "Ghost Orchid",
            "scientific_name": "Dendrophylax lindenii",
            "conservation_status": "Critically Endangered / Rare",
            "description": "Leafless epiphytic orchid whose pristine white frog-legged flower appears suspended in mid-air on tree bark.",
            "prompt_template": "a photograph of a rare white Ghost Orchid flower on a tree trunk (Dendrophylax lindenii)"
        },
        {
            "name": "Pitcher Plant",
            "scientific_name": "Nepenthes khasiana",
            "conservation_status": "Endangered (CITES Appendix I)",
            "description": "Carnivorous plant native to Meghalaya, India, bearing specialized pitcher-shaped modified leaves designed to trap and digest insects.",
            "prompt_template": "a photograph of an insect-trapping Pitcher Plant (Nepenthes khasiana)"
        },
        {
            "name": "Red Sandalwood",
            "scientific_name": "Pterocarpus santalinus",
            "conservation_status": "Endangered (IUCN)",
            "description": "Rare endemic hardwood tree of the Eastern Ghats of India, highly prized for its deep red heartwood and medicinal properties.",
            "prompt_template": "a photograph of Red Sandalwood tree and its distinctive foliage (Pterocarpus santalinus)"
        }
    ]
}

# Prompt templates for zero-shot species classification
DEFAULT_PROMPT_PREFIXES = [
    "a high quality photograph of a {}",
    "a close-up photo of a {}, a rare wildlife species",
    "a wildlife photo of the {} in nature",
]

# Academic Project Metadata
PROJECT_INFO = {
    "title": "RareVision AI",
    "subtitle": "Zero-Shot Image Classification for Rare Species",
    "project_type": "Third-Year B.Tech Computer Science Academic Project",
    "course": "Computer Vision & Deep Learning Laboratory",
    "model_backbone": "OpenAI CLIP ViT-B/32 (Vision Transformer Base, 32x32 patch size)",
    "embedding_dimension": 512,
    "team_members": [
        {"name": "Student 1", "role": "CLIP Model Loading & Image Embedding Generation"},
        {"name": "Student 2", "role": "Text Prompts, Similarity Scoring & Classification Logic"},
        {"name": "Student 3", "role": "Streamlit Web Application & Interactive UI/UX"},
        {"name": "Student 4", "role": "Testing, Validation, Sample Dataset & Academic Documentation"}
    ]
}

