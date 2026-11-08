"""Alternative Gradio Interface for RareVision AI.

Academic Project: Zero-Shot Image Classification for Rare Species
Model: OpenAI CLIP ViT-B/32
"""

import os
import sys
from pathlib import Path
from PIL import Image
import gradio as gr

# Ensure parent directory is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from clip_model import load_clip_model_and_processor, get_model_metadata
from classification import compute_zero_shot_classification
from config import DEFAULT_SPECIES_CATEGORIES

print("Initializing RareVision AI CLIP Engine...", flush=True)
model, processor, device = load_clip_model_and_processor()
meta = get_model_metadata(model, device)
print("CLIP Model Loaded Successfully on", device)

# Build default category list
all_predefined = []
for grp, items in DEFAULT_SPECIES_CATEGORIES.items():
    for item in items:
        all_predefined.append(item["name"])


def classify_species(image, selected_categories, custom_text_prompts, prompt_style):
    if image is None:
        return "Please upload an image.", {}

    # Compile candidate list
    candidate_labels = list(selected_categories) if selected_categories else []
    
    # Add custom prompts if any
    if custom_text_prompts and custom_text_prompts.strip():
        for line in custom_text_prompts.strip().splitlines():
            line = line.strip()
            if line and line not in candidate_labels:
                candidate_labels.append(line)

    if not candidate_labels:
        return "Please select at least one species category or enter custom prompts.", {}

    # Build prompts according to chosen style
    candidate_prompts = []
    meta_lookup = {}
    for label in candidate_labels:
        if "Detailed" in prompt_style:
            prompt = f"a high quality wildlife photograph of a {label} in natural habitat"
        elif "Scientific" in prompt_style:
            prompt = f"a close-up photo of {label}, a rare wildlife species"
        else:
            prompt = f"a photograph of a {label}"
        candidate_prompts.append(prompt)
        meta_lookup[label] = {"name": label, "conservation_status": "Candidate Pool"}

    # Run inference
    res = compute_zero_shot_classification(
        image=image,
        candidate_prompts=candidate_prompts,
        category_labels=candidate_labels,
        metadata_lookup=meta_lookup,
        processor=processor,
        model=model,
        device=device,
        top_k=5
    )

    top = res["top_category"]
    summary_text = (
        f"### 🏆 Top Match: **{top['label']}**\n"
        f"- **Cosine Similarity:** `{top['cosine_similarity']:.4f}`\n"
        f"- **Softmax Confidence:** `{top['similarity_percentage']}%`\n"
        f"- **Inference Latency:** `{res['elapsed_time_ms']} ms`\n"
    )

    # Dictionary for Gradio Label output (normalized probabilities 0.0 - 1.0)
    scores_dict = {
        item["label"]: float(item["similarity_percentage"] / 100.0)
        for item in res["predictions"]
    }

    return summary_text, scores_dict


# Curated sample images list if available
sample_image_paths = []
sample_dir = CURRENT_DIR / "sample_images"
if sample_dir.exists():
    sample_image_paths = [[str(p)] for p in sample_dir.glob("*.jpg")]

# Build Gradio UI
with gr.Blocks(title="RareVision AI – Zero-Shot Species Classifier") as demo:
    gr.Markdown("# 🌿 RareVision AI – Zero-Shot Species Classifier")
    gr.Markdown(
        "**Third-Year B.Tech Computer Science Project** | "
        "**Model:** Pretrained OpenAI CLIP ViT-B/32 (Zero-Shot Multimodal Learning)"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Species Photo (Animal, Bird, or Plant)")
            
            prompt_style_select = gr.Radio(
                choices=["Standard ('a photograph of a {species}')", "Detailed Wildlife", "Scientific Context"],
                value="Standard ('a photograph of a {species}')",
                label="Prompt Template Style"
            )

            category_checkboxes = gr.CheckboxGroup(
                choices=all_predefined,
                value=["Bengal Tiger", "Snow Leopard", "Red Panda", "Asian Elephant", "Indian Peacock", "Great Hornbill", "Pitcher Plant"],
                label="Active Candidate Species Pool"
            )

            custom_prompts_input = gr.Textbox(
                lines=2,
                placeholder="Enter custom species (one per line, e.g. Malabar Gliding Frog)...",
                label="Add Custom Candidate Prompts (Optional)"
            )

            btn_analyze = gr.Button("🚀 Analyze Image", variant="primary")

        with gr.Column(scale=1):
            output_summary = gr.Markdown("### Prediction Results will appear here...")
            output_labels = gr.Label(num_top_classes=5, label="Top Matching Species Categories")

    if sample_image_paths:
        gr.Examples(
            examples=sample_image_paths,
            inputs=input_image,
            label="Curated Test Sample Images (Click to Test Instantly)"
        )

    btn_analyze.click(
        fn=classify_species,
        inputs=[input_image, category_checkboxes, custom_prompts_input, prompt_style_select],
        outputs=[output_summary, output_labels]
    )

if __name__ == "__main__":
    print("Launching RareVision AI web interface...", flush=True)
    demo.launch(inbrowser=True, server_port=7860, share=False)

