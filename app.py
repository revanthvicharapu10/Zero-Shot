"""RareVision AI – Zero-Shot Image Classification for Rare Species.

Academic Project: Third-Year B.Tech Computer Science and Engineering
Model Backbone: Pretrained OpenAI CLIP ViT-B/32 (openai/clip-vit-base-patch32)
Team Division:
  - Member 1: CLIP Model Loading & Image Embedding Generation
  - Member 2: Text Prompts, Similarity Scoring & Classification Logic
  - Member 3: Streamlit Web Application & Interactive UI/UX
  - Member 4: Testing, Validation, Dataset Curation & Academic Documentation
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import torch
from PIL import Image

# Ensure project modules are resolvable
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_SPECIES_CATEGORIES,
    DEFAULT_PROMPT_PREFIXES,
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_FILE_SIZE_MB,
    SAMPLE_IMAGES_DIR,
    PROJECT_INFO
)
from image_processor import (
    validate_image_file,
    load_image_safely,
    create_thumbnail,
    ImageProcessingError
)
from clip_model import load_clip_model_and_processor, get_model_metadata
from classification import compute_zero_shot_classification, encode_image


# =========================================================================
# Page Configuration and Custom CSS Styling
# =========================================================================

st.set_page_config(
    page_title="RareVision AI – Zero-Shot Species Classification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Modern Dark Theme Theme Overrides */
.main {
    background-color: #0b0f19;
}

/* Card Styling */
.metric-card {
    background: linear-gradient(135deg, #131b2e 0%, #17223b 100%);
    border: 1px solid #1f2e4d;
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    margin-bottom: 1rem;
}

.highlight-card {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 1.3rem;
    color: #ecfdf5;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.2);
    margin-bottom: 1.2rem;
}

.info-banner {
    background-color: #1e293b;
    border-left: 4px solid #38bdf8;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1.2rem;
    font-size: 0.92rem;
    color: #cbd5e1;
}

.warning-banner {
    background-color: #291c10;
    border-left: 4px solid #f59e0b;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1.2rem;
    font-size: 0.9rem;
    color: #fde68a;
}

.badge-tag {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.4rem;
}

.badge-endangered {
    background-color: #7f1d1d;
    color: #fecaca;
    border: 1px solid #ef4444;
}

.badge-vulnerable {
    background-color: #78350f;
    color: #fef3c7;
    border: 1px solid #f59e0b;
}

.badge-threatened {
    background-color: #3b0764;
    color: #f5d0fe;
    border: 1px solid #c084fc;
}

.badge-least-concern {
    background-color: #064e3b;
    color: #a7f3d0;
    border: 1px solid #10b981;
}

.stat-number {
    font-size: 1.9rem;
    font-weight: 700;
    color: #34d399;
    line-height: 1.1;
}

.stat-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
}

/* Tab text styling */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================================
# Session State Initialization
# =========================================================================

if "classification_history" not in st.session_state:
    st.session_state.classification_history = []

if "selected_sample_image" not in st.session_state:
    st.session_state.selected_sample_image = None

if "custom_labels" not in st.session_state:
    st.session_state.custom_labels = []


# =========================================================================
# Model Caching
# =========================================================================

@st.cache_resource(show_spinner="Loading OpenAI CLIP ViT-B/32 backbone into memory...")
def get_cached_clip_model():
    """Load CLIP model and processor with Streamlit resource caching."""
    return load_clip_model_and_processor(DEFAULT_MODEL_NAME)


# =========================================================================
# Sidebar Controls & Project Overview
# =========================================================================

with st.sidebar:
    st.markdown("### 🌿 RareVision AI")
    st.caption("B.Tech CSE Academic Computer Vision Project")
    
    st.markdown("---")
    st.markdown("#### ⚙️ CLIP Configuration")

    # Load model and display live hardware device
    try:
        model, processor, device = get_cached_clip_model()
        model_meta = get_model_metadata(model, device)
        st.success(f"🟢 CLIP Loaded ({model_meta['device']})", icon="✅")
        with st.expander("🔍 Model Architecture Specs", expanded=False):
            st.write(f"**Backbone:** {model_meta['vision_backbone']}")
            st.write(f"**Embedding Dim:** {model_meta['projection_dim']} dimensions")
            st.write(f"**Total Params:** {model_meta['total_parameters']}")
            st.write(f"**Input Resolution:** {model_meta['vision_image_size']}")
            st.write(f"**Patch Size:** {model_meta['vision_patch_size']}")
    except Exception as exc:
        st.error(f"Failed to load CLIP: {exc}")
        model, processor, device = None, None, None

    st.markdown("---")
    st.markdown("#### 🏷️ Candidate Species Selection")

    selected_groups = st.multiselect(
        "Species Kingdoms",
        options=["Animals", "Birds", "Plants"],
        default=["Animals", "Birds", "Plants"],
        help="Filter the predefined candidate categories evaluated by CLIP zero-shot classification."
    )

    prompt_style = st.selectbox(
        "Prompt Template",
        options=[
            "Standard ('a photograph of a {species}')",
            "Detailed Wildlife ('a high quality wildlife photo of a {species} in natural habitat')",
            "Scientific Context ('a close-up photo of {species} ({scientific_name})')",
            "Raw Name Only ('{species}')"
        ],
        index=0,
        help="Prompt engineering style affects text embedding projection."
    )

    st.markdown("---")
    st.markdown("#### ➕ Add Custom Candidate")
    new_custom_name = st.text_input(
        "Custom Species Name",
        placeholder="e.g. Malabar Gliding Frog",
        help="Add any rare species or custom label to evaluate zero-shot."
    )
    if st.button("Add to Candidate List", use_container_width=True) and new_custom_name.strip():
        trimmed = new_custom_name.strip()
        if trimmed not in st.session_state.custom_labels:
            st.session_state.custom_labels.append(trimmed)
            st.toast(f"Added '{trimmed}' to candidate species!", icon="🎉")

    if st.session_state.custom_labels:
        st.caption(f"Custom Candidates ({len(st.session_state.custom_labels)}):")
        for idx, cl in enumerate(st.session_state.custom_labels):
            cols = st.columns([4, 1])
            cols[0].text(f"• {cl}")
            if cols[1].button("✕", key=f"del_custom_{idx}", help=f"Remove {cl}"):
                st.session_state.custom_labels.pop(idx)
                st.rerun()

    st.markdown("---")
    st.markdown("#### 🎓 Team Roles (CSE 3rd Year)")
    st.caption("• Member 1: CLIP Backbone & Image Encoder\n• Member 2: Text Encoder & Cosine Similarity\n• Member 3: Streamlit UI & Visual Dashboard\n• Member 4: Validation & Academic Documentation")


# =========================================================================
# Main Dashboard Header
# =========================================================================

st.title("🌿 RareVision AI – Zero-Shot Species Classifier")
st.markdown(
    "**Third-Year B.Tech Computer Science Project** &nbsp;|&nbsp; "
    "**Pretrained Foundation Model:** `OpenAI CLIP ViT-B/32` &nbsp;|&nbsp; "
    "**Paradigm:** *Zero-Shot Contrastive Language-Image Learning*"
)

st.markdown(
    """
    <div class="info-banner">
        <strong>💡 Academic Note:</strong> Zero-shot classification recognizes unseen categories by projecting 
        image and text prompts into a shared 512-dimensional embedding space. No task-specific fine-tuning or 
        dataset retraining was conducted.
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation Tabs
tab_classify, tab_inspector, tab_history, tab_education, tab_team = st.tabs([
    "🔍 Live Classification",
    "📐 Embeddings & Math Inspector",
    "🕒 Session History",
    "📚 Educational & Academic Hub",
    "👥 Team & Architecture"
])


# =========================================================================
# Helper: Build Candidate Prompt List
# =========================================================================

def build_candidates_list(groups: List[str], custom_items: List[str], style: str):
    """Compile active candidate prompts, display names, and metadata lookup."""
    prompts = []
    labels = []
    metadata = {}

    for group in groups:
        for item in DEFAULT_SPECIES_CATEGORIES.get(group, []):
            name = item["name"]
            sci = item["scientific_name"]

            if "Standard" in style:
                prompt_text = f"a photograph of a {name}"
            elif "Detailed Wildlife" in style:
                prompt_text = f"a high quality wildlife photograph of a {name} in its natural habitat"
            elif "Scientific Context" in style:
                prompt_text = f"a close-up photo of {name} ({sci}), a rare wildlife species"
            else:  # Raw
                prompt_text = name

            prompts.append(prompt_text)
            labels.append(name)
            metadata[name] = item

    # Add custom user labels
    for custom_name in custom_items:
        if "Standard" in style:
            p_text = f"a photograph of a {custom_name}"
        elif "Detailed Wildlife" in style:
            p_text = f"a high quality wildlife photograph of a {custom_name} in its natural habitat"
        elif "Scientific Context" in style:
            p_text = f"a close-up photo of {custom_name}, a rare wildlife species"
        else:
            p_text = custom_name

        prompts.append(p_text)
        labels.append(custom_name)
        metadata[custom_name] = {
            "name": custom_name,
            "scientific_name": "Custom Candidate",
            "conservation_status": "User-defined",
            "description": "User-supplied candidate prompt for zero-shot hypothesis testing."
        }

    return prompts, labels, metadata


# =========================================================================
# TAB 1: Live Zero-Shot Classification
# =========================================================================

with tab_classify:
    col_upload, col_preview = st.columns([1, 1], gap="medium")

    active_image = None
    image_source_name = "None"

    with col_upload:
        st.subheader("1. Provide Species Image")

        # Tab for upload vs sample gallery
        input_method = st.radio(
            "Input Selection Mode:",
            ["📤 Upload File (Drag & Drop)", "🖼️ Select from Sample Gallery"],
            horizontal=True
        )

        if input_method == "📤 Upload File (Drag & Drop)":
            uploaded_file = st.file_uploader(
                "Upload a wildlife or plant photograph (JPG, JPEG, PNG, WEBP):",
                type=["jpg", "jpeg", "png", "webp"],
                help=f"Maximum file size: {MAX_IMAGE_FILE_SIZE_MB} MB."
            )
            if uploaded_file is not None:
                is_valid, err_msg = validate_image_file(uploaded_file)
                if not is_valid:
                    st.error(f"⚠️ {err_msg}")
                else:
                    try:
                        active_image = load_image_safely(uploaded_file)
                        image_source_name = uploaded_file.name
                        st.success(f"Loaded: `{image_source_name}` ({active_image.width}x{active_image.height} px)")
                    except ImageProcessingError as exc:
                        st.error(f"Image error: {exc}")

        else:
            st.markdown("Choose a curated sample species image for immediate testing:")
            # List available sample images in sample_images directory
            sample_files = []
            if SAMPLE_IMAGES_DIR.exists():
                sample_files = [f for f in SAMPLE_IMAGES_DIR.glob("*.*") if f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS]

            if sample_files:
                sample_options = {f.stem.replace("_", " ").title(): f for f in sample_files}
                selected_sample_key = st.selectbox("Curated Test Dataset:", list(sample_options.keys()))
                if selected_sample_key:
                    chosen_path = sample_options[selected_sample_key]
                    try:
                        active_image = load_image_safely(chosen_path)
                        image_source_name = chosen_path.name
                        st.info(f"Selected: **{selected_sample_key}** (`{chosen_path.name}`)")
                    except Exception as e:
                        st.error(f"Could not load sample: {e}")
            else:
                st.warning("No local sample images found in `sample_images/`. You can upload any image using the upload tab!")

        # Candidate descriptions breakdown preview
        candidate_prompts, candidate_labels, meta_lookup = build_candidates_list(
            selected_groups, st.session_state.custom_labels, prompt_style
        )

        with st.expander(f"📋 Active Candidate Pool ({len(candidate_labels)} Species)", expanded=False):
            st.caption("Each candidate is transformed into a 512-dimensional text embedding:")
            for l, p in zip(candidate_labels[:12], candidate_prompts[:12]):
                st.code(f"{l}  →  \"{p}\"", language="text")
            if len(candidate_labels) > 12:
                st.caption(f"...and {len(candidate_labels) - 12} more candidates.")

    with col_preview:
        st.subheader("2. Image Preview & Action")
        if active_image is not None:
            st.image(active_image, caption=f"Input: {image_source_name}", use_container_width=True)
            analyze_button = st.button(
                "🚀 Analyze Image (Execute CLIP Zero-Shot Inference)",
                type="primary",
                use_container_width=True,
                disabled=(model is None)
            )
        else:
            st.info("👈 Please upload an image or choose a sample to activate zero-shot classification.")
            analyze_button = False

    # Execution & Results Section
    if analyze_button and active_image is not None and model is not None:
        if len(candidate_prompts) == 0:
            st.warning("Please select at least one species category from the sidebar.")
        else:
            with st.spinner("Executing multimodal projection and cosine similarity ranking..."):
                results = compute_zero_shot_classification(
                    image=active_image,
                    candidate_prompts=candidate_prompts,
                    category_labels=candidate_labels,
                    metadata_lookup=meta_lookup,
                    processor=processor,
                    model=model,
                    device=device,
                    top_k=5
                )

            top_match = results["top_category"]

            # Save to session history
            thumbnail = create_thumbnail(active_image)
            st.session_state.classification_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_name": image_source_name,
                "top_category": top_match["label"],
                "similarity_score": round(top_match["cosine_similarity"], 4),
                "softmax_pct": top_match["similarity_percentage"],
                "status": top_match["conservation_status"],
                "thumbnail": thumbnail,
                "elapsed_ms": results["elapsed_time_ms"]
            })

            st.markdown("---")
            st.subheader("3. Zero-Shot Prediction Results")

            # Conservation disclaimer alert
            st.markdown(
                """
                <div class="warning-banner">
                    <strong>⚠️ Conservation & Classification Disclaimer:</strong> 
                    Predictions reflect relative visual alignment with CLIP candidate prompts and are not 
                    official biological identifications or verified IUCN red-list determinations.
                </div>
                """,
                unsafe_allow_html=True
            )

            # Top Prediction Highlight Card
            col_res_left, col_res_right = st.columns([1, 1], gap="medium")

            with col_res_left:
                st.markdown(
                    f"""
                    <div class="highlight-card">
                        <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.85;">
                            🏆 Top Ranked Candidate Match
                        </div>
                        <h2 style="margin: 0.3rem 0; font-size: 1.85rem; color: #ffffff;">{top_match['label']}</h2>
                        <div style="font-style: italic; opacity: 0.9; margin-bottom: 0.8rem;">
                            Scientific Name: {top_match['scientific_name']}
                        </div>
                        <div style="display: flex; gap: 1.5rem; margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 0.8rem;">
                            <div>
                                <div class="stat-label" style="color: #d1fae5;">Cosine Similarity</div>
                                <div class="stat-number" style="color: #ffffff;">{top_match['cosine_similarity']:.4f}</div>
                            </div>
                            <div>
                                <div class="stat-label" style="color: #d1fae5;">Softmax Confidence</div>
                                <div class="stat-number" style="color: #ffffff;">{top_match['similarity_percentage']}%</div>
                            </div>
                            <div>
                                <div class="stat-label" style="color: #d1fae5;">Inference Latency</div>
                                <div class="stat-number" style="color: #ffffff;">{results['elapsed_time_ms']} <span style="font-size: 1rem;">ms</span></div>
                            </div>
                        </div>
                        <div style="margin-top: 0.8rem; font-size: 0.88rem; background: rgba(0,0,0,0.2); padding: 0.5rem 0.8rem; border-radius: 6px;">
                            <strong>Conservation Reference:</strong> {top_match['conservation_status']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if top_match.get("description"):
                    st.caption(f"**Species Background:** {top_match['description']}")

            with col_res_right:
                st.markdown("#### Top 5 Candidate Probabilities")
                top5_df = pd.DataFrame(results["predictions"])[
                    ["rank", "label", "similarity_percentage", "cosine_similarity"]
                ].copy()
                top5_df.columns = ["Rank", "Species", "Softmax Probability (%)", "Cosine Similarity"]

                # Altair Bar Chart
                chart = alt.Chart(top5_df).mark_bar(cornerRadius=6).encode(
                    x=alt.X("Softmax Probability (%):Q", title="Relative Softmax Confidence (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Species:N", sort="-x", title="Species Candidate"),
                    color=alt.Color("Softmax Probability (%):Q", scale=alt.Scale(scheme="greens"), legend=None),
                    tooltip=["Rank", "Species", "Softmax Probability (%)", "Cosine Similarity"]
                ).properties(height=260)

                st.altair_chart(chart, use_container_width=True)

            # Detailed Top 5 Table Breakdown
            st.markdown("#### 📊 Detailed Top-5 Scoring Breakdown")
            detail_records = []
            for pred in results["predictions"]:
                detail_records.append({
                    "Rank": f"#{pred['rank']}",
                    "Species Category": pred["label"],
                    "Scientific Name": pred["scientific_name"],
                    "Cosine Similarity (Raw)": f"{pred['cosine_similarity']:.4f}",
                    "Softmax Distribution": f"{pred['similarity_percentage']:.2f}%",
                    "Status": pred["conservation_status"],
                    "Evaluated Text Prompt": f"\"{pred['prompt']}\""
                })

            st.dataframe(
                pd.DataFrame(detail_records),
                use_container_width=True,
                hide_index=True
            )


# =========================================================================
# TAB 2: Embeddings & Mathematical Inspector
# =========================================================================

with tab_inspector:
    st.subheader("📐 High-Dimensional Multimodal Embedding Inspector")
    st.markdown(
        """
        In CLIP's dual-encoder framework, both visual and linguistic signals are projected onto a shared 
        **512-dimensional unit hypersphere** $\\mathbb{S}^{511} \\subset \\mathbb{R}^{512}$.
        """
    )

    col_math1, col_math2 = st.columns([1, 1], gap="medium")

    with col_math1:
        st.markdown(
            r"""
            <div class="metric-card">
                <h4>1. Cosine Similarity Formulation</h4>
                <p>For normalized image vector $\mathbf{u} = \frac{f_v(I)}{\|f_v(I)\|_2}$ and text vector $\mathbf{v}_i = \frac{f_t(T_i)}{\|f_t(T_i)\|_2}$:</p>
                $$\cos(\theta_i) = \mathbf{u} \cdot \mathbf{v}_i = \sum_{k=1}^{512} u_k \cdot v_{i,k}$$
                <p>Because $\|\mathbf{u}\| = \|\mathbf{v}_i\| = 1$, cosine similarity equals the standard Euclidean dot product.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_math2:
        st.markdown(
            r"""
            <div class="metric-card">
                <h4>2. Scaled Softmax Probability</h4>
                <p>CLIP applies a learned logit scale parameter $\tau = \exp(\text{logit\_scale}) \approx 100.0$ to sharpen temperature:</p>
                $$P(Y = i \mid I) = \frac{\exp(\tau \cdot \cos(\theta_i))}{\sum_{j=1}^{N} \exp(\tau \cdot \cos(\theta_j))}$$
                <p>This transforms raw unbounded directional alignments into a valid probability distribution over the evaluated candidate set.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Live vector inspection if an image is loaded
    if active_image is not None and model is not None:
        st.markdown("#### 🔬 Live Embedding Vector Sample (First 64 Dimensions of 512)")
        with torch.no_grad():
            img_embed = encode_image(active_image, processor, model, device).cpu().numpy().squeeze(0)

        vec_df = pd.DataFrame({
            "Dimension": [f"Dim {i+1}" for i in range(64)],
            "Value": img_embed[:64]
        })

        vec_chart = alt.Chart(vec_df).mark_bar(width=8).encode(
            x=alt.X("Dimension:N", sort=None, title="Embedding Dimension (1 - 64)"),
            y=alt.Y("Value:Q", title="Normalized Feature Magnitude"),
            color=alt.condition(
                alt.datum.Value > 0,
                alt.value("#10b981"),
                alt.value("#f43f5e")
            ),
            tooltip=["Dimension", "Value"]
        ).properties(height=240)

        st.altair_chart(vec_chart, use_container_width=True)

        col_norm1, col_norm2, col_norm3 = st.columns(3)
        col_norm1.metric("L2 Vector Norm", f"{np.linalg.norm(img_embed):.6f}", "Unit Normalized")
        col_norm2.metric("Embedding Dimensionality", f"{len(img_embed)}", "ViT-B/32 Projection")
        col_norm3.metric("Vector Min / Max", f"{img_embed.min():.3f} / {img_embed.max():.3f}")
    else:
        st.info("Upload an image in the 'Live Classification' tab to inspect its live 512-dimensional embedding vector.")


# =========================================================================
# TAB 3: Session Classification History
# =========================================================================

with tab_history:
    st.subheader("🕒 Session Classification History")
    st.caption("Logs all queries executed during the current active session.")

    if not st.session_state.classification_history:
        st.info("No classification runs recorded in this session yet. Upload and analyze an image to build history.")
    else:
        col_h_ctrl1, col_h_ctrl2 = st.columns([1, 4])
        with col_h_ctrl1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.classification_history = []
                st.rerun()

        with col_h_ctrl2:
            # Prepare exportable DataFrame
            export_df = pd.DataFrame([
                {
                    "Timestamp": h["timestamp"],
                    "Image Name": h["image_name"],
                    "Top Prediction": h["top_category"],
                    "Cosine Similarity": h["similarity_score"],
                    "Softmax Percentage": f"{h['softmax_pct']}%",
                    "Conservation Status": h["status"],
                    "Latency (ms)": h["elapsed_ms"]
                }
                for h in st.session_state.classification_history
            ])
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export History as CSV",
                data=csv_data,
                file_name=f"rarevision_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        st.markdown("---")
        for idx, item in enumerate(reversed(st.session_state.classification_history)):
            c_thumb, c_info = st.columns([1, 5])
            with c_thumb:
                if item.get("thumbnail"):
                    st.image(item["thumbnail"], width=110)
            with c_info:
                st.markdown(
                    f"""
                    **{item['top_category']}** &nbsp;|&nbsp; 
                    `Similarity: {item['similarity_score']:.4f}` &nbsp;|&nbsp; 
                    `Softmax: {item['softmax_pct']}%` &nbsp;|&nbsp;
                    `Time: {item['timestamp']}`
                    <br><span style="font-size: 0.85rem; color: #94a3b8;">Source File: {item['image_name']} &bull; Latency: {item['elapsed_ms']} ms &bull; Status: {item['status']}</span>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("<hr style='margin: 0.5rem 0; border-color: #1e293b;'>", unsafe_allow_html=True)


# =========================================================================
# TAB 4: Educational & Academic Hub
# =========================================================================

with tab_education:
    st.subheader("📚 Academic Principles & Technical Foundations")

    with st.expander("1. What is CLIP (Contrastive Language-Image Pretraining)?", expanded=True):
        st.markdown(
            """
            **CLIP** is an open-source multimodal neural network developed by OpenAI (Radford et al., 2021). 
            Unlike traditional deep learning classifiers that are restricted to fixed discrete classes (e.g., ImageNet's 1,000 labels), 
            CLIP learns to jointly align **visual content** and **free-form natural language** from 400 million internet image-text pairs.
            
            Key components:
            - **Vision Encoder:** A Vision Transformer (ViT-B/32) that splits an input image of size $224 \\times 224$ into non-overlapping $32 \\times 32$ patches, projects them linearly, adds positional embeddings, and applies multi-head self-attention.
            - **Text Encoder:** A Transformer that tokenizes text prompts (using Byte Pair Encoding, BPE) and projects the sequence to the same latent dimension.
            - **Joint Multimodal Embedding Space:** Both encoders project their respective representations to $\\mathbb{R}^{512}$.
            """
        )

    with st.expander("2. What Does 'Zero-Shot Classification' Mean?", expanded=True):
        st.markdown(
            """
            In traditional computer vision, recognizing a new species (such as a *Snow Leopard* or *Neelakurinji*) requires:
            1. Collecting hundreds or thousands of labeled training photographs.
            2. Training a softmax classifier with cross-entropy loss.
            3. Risking severe overfitting on rare classes where labeled samples are scarce.

            **In Zero-Shot Learning:**
            - No new parameters or weights are trained.
            - The model evaluates the semantic affinity between the image embedding and candidate textual hypotheses (e.g., *"a photograph of a Snow Leopard"*).
            - The model generalizes directly to arbitrary, unseen species categories at inference time.
            """
        )

    with st.expander("3. How Does Contrastive Learning Work?", expanded=False):
        st.markdown(
            """
            During pretraining, a batch of $N$ (image, text) pairs is fed simultaneously through the dual encoders:
            - There are $N$ positive (matching) pairs along the diagonal.
            - There are $N^2 - N$ negative (non-matching) pairs off the diagonal.
            - The model optimizes a symmetric InfoNCE loss that maximizes the cosine similarity of true matches while penalizing negative pairs:
            
            $$\\mathcal{L} = -\\frac{1}{2N} \\sum_{i=1}^N \\left( \\log \\frac{\\exp(\\tau \\cdot \\mathbf{u}_i \\cdot \\mathbf{v}_i)}{\\sum_j \\exp(\\tau \\cdot \\mathbf{u}_i \\cdot \\mathbf{v}_j)} + \\log \\frac{\\exp(\\tau \\cdot \\mathbf{u}_i \\cdot \\mathbf{v}_i)}{\\sum_j \\exp(\\tau \\cdot \\mathbf{u}_j \\cdot \\mathbf{v}_i)} \\right)$$
            """
        )

    with st.expander("4. Why ViT-B/32 Over Traditional CNNs?", expanded=False):
        st.markdown(
            """
            Traditional Convolutional Neural Networks (e.g. ResNet) process images through local receptive fields with inductive spatial biases.
            In contrast, the **Vision Transformer (ViT-B/32)**:
            - Treats image patches as a sequence of tokens analogous to words in natural language processing.
            - Uses global multi-head self-attention layers from the very first layer, enabling the model to relate distant parts of the animal/plant simultaneously (e.g., matching a tiger's stripes across body and face).
            """
        )

    with st.expander("5. Known Limitations of Zero-Shot Classification", expanded=False):
        st.markdown(
            """
            For academic viva evaluations, it is crucial to recognize these documented boundaries:
            - **Fine-Grained Visual Distinctions:** Sub-species with subtle morphologic variations (e.g. Bengal Tiger vs Siberian Tiger) can yield similar cosine similarities.
            - **Background Bias:** Pretrained models can associate contextual cues (e.g., snow, rocky crags) with specific species, occasionally matching context rather than the organism.
            - **Text Prompt Sensitivity:** Phrasing differences (*"a photo of a tiger"* vs *"a wild panthera tigris"*) yield slightly varying embeddings.
            """
        )


# =========================================================================
# TAB 5: Team Division & System Architecture
# =========================================================================

with tab_team:
    st.subheader("👥 Academic Team Division & System Pipeline")
    st.markdown("Third-Year B.Tech Computer Science and Engineering Academic Project")

    col_t1, col_t2 = st.columns([1, 1], gap="medium")

    with col_t1:
        st.markdown(
            r"""
            <div class="metric-card">
                <h4>👨‍💻 Team Responsibilities Matrix</h4>
                <ul>
                    <li><strong>Team Member 1:</strong> Pretrained CLIP model integration, GPU/CPU device management, model evaluation state, and image feature tensor extraction.</li>
                    <li><strong>Team Member 2:</strong> Text prompt tokenization, high-dimensional cosine similarity matrix calculation, temperature scaling, and softmax ranking algorithm.</li>
                    <li><strong>Team Member 3:</strong> Streamlit web application frontend, dashboard layout, responsive UI/UX, Altair visualization charts, and session history management.</li>
                    <li><strong>Team Member 4:</strong> Test suite development (pytest), image format validation, sample dataset curation, viva documentation, and README formulation.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_t2:
        st.markdown(
            r"""
            <div class="metric-card">
                <h4>🏗️ System Architecture Flowchart</h4>
                <ol>
                    <li><strong>Input Acquisition:</strong> User uploads JPG/PNG or selects curated sample image.</li>
                    <li><strong>Safety Validation:</strong> Verify file headers, MIME type, and size (&le; 15 MB).</li>
                    <li><strong>Image Preprocessing:</strong> Resizing to 224x224, bicubic interpolation, ImageNet normalization.</li>
                    <li><strong>Dual Encoding:</strong> 
                        <ul>
                            <li>Image &rarr; ViT-B/32 &rarr; 512-dim unit vector $\mathbf{u}$.</li>
                            <li>$N$ Text Prompts &rarr; Text Transformer &rarr; $N \\times 512$ unit vectors $\mathbf{v}$.</li>
                        </ul>
                    </li>
                    <li><strong>Similarity Computation:</strong> Matrix dot product $\mathbf{S} = \mathbf{u} \\cdot \mathbf{V}^T$.</li>
                    <li><strong>Post-Processing:</strong> Logit temperature scaling ($100 \\times$) &rarr; Softmax &rarr; Top-5 ranking.</li>
                    <li><strong>UI Rendering:</strong> Present predictions, confidence bar chart, and update session history.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================================
# Footer
# =========================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        <strong>RareVision AI</strong> &bull; Third-Year B.Tech Computer Science Academic Project &bull; 
        Powered by OpenAI CLIP & Hugging Face Transformers &bull; No Model Training Required
    </div>
    """,
    unsafe_allow_html=True
)

