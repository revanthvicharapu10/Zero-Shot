# RareVision AI – Zero-Shot Image Classification for Rare Species

**An Academic Computer Vision & Multimodal Deep Learning Project**  
*Third-Year B.Tech Computer Science and Engineering (CSE)*  
*Department of Computer Science & Engineering*

---

## 1. Project Abstract

In biodiversity conservation and ecological research, identifying rare, endangered, or elusive flora and fauna is critical for monitoring ecosystems and combating illegal wildlife trade. However, traditional supervised deep learning classifiers suffer from a major bottleneck: they require hundreds to thousands of meticulously labeled training photographs for every single species. For endangered or newly discovered species, such extensive labeled datasets simply do not exist.

**RareVision AI** presents a zero-shot image classification framework that eliminates the need for task-specific training, fine-tuning, or manual data labeling. Leveraging OpenAI's pretrained **CLIP (Contrastive Language–Image Pretraining)** model with a Vision Transformer backbone (`ViT-B/32`), RareVision AI bridges visual and linguistic modalities. An input photograph of an animal, bird, or plant is mapped into a shared 512-dimensional multimodal latent space alongside natural language candidate descriptions. By calculating the normalized cosine similarity between the image embedding and candidate text embeddings, the system ranks the candidate species in real time, delivering explainable predictions, similarity distributions, and confidence analytics through an interactive web application.

---

## 2. Introduction

Species classification has traditionally been formulated as a closed-set supervised pattern recognition task. While state-of-the-art Convolutional Neural Networks (CNNs) achieve high accuracy on standard benchmarks such as ImageNet-1K or iNaturalist, their utility collapses when applied to "long-tail" or rare species where training exemplars are extremely scarce or entirely absent.

Zero-Shot Learning (ZSL) transforms classification from closed-vocabulary label assignment to open-vocabulary semantic alignment. By exploiting multimodal foundation models trained on broad internet-scale image-text pairs, zero-shot systems generalize immediately to novel, unseen categories described purely in natural language. RareVision AI operationalizes this paradigm for wildlife and botanical recognition, offering an accessible, deployable, and educational platform for academic demonstration.

---

## 3. Problem Statement

Manual taxonomic identification of rare wildlife requires specialized zoological or botanical expertise and is impractical for large-scale field surveillance. While artificial intelligence can automate identification, conventional computer vision models present severe limitations:
1. **Extreme Data Scarcity:** Rare species have very few high-quality photographs, leading to severe overfitting in standard supervised neural networks.
2. **Fixed Vocabulary:** Standard models cannot recognize a species outside their hardcoded training classes without undergoing expensive retraining.
3. **High Computational Overhead:** Retraining multi-million parameter neural networks requires specialized GPU clusters and hours of training time.

There is a distinct need for a lightweight, zero-training solution that accepts arbitrary visual inputs, compares them against dynamically configured candidate categories, and provides real-time probabilistic rankings.

---

## 4. Project Objectives

- **Zero-Shot Generalization:** Identify candidate species without collecting training datasets or executing model weight updates.
- **Pretrained Foundation Backbone:** Implement OpenAI CLIP (`ViT-B/32`) using Hugging Face Transformers and PyTorch.
- **Dynamic Category Expansion:** Allow users to choose from curated sample categories (Animals, Birds, Plants) or enter arbitrary custom species names on the fly.
- **Multimodal Embedding Generation:** Extract and normalize 512-dimensional vector representations for both visual and linguistic inputs.
- **Mathematical Transparency:** Compute cosine similarity and temperature-scaled softmax distributions to illustrate the mathematical foundations of multimodal AI.
- **Educational Web Dashboard:** Deliver a modern, beginner-friendly Streamlit web application complete with visual analytics, vector inspectors, and session history tracking.

---

## 5. Existing System vs. Proposed System

| Feature | Existing Supervised Systems (ResNet, EfficientNet) | Proposed RareVision AI (CLIP Zero-Shot) |
| :--- | :--- | :--- |
| **Model Training** | Required (100–1,000+ images per class) | **Zero training / No fine-tuning required** |
| **Classification Paradigm** | Closed-set fixed classes ($N$ fixed output nodes) | **Open-vocabulary dynamic textual hypotheses** |
| **Rare Species Adaptability** | Extremely poor (fails due to severe data scarcity) | **High (transfers internet-scale multimodal knowledge)** |
| **Adding a New Species** | Requires gathering data, re-annotating, and retraining | **Instant (type species name into text prompt)** |
| **Modality** | Visual only | **Joint Multimodal (Vision + Natural Language)** |
| **Computational Footprint** | Heavy training compute required | **Lightweight inference only (runs on standard CPU/GPU)** |

---

## 6. Technology Stack

- **Programming Language:** Python 3.10+ (tested on Python 3.13)
- **Deep Learning Framework:** PyTorch (`torch>=2.0.0`)
- **Multimodal Backbone:** Hugging Face Transformers (`transformers>=4.30.0`) / OpenAI CLIP
- **Computer Vision Model:** `openai/clip-vit-base-patch32` (ViT-B/32, 512-dim embedding projection)
- **Image Processing:** Pillow (`PIL>=9.5.0`), NumPy (`numpy>=1.24.0`)
- **Web Application & UI:** Streamlit (`streamlit>=1.30.0`), Altair (`altair>=5.0.0`), Pandas (`pandas>=2.0.0`)
- **Testing & Quality Assurance:** Pytest (`pytest>=7.0.0`)

---

## 7. System Architecture

```
                      +-----------------------------+
                      |   User Input: Image File    |
                      |  (Upload or Sample Gallery) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Image Processor (PIL)     |
                      |  - Validate Format & Size   |
                      |  - Transpose EXIF Orientation|
                      |  - Convert to RGB           |
                      |  - Resize / Normalize       |
                      +--------------+--------------+
                                     |
                                     v
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
   +---------------------------+           +---------------------------+
   |   Vision Transformer      |           |     Text Transformer      |
   |   (ViT-B/32 Backbone)     |           |     (BPE Tokenizer)       |
   |   - 224x224 Resolution    |           |   - Candidate Prompts:    |
   |   - 32x32 Patch Slices    |           |     "a photo of a {X}"    |
   |   - Self-Attention Layers |           |   - Length <= 77 tokens   |
   |   - Linear Projection     |           |   - Linear Projection     |
   +-------------+-------------+           +-------------+-------------+
                 |                                       |
                 v                                       v
   +---------------------------+           +---------------------------+
   |  Image Embedding (u)      |           |   Text Embeddings (v_i)   |
   |  Shape: (1, 512)          |           |   Shape: (N, 512)         |
   |  L2 Normalized: ||u|| = 1 |           |   L2 Normalized: ||v|| = 1|
   +-------------+-------------+           +-------------+-------------+
                 |                                       |
                 +-------------------+-------------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Cosine Similarity Matrix   |
                      |        S_i = u . v_i        |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Temperature Logit Scaling  |
                      |      logits = tau * S_i     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Softmax Normalization    |
                      |   P_i = exp(s_i) / sum(...) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Ranking & UI Visualization  |
                      | - Top-5 Matching Candidates |
                      | - Interactive Bar Chart     |
                      | - Session History Logger    |
                      +-----------------------------+
```

---

## 8. Theoretical Foundations

### 8.1 CLIP (Contrastive Language–Image Pretraining)
Developed by OpenAI (Radford et al., 2021), CLIP trains a vision encoder and a text encoder simultaneously to predict which image corresponds to which text prompt in a batch of $N$ (image, text) pairs. 

### 8.2 Contrastive Learning Mechanism
Given a batch of $N$ pairs $\{(I_i, T_i)\}_{i=1}^N$:
- The $N$ diagonal entries $(I_i, T_i)$ represent positive pairs.
- The $N^2 - N$ off-diagonal entries $(I_i, T_j)$ ($i \neq j$) represent negative pairs.
- Both encoders are trained to maximize the cosine similarity of the true $N$ pairs while minimizing the similarity of incorrect pairs using a symmetric cross-entropy loss over similarity scores:

$$\mathcal{L} = -\frac{1}{2N} \sum_{i=1}^N \left( \log \frac{\exp(\tau \cdot \mathbf{u}_i \cdot \mathbf{v}_i)}{\sum_{j=1}^N \exp(\tau \cdot \mathbf{u}_i \cdot \mathbf{v}_j)} + \log \frac{\exp(\tau \cdot \mathbf{u}_i \cdot \mathbf{v}_i)}{\sum_{j=1}^N \exp(\tau \cdot \mathbf{u}_j \cdot \mathbf{v}_i)} \right)$$

where $\tau$ is a learned temperature parameter ($\approx 100.0$).

### 8.3 Image and Text Embeddings
1. **Vision Encoder:** The Vision Transformer divides an input image $I \in \mathbb{R}^{224 \times 224 \times 3}$ into $7 \times 7 = 49$ non-overlapping patches of $32 \times 32$ pixels. Each patch is flattened and linearly projected into a hidden dimension ($D = 768$). A learnable class token (`[CLS]`) and 1D positional embeddings are added. After 12 multi-head self-attention layers, the `[CLS]` token is projected to 512 dimensions and L2-normalized:

$$\mathbf{u} = \frac{f_v(I)}{\|f_v(I)\|_2} \in \mathbb{R}^{512}, \quad \|\mathbf{u}\|_2 = 1$$

2. **Text Encoder:** A candidate description $T$ is tokenized into Byte Pair Encoding (BPE) subwords. Positional encodings are added. After 12 transformer layers, the feature corresponding to the `[EOS]` (end-of-sequence) token is linearly projected to 512 dimensions and L2-normalized:

$$\mathbf{v} = \frac{f_t(T)}{\|f_t(T)\|_2} \in \mathbb{R}^{512}, \quad \|\mathbf{v}\|_2 = 1$$

### 8.4 Cosine Similarity & Softmax Probability
Because both visual and linguistic vectors lie on the unit hypersphere $\mathbb{S}^{511} \subset \mathbb{R}^{512}$, their cosine similarity is computed directly by the dot product:

$$\cos(\theta_i) = \mathbf{u} \cdot \mathbf{v}_i = \sum_{k=1}^{512} u_k \cdot v_{i,k} \in [-1, 1]$$

To convert raw geometric alignments into an intuitive relative probability distribution over the evaluated candidate set:

$$P(Y = i \mid I) = \frac{\exp(\tau \cdot \cos(\theta_i))}{\sum_{j=1}^N \exp(\tau \cdot \cos(\theta_j))}$$

---

## 9. Team Division (4-Member Academic Structure)

| Member | Specific Module | Key Deliverables & Responsibilities |
| :--- | :--- | :--- |
| **Team Member 1** | `clip_model.py` & Device Management | Pretrained model initialization, CUDA/CPU device routing, model evaluation mode (`model.eval()`), parameter inspection, and image feature tensor extraction. |
| **Team Member 2** | `classification.py` & Algorithms | Prompt engineering templates, text tokenization, matrix dot products, logit temperature scaling, softmax distribution calculation, and candidate ranking. |
| **Team Member 3** | `app.py` & Frontend UI/UX | Streamlit web application layout, dark-theme styling, metric cards, Altair similarity charts, file uploader, and session history management. |
| **Team Member 4** | `tests/`, `image_processor.py`, Documentation | Pytest test suite, file size/format validation, curated sample image dataset, project report, and academic viva presentation guide. |

---

## 10. Predefined Species Database

The application contains sample candidate categories across three primary wildlife kingdoms for testing and demonstration:

### 10.1 Animals
- **Bengal Tiger** (*Panthera tigris tigris*) – Endangered (IUCN)
- **Snow Leopard** (*Panthera uncia*) – Vulnerable (IUCN)
- **Red Panda** (*Ailurus fulgens*) – Endangered (IUCN)
- **Asian Elephant** (*Elephas maximus*) – Endangered (IUCN)
- **Indian Pangolin** (*Manis crassicaudata*) – Endangered (IUCN)
- **Black Panther** (*Panthera pardus melanistic*) – Vulnerable / Melanistic morph
- **Great Indian Bustard** (*Ardeotis nigriceps*) – Critically Endangered (IUCN)

### 10.2 Birds
- **Indian Peacock** (*Pavo cristatus*) – Least Concern (National Bird of India)
- **Great Hornbill** (*Buceros bicornis*) – Vulnerable (IUCN)
- **Indian Vulture** (*Gyps indicus*) – Critically Endangered (IUCN)
- **Greater Flamingo** (*Phoenicopterus roseus*) – Least Concern (IUCN)
- **Siberian Crane** (*Leucogeranus leucogeranus*) – Critically Endangered (IUCN)

### 10.3 Plants
- **Blue Vanda Orchid** (*Vanda coerulea*) – Threatened / CITES Appendix II
- **Neelakurinji** (*Strobilanthes kunthiana*) – Endemic Western Ghats shrub (blooms every 12 years)
- **Ghost Orchid** (*Dendrophylax lindenii*) – Critically Endangered / Rare leafless epiphyte
- **Pitcher Plant** (*Nepenthes khasiana*) – Endangered Carnivorous Flora (CITES Appendix I)
- **Red Sandalwood** (*Pterocarpus santalinus*) – Endangered Hardwood (IUCN)

*Important Academic Disclaimer:* Predefined sample species categories are curated for demonstration and zero-shot hypothesis testing. Predictions reflect visual alignment with candidate prompts and do not constitute certified biological taxonomic determinations or automated IUCN conservation classifications.

---

## 11. Project Directory Structure

```
rarevision_ai/
│
├── app.py                     # Main Streamlit web application dashboard
├── clip_model.py              # CLIP model loading and device initialization
├── image_processor.py         # Image validation, EXIF transposition, preprocessing
├── classification.py          # Zero-shot embedding calculation & ranking pipeline
├── config.py                  # System constants, species metadata, prompt templates
├── requirements.txt           # Python package dependencies
├── README.md                  # Comprehensive academic project documentation
│
├── sample_images/             # Curated sample species photographs for live viva demo
│   ├── bengal_tiger.jpg
│   ├── snow_leopard.jpg
│   ├── red_panda.jpg
│   ├── great_hornbill.jpg
│   ├── pitcher_plant.jpg
│   └── neelakurinji.jpg
│
└── tests/                     # Unit and integration test suite
    └── test_classification.py # Pytest validation tests
```

---

## 12. Installation and Setup Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Recommended: 8 GB RAM, optional CUDA GPU.

### Step 1: Clone or Navigate to Project Directory
```bash
cd rarevision_ai
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Test Suite
```bash
pytest tests/test_classification.py -v
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 13. Viva Voce & Academic Q&A Guide

### Q1: Why is this called "Zero-Shot" Classification?
**Answer:** It is called zero-shot because the model has never been trained, fine-tuned, or given labeled exemplar images for the specific downstream task of rare species classification. It classifies instances purely by projecting arbitrary natural language concepts and image pixels into a shared semantic latent space.

### Q2: What is the embedding dimension of CLIP ViT-B/32?
**Answer:** Both the vision encoder and text encoder project their outputs into a **512-dimensional vector space** ($\mathbb{R}^{512}$). The vectors are normalized using the Euclidean $L_2$ norm so that $\|\mathbf{u}\|_2 = 1.0$.

### Q3: Why does cosine similarity equal the dot product in this model?
**Answer:** The cosine similarity formula is:
$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
Because both $\mathbf{u}$ and $\mathbf{v}$ are normalized to unit length ($\|\mathbf{u}\|_2 = 1$ and $\|\mathbf{v}\|_2 = 1$), the denominator equals $1 \times 1 = 1$. Consequently, cosine similarity simplifies directly to the dot product $\mathbf{u} \cdot \mathbf{v}$.

### Q4: Why is Vision Transformer (ViT) preferred over traditional CNNs in CLIP?
**Answer:** Vision Transformers split images into patch tokens and apply multi-head self-attention. This allows the model to compute global, all-to-all relationships between image regions across the entire image simultaneously, whereas CNNs rely on local convolutional filters with constrained receptive fields.

### Q5: What are the main limitations of this system?
**Answer:** 
1. Sub-species with extremely subtle morphological differences (e.g., Siberian Tiger vs. Bengal Tiger) can yield very close cosine similarities.
2. Background bias: if an animal is in an unusual environment (e.g. an elephant in snow), context tokens may compete with the object features.
3. Prompt phrasing: prompt engineering slightly shifts the text embedding vector.

---

## 14. Future Enhancements

1. **Species Localization with Bounding Boxes:** Incorporate open-vocabulary object detectors (such as Grounding DINO or OWL-ViT) to locate multiple rare organisms in a single scene before classification.
2. **Hierarchical Taxonomic Knowledge Integration:** Incorporate biological taxonomy trees (Kingdom $\to$ Phylum $\to$ Class $\to$ Order $\to$ Family $\to$ Genus $\to$ Species) to constrain zero-shot candidate pruning.
3. **Audio-Visual Multimodal Expansion:** Integrate bioacoustic foundation models (e.g. CLAP) to classify rare birds and primates from vocalizations alongside camera trap imagery.
4. **Offline Edge Deployment:** Quantize CLIP to INT8/FP16 format for deployment on solar-powered field microcontrollers (Raspberry Pi, NVIDIA Jetson) for remote forest surveillance.

---

## 15. References & Academic Citations

1. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. International Conference on Machine Learning (ICML).
2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR.
3. IUCN (International Union for Conservation of Nature). *The IUCN Red List of Threatened Species*. Version 2024-2.
4. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP System Demonstrations.

