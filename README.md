# CV Perception Challenge 2025 – Google DeepMind  
**Efficient Video Question Answering for Hour-Long Videos**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/JAX-0.4%2B-green" alt="JAX">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Overview

This repository implements a **scalable Video Question Answering (VQA)** system designed for **hour-long multimodal videos**, as part of the **CV Perception Challenge 2025** by Google DeepMind.

We combine **state-of-the-art foundation models** with **custom temporal compression** to enable efficient processing and accurate reasoning over long-form video content.

### Core Components

| Component | Model | Role |
|--------|-------|------|
| **Video Encoder** | VideoPrism (v1-base) | Spatio-temporal video embedding |
| **Text Encoder + Alignment** | CLIP (ViT-B/32) | Zero-shot multimodal QA |
| **Temporal Compressor** | `IntermediateModel` (CNN decoder) | Compresses video into a single representative image |
| **Frame Sampler** | Temporal similarity (SSIM-based) | Reduces redundancy in long videos |

---

## Key Innovations

1. **Temporal Frame Compression**  
   - Uses structural similarity (SSIM) to sample only *informative* frames  
   - Reduces 1-hour video (~100k frames) → ~8–16 key frames  
   - Preserves semantic transitions and actions

2. **Learned Video-to-Image Mapping**  
   - `IntermediateModel`: Projects `[T, 768]` → reconstructed image  
   - Acts as a **visual summary generator** trained to retain QA-relevant content

3. **Zero-Shot VQA via CLIP**  
   - No task-specific fine-tuning  
   - Robust across diverse questions and domains


---

## Installation

```bash
# Clone the repo
git clone https://github.com/LeNotaCoder/Video_Question_Answering
cd Video_Question_Answering

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
