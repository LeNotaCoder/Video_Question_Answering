# Video-Question Answering with CLIP + LSTM

A **Video QA** model that combines **CLIP** (for visual and textual understanding) with an **LSTM** to process temporal video frame embeddings. The model learns to answer multiple-choice questions about video content by aligning video representations with text options in CLIP's joint embedding space.

---
<img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python"> <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch"> <img src="https://img.shields.io/badge/JAX-0.4%2B-green" alt="JAX"> <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
---

## Overview

This project implements a **zero-shot-like** video question-answering system using:
- **CLIP** (`openai/clip-vit-base-patch32`) for encoding video frames and text options.
- **LSTM** to aggregate temporal information from frame-level CLIP embeddings.
- **Cosine similarity** in CLIP space to score text options against the video context.

The model is trained end-to-end on video-question-option triples using **cross-entropy loss** over similarity scores.

---

## Features

- Extracts **CLIP visual embeddings** per frame using `av` and `PIL`.
- Handles **variable-length videos** via padding in a custom collate function.
- Uses **frozen CLIP** (no fine-tuning) + trainable **LSTM + projection head**.
- Supports **batch processing** with GPU acceleration.
- Checkpointing for incremental training across data chunks.

---

## Requirements

```bash
pip install torch torchvision transformers pillow opencv-python matplotlib av pandas

```
---
