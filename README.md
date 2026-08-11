---
title: SchizBrain
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🧠 SchizBrain: AI-Assisted Schizophrenia Detection from Structural MRI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SchizBrain** is an end-to-end deep learning framework designed to detect Schizophrenia from T1-weighted NIfTI 3D structural brain scans. It utilizes a state-of-the-art Hybrid Convolutional Neural Network (CNN) and Vision Transformer (ViT) architecture, deeply fused with clinical metadata (Age and Gender) to provide robust, interpretable predictions.

## 🔬 Core Architecture

The core model (`SchizoBrain`) relies on a two-stream processing pipeline before utilizing late-fusion at the classification head:

1. **Local Feature Extraction (CNN)**: Uses a 3D ResNet-50 backbone pre-trained on [MedicalNet](https://github.com/Tencent/MedicalNet). This component acts as a highly effective feature extractor for volumetric medical imaging.
2. **Global Context (ViT)**: The high-level CNN features are tokenized and processed by a 6-layer Vision Transformer, allowing the model to capture long-range structural dependencies across distinct brain regions.
3. **Clinical Metadata Fusion**: Demographic variables (Normalized Age and Encoded Gender) are embedded via a localized MLP and late-fused with the global brain representations.

<p align="center">
  <img src="docs/architecture.png" alt="SchizBrain Architecture" width="80%">
</p>
*(Placeholder for Architecture Diagram)*

## ✨ Key Features

- **Hybrid CNN+ViT Pipeline**: Combines local texture recognition with global topological modeling.
- **Clinical Explainability (XAI)**: Includes Grad-CAM (CNN attention localization) and Attention Rollout (ViT global patch attendance) modules so medical professionals can verify AI decisions.
- **Robust Training Engine**: Leverages `FocalLoss` for dataset class imbalances, stochastic depth (DropPath), and intensive TorchIO 3D medical augmentations (MixUp, affine transformations, noise).
- **FastAPI Web Interface**: Includes a lightweight, containerized frontend allowing users to upload `.nii.gz` files, run autonomous Python-only skull-stripping, and generate clinical PDF reports.

## 📂 Repository Structure

The project has been heavily refactored for a clean, modular, and research-grade layout:

```text
schizbrain/
├── src/                      # Core Python Package
│   ├── api/                  # FastAPI web server & endpoints
│   ├── data/                 # Data loading, MRIDataset, transformations, skull_strip
│   ├── evaluation/           # Grad-CAM, Attention Maps, metrics 
│   ├── model/                # Model architectures (CNN Blocks, ViT layers, Hybrid Fusion)
│   └── training/             # Train/Val loops, FocalLoss, Schedulers
│
├── train.py                  # Entry point for model training & cross-validation
├── eval.py                   # Entry point for ensemble evaluation
├── Dockerfile                # Docker environment configuration
└── requirements.txt          # Package dependencies
```

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies (PyTorch 2.6+ recommended):

```bash
git clone https://github.com/byreddyrohanreddy/schizbrain.git
cd schizbrain
pip install -r requirements.txt
```

### 2. Dataset Preparation

Ensure your MRI data is processed into normalized `(1, 96, 96, 96)` tensors or raw `.nii.gz` files. 
Create a metadata CSV (`data/metadata_pt.csv`) with the following columns:

```csv
filepath,label,age,gender,site
data/processed/scan_001.nii.gz,0,34,M,0
data/processed/scan_002.nii.gz,1,25,F,1
```
*(Label: `0` for Healthy, `1` for Schizophrenia)*

### 3. Model Training

Start the 5-fold Stratified Cross-Validation training loop. The script automatically handles MixUp augmentation and Early Stopping.

```bash
python train.py
```
Checkpoints will be automatically saved under `experiments/checkpoints/`.

### 4. Running the Web Server (Clinical UI)

Deploy the web application locally to upload MRI scans and generate predictions:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Navigate to `http://localhost:8000` in your web browser.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Citation

If you use SchizBrain in your research, please cite:

```bibtex
@misc{reddy2026schizbrain,
  author = {Rohan Reddy Byreddy},
  title = {SchizBrain: AI-Assisted Schizophrenia Detection from Structural MRI},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/byreddyrohanreddy/schizbrain}}
}
```
