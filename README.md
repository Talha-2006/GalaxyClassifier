# 🌌 Galaxy Morphology Classifier

> *Teaching machines to see the universe — one spiral arm at a time.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Dataset](https://img.shields.io/badge/Dataset-Galaxy_Zoo_2-8B5CF6?style=flat-square)](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge)
[![Status](https://img.shields.io/badge/Status-In_Development-F59E0B?style=flat-square)]()

---

## What Is This?

This project trains a **Convolutional Neural Network (CNN)** to classify galaxies by their morphological type — whether a galaxy is **elliptical**, **spiral**, or a **merger/artifact** — using real citizen-science data from the [Galaxy Zoo 2](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) Kaggle competition.

Galaxy morphology is a fundamental astrophysical property. The shape of a galaxy encodes its formation history, star-formation rate, and environmental interactions. Automating this classification at scale is an active research problem — this project builds a deep learning pipeline to tackle it.

---

## Results

| Metric | Value |
|--------|-------|
| Task | 3-class galaxy morphology classification |
| Dataset | Galaxy Zoo 2 — 61,578 training images |
| Baseline Accuracy | *In training* |
| Model | Custom CNN (built from scratch) |
| Hardware | NVIDIA RTX 3090 (CUDA 12.6) |

> Final accuracy figures will be updated upon training completion.

---

## The Dataset

**Galaxy Zoo 2** is a citizen science dataset where hundreds of thousands of volunteers classified galaxy images from the Sloan Digital Sky Survey (SDSS). Each galaxy was answered against an 11-question decision tree, producing **37 soft probabilistic label values** per image — representing the fraction of volunteers who chose each answer.

### Label Engineering

The raw labels are *probabilistic*, not categorical. To convert them into hard training labels, I:

1. Extracted the three Question 1 columns (`Class1.1`, `Class1.2`, `Class1.3`), which correspond to **smooth/elliptical**, **featured/spiral**, and **star or artifact**
2. Assigned the class with the highest volunteer agreement via `idxmax`
3. Applied a **confidence threshold of ≥ 0.65** — samples where no class exceeded this threshold were excluded as ambiguous

This threshold filtering is a deliberate design choice: training on ambiguous samples would inject noise that degrades model generalization.

---

## Architecture

```
Raw JPG Images (424×424)
        │
        ▼
  Data Augmentation
  (RandomHorizontalFlip, RandomRotation, ColorJitter)
        │
        ▼
  Normalization → ImageNet stats
        │
        ▼
  Custom CNN
  [Conv → BatchNorm → ReLU → MaxPool] × N layers
        │
        ▼
  Fully Connected Head → 3 outputs
        │
        ▼
  Softmax → Class Probabilities
  (Elliptical | Spiral | Artifact)
```

### Why a Custom CNN?

This project builds the convolutional network from scratch rather than using a pretrained backbone. The goal is to understand what the network actually learns — how early layers detect low-level features like edges and curves, and how deeper layers compose these into higher-level structures like spiral arms or elliptical envelopes. Building it from scratch also gives full control over the architecture decisions: depth, kernel sizes, pooling strategy, and regularization.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Deep Learning Framework | PyTorch 2.x |
| GPU Acceleration | CUDA 12.6 |
| Data Processing | pandas, NumPy |
| Image Handling | PIL, torchvision |
| Visualization | Matplotlib |
| Environment | venv (Python 3.11) |
| Hardware | NVIDIA GeForce RTX 3090 |

---

## Project Structure

```
GalaxyClassifier/
│
├── data/
│   ├── images/
│   │   └── train/
│   │       └── images_training_rev1/   # 61,578 galaxy JPGs
│   └── training_solutions_rev1.csv     # Soft probabilistic labels
│
├── notebooks/
│   └── eda.ipynb                       # Exploratory Data Analysis
│
├── src/
│   ├── dataset.py                      # PyTorch Dataset class
│   ├── model.py                        # CNN architecture
│   ├── train.py                        # Training loop
│   └── evaluate.py                     # Evaluation & metrics
│
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/galaxy-morphology-classifier.git
cd galaxy-morphology-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv311
# Windows
venv311\Scripts\activate
# macOS/Linux
source venv311/bin/activate
```

### 3. Install dependencies

```bash
# PyTorch with CUDA 12.6 (GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Other dependencies
pip install -r requirements.txt
```

### 4. Download the dataset

You'll need a [Kaggle API token](https://www.kaggle.com/docs/api). Once configured:

```bash
kaggle competitions download -c galaxy-zoo-the-galaxy-challenge
```

---

## Roadmap

- [x] Environment setup (PyTorch + CUDA)
- [x] Dataset download & label engineering
- [ ] Exploratory Data Analysis (EDA)
- [ ] Custom CNN architecture
- [ ] Model training & evaluation
- [ ] Confusion matrix & per-class metrics
- [ ] Advanced: probabilistic label modeling (KL divergence loss)

---

## Background & Motivation

This project sits at the intersection of two interests: **machine learning** and **astronomy**. Galaxy morphology classification is a genuinely hard problem at scale — the SDSS has imaged hundreds of millions of galaxies, far more than humans can manually classify. Projects like Galaxy Zoo crowd-sourced this problem; deep learning is the next step.

Beyond the science, this project demonstrates a complete ML pipeline: from raw, messy probabilistic labels through feature engineering, model design, and GPU-accelerated training.

---

## Future Work

- **Probabilistic label modeling** — instead of hard labels, model the full volunteer distribution using grouped softmax outputs and KL divergence loss. This treats label uncertainty as signal, not noise.
- **Dataset expansion** — experiment with [Galaxy10 DECals](https://astronn.readthedocs.io/en/latest/galaxy10.html), a cleaner 10-class dataset useful for rapid iteration.
- **Transfer learning** — explore fine-tuning a pretrained ResNet backbone as a performance comparison against the custom CNN.
- **Astronomy RAG application** — a retrieval-augmented generation system over astrophysics literature as a follow-up project.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Galaxy Zoo 2](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/) and the Zooniverse citizen science community
- [Kaggle Galaxy Zoo Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge)
- Sloan Digital Sky Survey (SDSS)
