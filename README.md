# 🦴 ORION — RSNA 2026 Knee Abnormality Detection

> **O**ptimized **R**adiological **I**ntelligence for **O**rthopaedic A**N**alysis  
> A research-grade, production-quality system targeting **Top-10** in the RSNA 2026 Knee Abnormality Detection Kaggle Competition.

---

## 🎯 Competition Overview

| Property | Detail |
|---|---|
| **Task** | Multi-label classification: 12 knee abnormalities per MRI study |
| **Input** | DICOM MRI (sagittal/coronal/axial) + radiology reports (9 languages) |
| **Metric** | Macro-averaged AUC ROC (12 targets) |
| **Dataset** | 5,000+ studies, 16 institutions, ~58 expert-labeled |
| **Deadline** | October 22, 2026 |

### Target Labels
`ACL | MCL | Medial Meniscus | Lateral Meniscus | Medial OA | Lateral OA | PF OA | Effusion | Synovitis | Baker's Cyst | Contusion | Fracture`

---

## 🏗️ Architecture Overview

```
DICOM Images ──┐
               ├──► Preprocessing ──► Feature Extraction ──► Fusion ──► Multi-label Head ──► AUC
Radiology Txt ─┘       Pipeline       (Vision + NLP)        Layer        (12 classes)
```

Key design choices:
- **Multi-Instance Learning (MIL)** for slice aggregation
- **Cross-modal attention fusion** for image + report integration
- **Asymmetric Loss** to handle severe label imbalance
- **Programmatic weak supervision** (Snorkel) to expand labeled data from 58 → 5000+
- **Ensemble** of Swin V2, ConvNeXt, BiomedCLIP backbones

---

## 🚀 Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/your-org/ORION.git
cd ORION

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install in editable mode with dev extras
pip install -e ".[dev]"
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env to set your paths and API keys
```

### 3. Preprocess Data
```bash
python scripts/preprocess.py --config configs/data/preprocessing.yaml --data-dir /path/to/dicom
```

### 4. Extract Weak Labels
```bash
python scripts/extract_weak_labels.py --reports-csv data/train_reports.csv --output data/weak_labels.csv
```

### 5. Train Baseline
```bash
python scripts/train.py --config configs/experiment/baseline_resnet50.yaml
```

### 6. Full Cross-Validation
```bash
python scripts/cross_validate.py --config configs/experiment/swin_v2_base.yaml
```

### 7. Generate Submission
```bash
python scripts/ensemble.py --model-dirs runs/fold0 runs/fold1 runs/fold2 runs/fold3 runs/fold4
python scripts/submit.py --ensemble-dir runs/ensemble --output submission.csv
```

---

## 📁 Project Structure

```
ORION/
├── configs/          # YAML configuration files
├── src/orion/        # Core library (importable)
│   ├── data/         # DICOM processing, datasets, transforms
│   ├── models/       # Backbones, fusion, heads, losses
│   ├── training/     # Trainer, callbacks, schedulers
│   ├── inference/    # Prediction, TTA, ensembling
│   ├── evaluation/   # Metrics, calibration, error analysis
│   ├── explainability/  # Grad-CAM, attention maps
│   ├── pretraining/  # SSL: MAE, CLIP, contrastive
│   ├── weak_supervision/ # Snorkel, cleanlab, FixMatch
│   └── utils/        # Config, logging, I/O helpers
├── scripts/          # Training, evaluation, submission entry points
├── notebooks/        # Research and EDA notebooks
├── tests/            # Unit, integration, smoke tests
├── docs/             # Architecture docs, experiment log
├── research/         # Ablations, visualizations, ideas
└── deployment/       # FastAPI server, ONNX export, quantization
```

---

## 📊 Experiment Tracking

All experiments are logged to Weights & Biases:
```bash
wandb login
export WANDB_PROJECT="rsna-knee-2026"
```

---

## 🧪 Running Tests
```bash
# All tests
pytest

# Only fast unit tests
pytest tests/unit/ -m "not slow"

# Smoke tests (forward pass checks)
pytest tests/smoke/ -v

# With coverage
pytest --cov=src/orion --cov-report=html
```

---

## 📖 Learning Path

This codebase is structured as a **research curriculum**. Study modules in this order:

1. `docs/papers/mrnet.md` — Historical baseline
2. `src/orion/data/dicom/` — Understanding DICOM and MRI data
3. `src/orion/data/mri/physics.py` — MRI physics fundamentals
4. `src/orion/models/backbones/resnet.py` — CNN from scratch
5. `src/orion/models/backbones/vit.py` — Vision Transformer from scratch
6. `src/orion/models/necks/slice_aggregator.py` — MIL theory
7. `src/orion/models/losses/asymmetric.py` — Loss functions for imbalance
8. `src/orion/weak_supervision/label_model.py` — Snorkel label aggregation
9. `src/orion/models/fusion/cross_attention.py` — Multimodal fusion
10. `src/orion/explainability/gradcam.py` — Interpretability

---

## 📜 Key Papers

| Paper | Use in ORION |
|---|---|
| [MRNet (Bien et al., 2018)](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699) | Baseline architecture reference |
| [CoPAS (2024)](https://arxiv.org/abs/...) | Multi-view attention for knee MRI |
| [BiomedCLIP (2023)](https://arxiv.org/abs/2303.00915) | Pretrained vision-language backbone |
| [Focal Loss (Lin et al., 2017)](https://arxiv.org/abs/1708.02002) | Imbalanced training |
| [Asymmetric Loss (Ben-Baruch et al., 2020)](https://arxiv.org/abs/2009.14119) | Multi-label imbalance |
| [Attention MIL (Ilse et al., 2018)](https://arxiv.org/abs/1802.04712) | Slice aggregation |
| [Snorkel (Ratner et al., 2017)](https://arxiv.org/abs/1711.10160) | Weak label aggregation |
| [Swin Transformer V2 (Liu et al., 2022)](https://arxiv.org/abs/2111.09883) | Primary backbone |

---

## 🏆 Target Performance

| Phase | Strategy | Expected Macro AUC |
|---|---|---|
| Baseline | ResNet-50 + expert labels only | ~0.72 |
| Weak Labels | + Snorkel report extraction | ~0.80 |
| Strong Backbone | Swin V2 + weak labels | ~0.85 |
| Multimodal | + Cross-modal fusion | ~0.88 |
| Full Ensemble | 5-fold × 3 architectures + TTA | ~0.90+ |

---

## 📝 License
MIT License — see [LICENSE](LICENSE)
