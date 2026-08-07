"""
ORION: Optimized Radiological Intelligence for Orthopaedic Analysis

This package contains the core library for the RSNA 2026 Knee Abnormality Detection
Kaggle Competition. It is designed as a modular research system.

Core Modules:
- `data`: DICOM preprocessing, datasets, transforms, splits.
- `models`: Vision backbones, text encoders, necks, fusion, heads, losses.
- `training`: Trainer, evaluator, optimizers, callbacks.
- `inference`: Predictor, TTA, ensembling.
- `evaluation`: Metrics, calibration, error analysis.
- `explainability`: Grad-CAM, attention visualization.
- `weak_supervision`: NLP label extraction, label aggregation.
"""

__version__ = "0.1.0"
__author__ = "ORION Research Team"
