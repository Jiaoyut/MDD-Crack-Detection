# Enhancing Crack Detection via Memory-Aware Dynamic Knowledge Distillation

[![DOI](https://zenodo.org/badge/1123080674.svg)](https://doi.org/10.5281/zenodo.18059674)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

> **Note:** This repository contains the official implementation of the paper **"Enhancing Crack Detection via Memory-Aware Dynamic Knowledge Distillation"**, submitted to *Multimedia Systems*.

## 📖 Introduction

**Memory-Aware Dynamic Distillation (MDD)** is a novel framework tailored for crack detection on resource-constrained devices. It addresses the **catastrophic forgetting** problem in distillation by introducing a "Review Mechanism" inspired by human memory curves.

**Key Performance:**
On the Crack Segmentation dataset, MDD achieves **79.1% mAP50** using a lightweight **YOLO11n-seg** student, outperforming standard KD by **5%** with only **4.9%** of the teacher's parameters.

## 📂 Directory Structure

The project is organized as follows:

```text
MDD-Crack-Detection/
├── datasets/               # Dataset root directory
│   └── crack-seg/          # The specific Crack Segmentation Dataset
│       ├── train/          # Training set (images/ & labels/)
│       ├── valid/          # Validation set
│       └── test/           # Test set
├── models/                 # MDD modules
├── predictOut/             # Output directory for inference results (visualizations)
├── runs/                   # Directory storing training logs and checkpoints
│   ├── segment/            # Segmentation training logs
│   └── simple_distill/     # Distillation experiment logs
├── testdata/               # Sample images used for quick testing/inference
├── utils/                  # Utility scripts and helper functions
├── weights/                # Pre-trained model weights
   └── MDD/
      └── weights/        # Contains the best trained MDD model
```

## Training (Reproduce MDD)
To train the student model using the MDD framework:

```Bash
python teacher2student.py
```

##  Citation
If you find this work helpful for your research, please consider citing our paper:

```text
@article{MDD_Crack_2025,
  title={Enhancing Crack Detection via Memory-Aware Dynamic Knowledge Distillation},
  author={Zhao, Liang and Jiao, Yutong and Chen, Dengfeng and Liu, Shipeng},
  journal={Multimedia Systems},
  year={2025},
  note={Under Review}
}
```
