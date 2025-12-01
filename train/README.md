# 🚂 Training Scripts

This directory contains the training pipelines for GADCN and various baseline models.

## Key Scripts

| File | Description |
| :--- | :--- |
| **[`riemann_train.py`](riemann_train.py)** | **Main Training Loop**. Trains the GADCN model. Handles hyperparameter parsing, model initialization, training loops, and validation. |
| **[`model_train.py`](model_train.py)** | **Generic Trainer**. A flexible training script for standard deep learning models (CNNs, LSTMs) without specific Riemannian components. |
| **[`spdnet.py`](spdnet.py)** | **SPDNet Training**. Implementation and training loop for SPDNet, a deep learning architecture specifically designed for SPD matrices. |
| **[`rifu_train.py`](rifu_train.py)** | **Riemannian Fusion**. Training script for models that fuse Riemannian features with other modalities. |
| **[`train_bci_cs.py`](train_bci_cs.py)** | **Cross-Subject Training**. Specialized script for Cross-Subject (Leave-One-Subject-Out) training scenarios on BCI datasets. |
| **[`train_physio_cs.py`](train_physio_cs.py)** | **Physionet Training**. Training pipeline optimized for the large-scale Physionet dataset. |

## Usage

To train GADCN on BCI IV 2a:
```bash
python train/riemann_train.py --dataset bci42a --epochs 100 --batch_size 32 --lr 0.001
```
