# 🚂 Training Scripts

This directory contains the training pipelines for GADCN and various baseline models.

## Key Scripts

| File | Description |
| :--- | :--- |
| **[`riemann_train.py`](riemann_train.py)** | **Main Training Loop**. Trains the GADCN model. Handles hyperparameter parsing, model initialization, training loops, and validation. |
| **[`model_train.py`](model_train.py)** | **Generic Trainer**. A flexible training script for standard deep learning models (CNNs, LSTMs) without specific Riemannian components. |
| **[`spdnet.py`](spdnet.py)** | **SPD-DCNet (Deep Congruence Network)**. Implements a deep network for SPD matrices using `SPDLinear` layers and Log-Euclidean mapping. Includes a **Subject Alignment Loss** to minimize inter-subject variability. |
| **[`rifuce_train.py`](rifuce_train.py)** | **RiFUNet Classifier**. Training script for the end-to-end **RiFUNet + DANN** model. Combines manifold alignment (RiFuNet), discriminative learning (Fisher Loss), and domain adaptation (GRL). |
| **[`train_bci_cs.py`](train_bci_cs.py)** | **Cross-Subject Training**. Specialized script for Cross-Subject (Leave-One-Subject-Out) training scenarios on BCI datasets. |
| **[`train_physio_cs.py`](train_physio_cs.py)** | **Physionet Training**. Training pipeline optimized for the large-scale Physionet dataset. |

## Usage

To train GADCN on BCI IV 2a:
```bash
python train/riemann_train.py --dataset bci42a --epochs 100 --batch_size 32 --lr 0.001
```
