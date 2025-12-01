# GADCN: Graph Attention Dual Convolution Networks for EEG Analysis

This repository contains the official implementation of the research paper **"GADCN: Graph Attention Dual Convolution Networks"** (ICML 2026). It provides a comprehensive framework for EEG signal analysis, leveraging **Riemannian Geometry** and **Geometric Deep Learning**.

## 📄 Research Paper

**Title**: GADCN: Graph Attention Dual Convolution Networks
**File**: [references/arxiv_gadcn_icml_2026.pdf](references/arxiv_gadcn_icml_2026.pdf)

The research focuses on decoding motor imagery and other EEG paradigms by treating EEG signals as data on a Riemannian manifold of Symmetric Positive Definite (SPD) matrices.

---

## 🏗️ System Architecture

### Preprocessing Pipeline
![Preprocessing](references/Preprocessing_modules.png)

### Classifier Architecture
![Classifier](references/Classifier_modules.png)

---

## 🧮 Mathematical Foundations

This repository implements advanced signal processing and machine learning techniques rooted in differential geometry.

### 1. Riemannian Geometry of EEG
EEG signals are often represented by their covariance matrices. A covariance matrix **C** (size *C x C*) is **Symmetric Positive Definite (SPD)**. The space of SPD matrices forms a Riemannian manifold, not a Euclidean vector space.

The **Riemannian distance** between two SPD matrices is calculated using the affine-invariant metric, which relies on the eigenvalues of the product of one matrix and the inverse of the other.

### 2. Tangent Space Mapping
To apply standard Euclidean machine learning algorithms (like SVMs or Logistic Regression), we project the SPD matrices onto the **Tangent Space** at a reference point (usually the geometric mean of the dataset).

The mapping is defined by the **Log-Euclidean** map, which transforms the curved manifold structure into a flat Euclidean vector space that can be vectorized.

### 3. Core Research Components
This repository implements the novel contributions of the paper, focusing on geometry-aware manifold learning:

#### **1. RiFU (Riemannian Fusion Pre-Aligner)**
**File**: [`models/riemann/rifu.py`](models/riemann/rifu.py)
A **Riemannian U-Net** that performs non-linear alignment of covariance matrices in the SPD space. It uses a congruence-based encoder-decoder architecture to align subject-specific manifolds to a common reference space, enhancing cross-subject generalization.

#### **2. DCR (Domain Covariance Re-Alignment)**
**File**: [`models/riemann/dcrbifa.py`](models/riemann/dcrbifa.py)
A **Dual-Fisher** optimization module that iteratively rotates covariance matrices in the log-Euclidean space. It simultaneously:
*   Maximizes **Class Fisher** information (separation between classes).
*   Minimizes **Subject Fisher** information (alignment between subjects).

#### **3. SPD-DCNet (Deep Congruence Network)**
**File**: [`train/spdnet.py`](train/spdnet.py)
A deep neural network architecture designed specifically for SPD matrices. It consists of **SPDLinear layers** (congruence transforms) followed by a Log-Euclidean mapping and a standard classifier, trained with a subject-alignment loss.

#### **4. RiFUNet Classifier**
**File**: [`train/rifuce_train.py`](train/rifuce_train.py)
An end-to-end classification framework that integrates:
*   **RiFuNet Backbone**: For manifold alignment.
*   **Fisher Loss**: To enforce discriminative feature learning.
*   **DANN (Domain Adversarial Neural Network)**: To learn subject-invariant features via a Gradient Reversal Layer (GRL).

---

## 📂 Repository Structure

The repository is organized into modular components. Click on the folder names for detailed documentation.

| Folder | Description |
| :--- | :--- |
| **[`models/`](models/README.md)** | **Core Research Code**. Contains GADCN, Riemannian models, and deep learning architectures. |
| **[`models/riemann/`](models/riemann/README.md)** | **Riemannian Geometry**. Implementations of MDM, Tangent Space, and SPD-specific layers. |
| **[`preprocessing/`](preprocessing/README.md)** | **Data Cleaning**. Artifact removal, filtering, and signal normalization. |
| **[`feature_extraction/`](feature_extraction/README.md)** | **Feature Engineering**. Covariance estimation, spectral power, and temporal features. |
| **[`data_construction/`](data_construction/README.md)** | **Dataset Loaders**. Scripts to format BCI IV 2a, Physionet, and other datasets. |
| **[`train/`](train/README.md)** | **Training Scripts**. Pipelines for training GADCN and baseline models. |
| **[`eval/`](eval/README.md)** | **Evaluation**. Metrics, cross-validation, and performance analysis. |
| **[`results/`](results/README.md)** | **Figures & Reports**. Generated confusion matrices, training logs, and visualizations. |
| **[`references/`](references/)** | **Papers & Diagrams**. The original paper and architecture diagrams. |

---

## 🚀 Getting Started

### Prerequisites
*   **Python 3.8+**
*   **PyTorch 2.0+** (with CUDA 12.1 for GPU acceleration)
*   **MNE-Python** (for EEG handling)
*   **PyRiemann** (for Riemannian geometry)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/SanjeevM2004/EEG_research.git
    cd EEG_research
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Usage Example

### Usage Example

To train the **RiFUNet** classifier on the BCI IV 2a dataset:

```bash
python train/rifuce_train.py
```

---

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{gadcn2026,
      title={Geometry-Aware Deep Congruence Networks for Manifold Learning in Cross-Subject Motor Imagery}, 
      author={Sanjeev Manivannan, Chandrashekar Lakshminarayan},
      year={2026},
      eprint={2511.18940},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.18940}, 
}
```
