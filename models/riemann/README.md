# 📐 Riemannian Geometry Models

This folder (`models/riemann`) contains the core research implementations based on **Riemannian Geometry** for Symmetric Positive Definite (SPD) matrices. These methods are essential for robust EEG classification.

## 📚 Mathematical Concepts

### 1. Minimum Distance to Mean (MDM)
**File**: [`mdm.py`](mdm.py)
A classifier that assigns a sample covariance matrix to the class whose mean covariance is closest in terms of Riemannian distance.

### 2. Tangent Space Logistic Regression (TSLR)
**File**: [`tslr.py`](tslr.py)
Projects SPD matrices into the tangent space at the geometric mean. The tangent vectors are then classified using standard Logistic Regression. This allows the use of Euclidean linear classifiers on curved manifolds.

### 3. Covariance CNN (CovCNN)
**File**: [`covcnn.py`](covcnn.py)
A Deep Learning approach that treats the covariance matrix as an image. It applies 2D convolutions directly to the SPD matrix structure.
*   **Input**: *C* (size *C x C*)
*   **Layers**: Conv2D → ReLU → AdaptiveAvgPool → Linear.

### 4. RiFU (Riemannian Fusion)
**File**: [`rifu.py`](rifu.py)
A **Riemannian U-Net** for manifold alignment.
*   **Architecture**: Encoder-Decoder structure using **SPDLinear** layers (congruence transforms $W^T C W$).
*   **Function**: Aligns subject-specific covariance manifolds to a common reference space.
*   **Losses**: Trained with a combination of Fisher loss (class separation), Subject Fisher loss (subject alignment), and Riemannian reconstruction loss.

### 5. DCR (Domain Covariance Re-Alignment)
**File**: [`dcrbifa.py`](dcrbifa.py)
A **Dual-Fisher** optimization module (`DCRPreAlignerDualFast`).
*   **Objective**: Iteratively learns a rotation matrix $R$ in the Log-Euclidean space.
*   **Optimization**:
    *   **Maximize Class Fisher**: Increases separation between different motor imagery classes.
    *   **Minimize Subject Fisher**: Decreases distance between different subjects' distributions.
*   **Result**: A subject-invariant representation that boosts classifier performance in cross-subject settings.

## File List

| File | Description |
| :--- | :--- |
| `mdm.py` | Wrapper for PyRiemann's MDM classifier. |
| `tslr.py` | Tangent Space Logistic Regression pipeline. |
| `covcnn.py` | CNN architecture operating on Covariance matrices. |
| `dcra.py` | Domain Covariance Re-Alignment for domain adaptation. |
| `urpa.py` | Unsupervised Riemannian Procrustes Analysis. |
| `gcn.py` | (Note: Main GCN is in parent folder, this might be a variant). |
| `riemann_gpu.py` | CUDA-accelerated Riemannian operations (Log-Euclidean, Geodesic distance). |
