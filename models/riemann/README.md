# 📐 Riemannian Geometry Models

This folder (`models/riemann`) contains the core research implementations based on **Riemannian Geometry** for Symmetric Positive Definite (SPD) matrices. These methods are essential for robust EEG classification.

## 📚 Mathematical Concepts

### 1. Minimum Distance to Mean (MDM)
**File**: [`mdm.py`](mdm.py)
A classifier that assigns a sample covariance matrix *C* to the class *k* whose mean covariance *C̄_k* is closest in terms of Riemannian distance:

> k_hat = argmin_k δ_R(C, C̄_k)

where *δ_R* is the affine-invariant Riemannian metric.

### 2. Tangent Space Logistic Regression (TSLR)
**File**: [`tslr.py`](tslr.py)
Projects SPD matrices into the tangent space at the geometric mean *C_ref*. The tangent vectors are then classified using standard Logistic Regression.

> S_i = Log_Cref(C_i)

This allows the use of Euclidean linear classifiers on curved manifolds.

### 3. Covariance CNN (CovCNN)
**File**: [`covcnn.py`](covcnn.py)
A Deep Learning approach that treats the covariance matrix as an image. It applies 2D convolutions directly to the SPD matrix structure.
*   **Input**: *C* (size *C x C*)
*   **Layers**: Conv2D → ReLU → AdaptiveAvgPool → Linear.

### 4. Domain Covariance Re-Alignment (DCR)
**File**: [`dcra.py`](dcra.py)
A transfer learning technique that aligns the covariance distributions of different subjects (domains) to a common reference, reducing inter-subject variability.

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
