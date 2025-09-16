# Final Plan: Multi-Domain EEG → Spatio-Temporal GNN (with MAE pretraining)

## 0) Data & Preprocessing (non-negotiable)
1. **Filtering & referencing:** Band-pass 1–40 Hz (or task-specific), notch at 50 Hz, common average/bipolar reference.  
2. **Artifact control:** Amplitude/z-score rejection ± ICA/ASR; bad-channel interpolation if needed.  
3. **Epoching:** Fixed windows (e.g., 2 s, 50% overlap). Label each epoch.  
4. **Per-subject normalization:** Z-score each feature/channel using a calibration span.  
5. **Covariance shrinkage:** For each epoch \(t\), compute SPD covariance \(C_t\) with Ledoit–Wolf (or \(\hat C=(1-\lambda)C+\lambda\,\tfrac{\mathrm{tr}(C)}{M}I\)); add \(\epsilon I\) before matrix functions.

---

## 1) Node Features (per channel \(i\) at epoch \(t\))
- **Temporal stats:** mean, variance, skew, kurtosis, RMS, line length, Hjorth (activity/mobility/complexity).  
- **Spectral powers:** Welch PSD → band powers \(\delta,\theta,\alpha,\beta,\gamma\); relative powers; key band ratios.  
- **Spectral shape/entropy:** spectral slope/intercept (1/f), **spectral entropy** \(H_s=-\sum_k p_k\log p_k/\log K\) with \(p_k=P_k/\sum P_k\). *(No Gaussianity required for \(H_s\).)*  
- **(Optional) time–freq envelopes, sample entropy** if stable on your window.

Concatenate to vector \(z_{i,t}\). This is your **independent feature set** by design.

---

## 2) Self-Supervised Pretraining (final choice: **Masked Autoencoder**) 
- **Input:** \(z_{i,t}\). Mask a random subset of dimensions (e.g., 20–30%), optionally add small noise.
- **Encoder \(E\)** → embedding \(x_{i,t}\in\mathbb{R}^{d_e}\).  
- **Decoder \(D\)** reconstructs **only masked** dims.  
- **Loss (masked MSE):** \(\mathcal{L}_{\text{MAE}}=\|M\odot (D(E(\tilde z))-z)\|_2^2\) (Huber is a robust alternative).  
- **Use after pretrain:** **Discard the decoder**; freeze or lightly fine-tune **encoder \(E\)** to produce node embeddings \(x_{i,t}=E(z_{i,t})\).

*Why MAE fits your case:* your features are heterogeneous and nominally independent; masking forces the encoder to learn **latent dependencies** across them—just what you want.

---

## 3) Spatial Adjacency (within epoch \(t\)): multi-view + learned fusion
You’ll build three channel–channel matrices, normalize/sparsify each, then fuse with learned softmax weights.

**(a) Log-Euclidean (single covariance map):**  
- Compute **one** SPD covariance per epoch: \(C_t\).  
- Map to Euclidean tangent: \(S_t=\log(C_t)\) (eigendecompose \(C_t=U\Lambda U^\top\); \(S_t=U\log\Lambda\,U^\top\)).  
- Form an adjacency proxy \(A^{LE}_t\) from \(S_t\), e.g., take \(|S_{t,ij}|\), keep top-\(k\) per node, symmetrize, then degree-normalize.

**(b) Mutual Information (power-based):**  
- Build channel vectors \(p_{i,t}\) (bandpower profile or power envelopes).  
- Estimate MI (e.g., Kraskov kNN). Normalize to \([0,1]\) → \(A^{MI}_t\).

**(c) Spearman rank correlation:**  
- Between amplitude envelopes or band-limited signals → \(|\rho_s|\) → \(A^{SP}_t\).

**Normalize & sparsify each:** symmetric normalization \(\tilde A = D^{-1/2}(A+\epsilon I)D^{-1/2}\); top-\(k\) neighbors per node.

**Learned fusion (your choice):**  
\[
\alpha=\mathrm{softmax}(w),\qquad A_t=\sum_{m\in\{LE,MI,SP\}} \alpha_m\, \tilde A^{(m)}_t
\]
*(Optional)* put a gentle prior that favors the Riemann view if you want cross-subject robustness.

---

## 4) Temporal “Memory” Coupling (AIRM-weighted)
- Compute **AIRM distance** between consecutive epoch covariances:
\[
d_{t,t+1}=\left\|\log\!\big(C_t^{-1/2}\,C_{t+1}\,C_t^{-1/2}\big)\right\|_F
\]
- Convert to kernel weight (your idea):  
\[
s_{t\to t+1}=\exp(-\beta\, d_{t,t+1})
\]
Use \(s_{t\to t+1}\) to modulate how much information flows from \(t\) to \(t+1\).

**Two implementations (both acceptable):**
1) **GRU/LSTM gate bias (your original):** inject \(-\gamma d_{t,t+1}\) into the forget/update gate bias so larger distance ⇒ more forgetting.  
2) **ST temporal conv (ablation):** in a causal 1D temporal conv, multiply the lag-1 contribution (and/or residual) by \(s_{t\to t+1}\). This achieves the same AIRM-weighted memory without RNNs.

---

## 5) Backbone Models (you’ll train at least one; others for ablation)

**Spatial layer (per epoch t):**
- Start with **GCN**:
\[
H_t^{sp} = \sigma\Big(D_t^{-1/2}(A_t+I)D_t^{-1/2} X_t W_s\Big).
\]

- **Fused attention within each relation (edge gating):** for each relation \(r \in \{R,E,S\}\),
\[
\tilde{A}_{ij,t}^{(r)} = A_{ij,t}^{(r)} \cdot \sigma\Big(a_r^\top [h_i \Vert h_j]\Big),
\]
where \(h_i,h_j\) are the current node embeddings; single-head, low-dim gate vector \(a_r\).

- **Relational fusion with Riemann prior (cross-attention over edge types, Ablation):**
Compare models **with and without Relational GNN**. In the Relational variant, treat Riemann as structural prior supplying the **query**, and entropy/spearman as **keys/values**:
\[
\alpha_{ij}^{(r)} = \operatorname{softmax}_r \left( \tfrac{q_{ij}^\top k_{ij}^{(r)}}{\sqrt{d}} \right), \qquad
A^{fused}_{ij,t} = \sum_{r \in \{R,E,S\}} \alpha_{ij}^{(r)} \tilde{A}_{ij,t}^{(r)}.
\]
This anchors geometry in Riemann edges while allowing information-theoretic (entropy) and synchrony (spearman) edges to modulate message flow.

**Temporal layer (choose primary; others = ablation):**
- **Primary:** GRU over node trajectories \(\{H_t^{sp}\}\), with **AIRM kernel** biasing the update gate.  
- **Ablation 1:** LSTM with AIRM-biased forget gate.  
- **Ablation 2:** ST-Temporal Conv (TCN, causal/dilated), with AIRM kernel scaling the lag-1/residual path.

**Head:** per-time (per-epoch) classifier (MLP) with label smoothing; optionally class-balanced loss.

---

## 6) Training Schedule
1. **Pretrain MAE** on all unlabeled epochs (pool subjects).  
2. **Freeze or lightly fine-tune** encoder \(E\); build \(A^{LE}, A^{MI}, A^{SP}\) per epoch; learn fusion \(\alpha\).  
3. **Train GCN(+shallow attention) + temporal layer** (GRU primary).  
4. **Evaluate**; then run ablations (LSTM vs GRU vs ST-TCN; with/without shallow attention; with/without Riemann-favoring prior).

---

## 7) Losses & Regularization
- **Primary:** Cross-entropy (per epoch \(t\)).  
- **Edge sparsity (optional):** \(\lambda\sum_t\|A_t\|_1\).  
- **DropEdge/NodeDropout**, weight decay \(1\text{e-}4\), early stopping.  
- Ensure **no temporal leakage** (strictly causal splits; don’t overlap train/test windows from the same sequence).

---

## 8) Practical Defaults (so you can start now)
- Epoch length: 2 s, 50% overlap; Fs 128–256 Hz.  
- Covariance: Ledoit–Wolf; \(\epsilon=1\text{e-}6\).  
- MAE: mask 30% dims; encoder MLP (d→256→128), decoder (128→256→d); **use encoder output (128-d)**.  
- Graph: top-\(k=8\) neighbors per node per view; softmax fusion learnable.  
- Spatial: 2 GCN layers (128→128), GELU, LayerNorm, shallow edge gate.  
- Temporal (**primary**): GRU hidden 128; **AIRM bias** coefficient \(\gamma\in[0.5,2]\).  
- ST-TCN (**ablation**): kernel 3, dilations [1,2,4], causal; scale lag-1 by \(s_{t\to t+1}\).  
- Optim: AdamW, lr 1e-3 (backbone), 3e-4 (fusion/attention), cosine decay.  
- Metrics: accuracy, balanced acc, macro-F1, AUROC; within- and cross-subject.

---

## 9) What to Report (for your thesis/paper)
- **Baselines:** CSP+LDA, Tangent-space LR (pyRiemann), EEGNet.  
- **Ablations:** MAE vs no-pretrain; GCN vs GCN+shallow attention; GRU vs LSTM vs ST-TCN; with/without AIRM kernel; single-view vs fused \(A\); with/without Riemann-favoring prior.  
- **Robustness:** calibration length, top-\(k\) sensitivity, effect of shrinkage, cross-subject transfer.

---

### TL;DR architecture (one pass)
1) \(z_{i,t}\) → **MAE encoder \(E\)** → \(x_{i,t}\).  
2) \(C_t\) → \(S_t=\log C_t\) → \(A^{LE}_t\); plus \(A^{MI}_t\), \(A^{SP}_t\) → **fuse with softmax** → \(A_t\).  
3) **GCN (+ shallow edge gate)** on \(A_t\): \(X_t \to H^{sp}_t\).  
4) **Temporal (GRU)** with **AIRM-biased gate** using \(s_{t\to t+1}=\exp(-\beta d_{t,t+1})\).  
5) **Classifier head** → per-epoch prediction.

