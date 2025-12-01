"""
train_dcr_tsa_lda_end2end_loso.py
------------------------------------------------------------
Run end-to-end DCR + TSA + LDA (differentiable)
for cross-subject (LOSO) zero-shot evaluation.

Requires:
    models/
        riemann/
            dcr_tsa_lda_end2end.py  ← contains class DCR_TSA_LDA
    EEG_data/bci_active4.pt:
        data["ra_covs"], data["labels"], data["subj"]
------------------------------------------------------------
"""

import numpy as np
import torch
from time import time
from sklearn.model_selection import LeaveOneGroupOut

# ============================================================
# 1) CONFIG
# ============================================================
CACHE_PATH = "./EEG_data/bci_active4.pt"
DATASET = "BCIIV2a"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 72)
print(" End-to-End DCR + TSA + LDA (Differentiable)")
print(f" Dataset: {DATASET}")
print(" Evaluation: Leave-One-Subject-Out (LOSO, zero-shot)")
print("=" * 72)

# ============================================================
# 2) LOAD DATA
# ============================================================
print("📦 Loading data...")
data = torch.load(CACHE_PATH, map_location="cpu")

covs = data["ra_covs"]        # list/tensor of RA covariances per trial
labels = np.array(data["labels"])
subjects = np.array(data["subj"])

# Convert to numpy arrays
covs = [c.numpy() if torch.is_tensor(c) else np.array(c) for c in covs]
X_cov = np.stack(covs)                 # (N, d, d)
y = labels.astype(int)
groups = np.asarray(subjects)

# Sanity check & trim if needed
n = min(len(X_cov), len(y), len(groups))
if (len(X_cov), len(y), len(groups)) != (n, n, n):
    print(f"⚠️ Length mismatch detected → trimming to {n}")
    X_cov, y, groups = X_cov[:n], y[:n], groups[:n]

assert len(X_cov) == len(y) == len(groups)
print(f"Loaded {len(X_cov)} RA covariances from {len(np.unique(groups))} subjects.\n")

# Per-subject counts (optional)
uniq, cnts = np.unique(groups, return_counts=True)
for sid, c in zip(uniq, cnts):
    print(f"  {sid:>6} : {c} trials")
print()

# Model dims
d = X_cov.shape[-1]
n_classes = len(np.unique(y))
print(f"[info] d = {d} channels, classes = {n_classes}\n")

# ============================================================
# 3) IMPORT MODEL
# ============================================================
from models.riemann.dcr_tslr import DCR_TSLR

# Default hyperparams (good starting point for BCIIV-2a)
model_kwargs = dict(
    d=d,
    n_classes=n_classes,
    steps=400,
    lr=1e-3,
    weight_decay=0.0,
    fisher_weight=1e-2,
    r_identity_reg=1e-4,
    #lda_shrinkage=1e-2,
    learn_lambda=True,
    device=DEVICE,
    verbose=True,
)

print("Model: DCR_TSLR")
print(f"Params: {model_kwargs}\n")

# ============================================================
# 4) LOSO EVALUATION
# ============================================================
logo = LeaveOneGroupOut()
accs, per_subject = [], {}
t0 = time()

print("=" * 72)
print("🧠 Running LOSO Evaluation (Zero-Shot Cross-Subject)")
print("=" * 72)

for train_idx, test_idx in logo.split(X_cov, y, groups):
    test_subj = np.unique(groups[test_idx])[0]

    X_train, X_test = X_cov[train_idx], X_cov[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # (Re)create fresh model per fold
    clf = DCR_TSLR(**model_kwargs)

    try:
        clf.fit(X_train, y_train)
        # inference uses cached LDA params (mu_, Sigma_inv_, log_pi_)
        scores = clf._forward_scores(X_test)
        y_pred = torch.argmax(scores, dim=1).cpu().numpy()
        acc = np.mean(y_pred == y_test)
    except Exception as e:
        print(f"  ❌ ERROR DCR_TSA_LDA on {test_subj}: {type(e).__name__}: {e}")
        acc = 0.0

    accs.append(acc)
    per_subject[test_subj] = acc
    print(f"  ✅ Subject {test_subj:>5}: Accuracy = {100*acc:5.2f}%")

elapsed = time() - t0
mean_acc, std_acc = np.mean(accs), np.std(accs)

print("\n" + "=" * 72)
print(f"→ LOSO Mean Accuracy = {100*mean_acc:.2f}% ± {100*std_acc:.2f}%")
print(f"⏱️ Computation Time: {elapsed:.2f}s")
print("=" * 72)

# ============================================================
# 5) PER-SUBJECT SUMMARY
# ============================================================
print("\nPer-Subject Accuracies:")
for subj, acc in per_subject.items():
    print(f"  {subj:>6} : {100*acc:5.2f}%")
