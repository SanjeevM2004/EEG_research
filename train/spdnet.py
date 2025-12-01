"""
SPDNet-Log with Subject Alignment Loss (Mini-DCR)
LOSO evaluation on BCI-IV 2a (bci_active4.pt)
"""

from __future__ import annotations
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from time import time

# ============================================================
# CONFIG
# ============================================================

CACHE_PATH = "./EEG_data/bci_active4.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STEPS = 1000
BATCH_SIZE = 256
LR = 1e-3

LAMBDA_CE = 1.0
LAMBDA_FISHER = 1.0
LAMBDA_WITHIN = 1.0
LAMBDA_BETWEEN = 1.0

# ⭐ NEW: SUBJECT ALIGNMENT WEIGHT
LAMBDA_SUBJ = 0.05

SEED = 50


# ============================================================
# SPD HELPERS
# ============================================================

def symmetrize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.transpose(-1, -2))


def logm_spd(C: torch.Tensor, eps_eig: float = 1e-5, jitter: float = 1e-4) -> torch.Tensor:
    C = symmetrize(C.float())
    B, d, _ = C.shape
    eye = torch.eye(d, device=C.device, dtype=C.dtype).unsqueeze(0)
    C = C + jitter * eye

    w, V = torch.linalg.eigh(C.double())
    w = torch.clamp(w, min=eps_eig)
    S = V @ torch.diag_embed(torch.log(w)) @ V.transpose(-1, -2)
    return symmetrize(S.float())


def init_spdnet_from_global_mean(model: SPDNetLog, C_train: torch.Tensor):
    """
    Initialize SPDNetLog's SPDLinear weights from the eigenbasis
    of the global mean covariance of C_train.

    C_train: (N, d, d) RA-normalized covariances (torch.Tensor, on DEVICE)
    """

    # 1) Compute global mean covariance (simple Euclidean mean is fine)
    M = C_train.mean(dim=0)          # (d, d)
    M = symmetrize(M)

    # 2) Eigendecomposition: M = U Λ U^T
    #    U: d × d (rotation to global eigenbasis)
    eigvals, U = torch.linalg.eigh(M)  # U: d × d
    U = U.float()

    layers = model.spd_layers
    assert len(layers) == 3, "Expecting 3 SPD layers for [d, 2d, 2d, d]"

    d = U.shape[0]
    d2 = 2 * d

    # ----- Layer 0: d → 2d  (W0: d × 2d) -----
    # Use [U, U] / sqrt(2) for a symmetric duplication in the eigenbasis
    W0 = torch.cat([U, U], dim=1) / np.sqrt(2.0)   # (d, 2d)
    layers[0].W.data = W0.to(layers[0].W.data.device)

    # ----- Layer 1: 2d → 2d (W1: 2d × 2d) -----
    # Start as identity: no change initially
    W1 = torch.eye(d2)
    layers[1].W.data = W1.to(layers[1].W.data.device)

    # ----- Layer 2: 2d → d  (W2: 2d × d) -----
    # Use [U; U] / sqrt(2)  (vertical stacking)
    W2 = torch.cat([U, U], dim=0) / np.sqrt(2.0)   # (2d, d)
    layers[2].W.data = W2.to(layers[2].W.data.device)

    # (Optional) initialize classifier near zero
    nn.init.zeros_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)


def tangent_upper_vec(S: torch.Tensor) -> torch.Tensor:
    B, d, _ = S.shape
    idx = torch.triu_indices(d, d, device=S.device)
    return S[:, idx[0], idx[1]]

def subject_alignment_loss(
    feat: torch.Tensor,
    subjects: torch.Tensor,
    lambda_within: float = 1.0,
    lambda_between: float = 1.0,
) -> torch.Tensor:
    """
    Subject Fisher Loss
    --------------------
    Computes:
        W_s = within-subject scatter
        B_s = between-subject scatter

    Returns:
        L = λ_w * W_s  -  λ_b * B_s   (standard Fisher sign)
    """

    B, D = feat.shape
    unique_subj = torch.unique(subjects)

    subj_means = []
    subj_sizes = []
    within = feat.new_tensor(0.0)

    # ------------------------------
    # 1) WITHIN-SUBJECT SCATTER
    # ------------------------------
    for s in unique_subj:
        idx = (subjects == s)
        n_s = idx.sum()

        if n_s <= 1:
            continue

        f_s = feat[idx]                      # (n_s, D)
        mu_s = f_s.mean(0, keepdim=True)     # (1, D)

        within += ((f_s - mu_s) ** 2).sum() / (n_s * D)

        subj_means.append(mu_s)
        subj_sizes.append(n_s)

    # If only one subject in batch → no Fisher structure
    if len(subj_means) <= 1:
        return feat.new_tensor(0.0)

    subj_means = torch.cat(subj_means, dim=0)      # (S, D)
    subj_sizes = torch.tensor(subj_sizes, device=feat.device, dtype=feat.dtype)

    # ------------------------------
    # 2) BETWEEN-SUBJECT SCATTER
    # ------------------------------
    mu_global = subj_means.mean(0, keepdim=True)    # mean of subject means

    # weight by subject size (optional but more stable)
    between = (((subj_means - mu_global) ** 2).sum(dim=1) * subj_sizes).sum()
    between = between / (subj_sizes.sum() * D)

    # ------------------------------
    # 3) FINAL LOSS
    # ------------------------------
    loss = lambda_within * within - lambda_between * between
    return loss


# ============================================================
# SPDLinear LAYER
# ============================================================

class SPDLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, eps: float = 1e-4):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.eps = eps

        # dummy init (will be overridden by data-based init)
        init = torch.randn(d_in, d_out) * (1.0 / np.sqrt(d_in))
        self.W = nn.Parameter(init)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        C = symmetrize(C.float())
        B, d_in, _ = C.shape

        W = self.W.float()
        C = C + self.eps * torch.eye(d_in, device=C.device).unsqueeze(0)

        CW = C @ W
        C_out = CW.transpose(1, 2) @ W
        C_out = symmetrize(C_out)

        C_out = C_out + self.eps * torch.eye(self.d_out, device=C.device).unsqueeze(0)
        return C_out

# ============================================================
# SPDNet-LOG MODEL
# ============================================================

class SPDNetLog(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, spd_dims: list[int] | None = None):
        super().__init__()
        d_in = n_channels

        if spd_dims is None:
            spd_dims = [d_in, d_in, d_in, d_in]

        layers = []
        for din, dout in zip(spd_dims[:-1], spd_dims[1:]):
            layers.append(SPDLinear(din, dout))
        self.spd_layers = nn.ModuleList(layers)

        d_last = spd_dims[-1]
        self.d_last = d_last
        self.feat_dim = d_last * (d_last + 1) // 2

        self.classifier = nn.Linear(self.feat_dim, n_classes)

    def forward(self, C: torch.Tensor):
        C_curr = symmetrize(C.float())
        for layer in self.spd_layers:
            C_curr = layer(C_curr)

        S_last = logm_spd(C_curr)
        feat = tangent_upper_vec(S_last)
        logits = self.classifier(feat)
        return logits, S_last


# ============================================================
# FISHER LOSS
# ============================================================

def fisher_feature_loss(feat, y, lambda_within=1.0, lambda_between=1.0):
    B, D = feat.shape
    mu_global = feat.mean(0, keepdim=True)

    within = 0.0
    between = 0.0
    count = 0

    for c in torch.unique(y):
        idx = (y == c)
        if idx.sum() <= 1:
            continue

        f = feat[idx]
        mu_c = f.mean(0, keepdim=True)

        within += ((f - mu_c) ** 2).sum()
        between += f.shape[0] * ((mu_c - mu_global) ** 2).sum()
        count += f.shape[0]

    if count == 0:
        return torch.tensor(0.0, device=feat.device)

    within = within / (count * D)
    between = between / (count * D)

    return lambda_within * within - lambda_between * between


# ============================================================
# TRAINING + LOSO
# ============================================================

def prepare_covs(covs_np):
    C = torch.from_numpy(covs_np).float()
    C = symmetrize(C)
    tr = C.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    return C / (tr + 1e-6)


def train_one_fold(X_train_np, y_train_np, subj_train_np, n_classes):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    C_train = prepare_covs(X_train_np).to(DEVICE)
    y = torch.tensor(y_train_np, dtype=torch.long, device=DEVICE)
    subj = torch.tensor(subj_train_np, dtype=torch.long, device=DEVICE)

    N, d, _ = C_train.shape

    model = SPDNetLog(
        n_channels=d,
        n_classes=n_classes,
        spd_dims=[d, d*2, d*2, d],
    ).to(DEVICE)

    # ⭐ NEW: initialize W using rotation to global mean
    init_spdnet_from_global_mean(model, C_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    idx_all = torch.arange(N, device=DEVICE)

    best_loss = float("inf")
    best_state = None

    model.train()
    for step in range(1, STEPS + 1):
        batch_idx = idx_all[torch.randint(0, N, (min(BATCH_SIZE, N),), device=DEVICE)]

        C_b = C_train[batch_idx]
        y_b = y[batch_idx]
        subj_b = subj[batch_idx]

        logits, S_last = model(C_b)
        ce = F.cross_entropy(logits, y_b)

        feat_b = tangent_upper_vec(S_last)
        f_loss = fisher_feature_loss(feat_b, y_b)
        subj_loss = subject_alignment_loss(feat_b, subj_b)

        total_loss = (
            LAMBDA_CE * ce
            + LAMBDA_FISHER * f_loss
            + LAMBDA_SUBJ * subj_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {"model": model.state_dict(), "loss": best_loss, "step": step}

        if step == 1 or step % 100 == 0:
            print(
                f"[Train] Step {step}/{STEPS} | "
                f"Loss={total_loss.item():.6f} | CE={ce.item():.4f} | "
                f"Fisher={f_loss.item():.4f} | Subj={subj_loss.item():.4f}"
            )

    model.load_state_dict(best_state["model"])
    model.eval()
    return model

@torch.no_grad()
def evaluate_model(model, X_test_np, y_test_np):
    C_test = prepare_covs(X_test_np).to(DEVICE)
    y_test = torch.tensor(y_test_np, dtype=torch.long, device=DEVICE)

    logits, _ = model(C_test)
    preds = torch.argmax(logits, dim=-1)
    return (preds == y_test).float().mean().item()


def run_loso_experiment(covs, labels, subjects):
    S_ids = np.unique(subjects)
    n_classes = int(labels.max() + 1)

    accs = []
    t0 = time()

    for sid in S_ids:
        train_mask = (subjects != sid)
        test_mask = (subjects == sid)

        model = train_one_fold(
            covs[train_mask],
            labels[train_mask],
            subjects[train_mask],
            n_classes,
        )

        acc = evaluate_model(model, covs[test_mask], labels[test_mask])
        accs.append(acc)

        print(f"  Subject {sid}: {100 * acc:.2f}%")

    accs = np.array(accs)
    print("\nLOSO Mean =", 100 * accs.mean(), "±", 100 * accs.std())


# ============================================================
# MAIN
# ============================================================

def encode_subjects(subj_arr):
    """
    Converts subject labels like 'A01T' into integer IDs 0...S-1.
    """
    unique = np.unique(subj_arr)
    mapping = {s: i for i, s in enumerate(unique)}
    return np.array([mapping[s] for s in subj_arr], dtype=np.int32)


def main():
    data = torch.load(CACHE_PATH, map_location="cpu")
    covs = np.stack([c.cpu().numpy() for c in data["ra_covs"]]).astype(np.float32)
    labels = np.array(data["labels"]).astype(int)

    # ⭐ FIX: convert subjects to numeric IDs
    raw_subjects = np.array(data["subj"])
    subjects = encode_subjects(raw_subjects)

    run_loso_experiment(covs, labels, subjects)


if __name__ == "__main__":
    main()

#======================================================================
#🧾 SPDNet-Log (SPD layers + log-tangent LR) – LOSO SUMMARY
#======================================================================
#      A01T : 69.10%
#      A02T : 27.08%
#      A03T : 81.60%
#      A04T : 43.75%
#      A05T : 42.36%
#      A06T : 36.81%
#      A07T : 59.03%
#      A08T : 75.69%
#      A09T : 65.97%
#----------------------------------------------------------------------
#→ LOSO Mean Accuracy = 55.71% ± 17.84%
#⏱️ Total Time: 195.52s
#======================================================================

'''
[Train] Step 1/1000 | Loss=1.399039 | CE=1.3863 | Fisher=0.0089 | Subj=0.0765
[Train] Step 100/1000 | Loss=1.175640 | CE=1.1627 | Fisher=0.0091 | Subj=0.0774
[Train] Step 200/1000 | Loss=1.069529 | CE=1.0569 | Fisher=0.0089 | Subj=0.0754
[Train] Step 300/1000 | Loss=1.045022 | CE=1.0321 | Fisher=0.0091 | Subj=0.0751
[Train] Step 400/1000 | Loss=1.013723 | CE=1.0006 | Fisher=0.0093 | Subj=0.0764
[Train] Step 500/1000 | Loss=0.950062 | CE=0.9374 | Fisher=0.0090 | Subj=0.0737
[Train] Step 600/1000 | Loss=0.909379 | CE=0.8954 | Fisher=0.0099 | Subj=0.0814
[Train] Step 700/1000 | Loss=0.991747 | CE=0.9797 | Fisher=0.0085 | Subj=0.0704
[Train] Step 800/1000 | Loss=0.890252 | CE=0.8773 | Fisher=0.0092 | Subj=0.0750
[Train] Step 900/1000 | Loss=0.883967 | CE=0.8715 | Fisher=0.0088 | Subj=0.0733
[Train] Step 1000/1000 | Loss=0.843871 | CE=0.8315 | Fisher=0.0087 | Subj=0.0731
  Subject 0: 68.40%
[Train] Step 1/1000 | Loss=1.398986 | CE=1.3863 | Fisher=0.0088 | Subj=0.0777
[Train] Step 100/1000 | Loss=1.110264 | CE=1.0973 | Fisher=0.0090 | Subj=0.0788
[Train] Step 200/1000 | Loss=1.015927 | CE=1.0033 | Fisher=0.0088 | Subj=0.0765
[Train] Step 300/1000 | Loss=1.005891 | CE=0.9930 | Fisher=0.0091 | Subj=0.0757
[Train] Step 400/1000 | Loss=0.918858 | CE=0.9059 | Fisher=0.0091 | Subj=0.0763
[Train] Step 500/1000 | Loss=0.864595 | CE=0.8520 | Fisher=0.0089 | Subj=0.0749
[Train] Step 600/1000 | Loss=0.840029 | CE=0.8262 | Fisher=0.0098 | Subj=0.0816
[Train] Step 700/1000 | Loss=0.862899 | CE=0.8511 | Fisher=0.0082 | Subj=0.0706
[Train] Step 800/1000 | Loss=0.831911 | CE=0.8189 | Fisher=0.0092 | Subj=0.0760
[Train] Step 900/1000 | Loss=0.808389 | CE=0.7962 | Fisher=0.0086 | Subj=0.0734
[Train] Step 1000/1000 | Loss=0.753621 | CE=0.7413 | Fisher=0.0086 | Subj=0.0743
  Subject 1: 29.86%
[Train] Step 1/1000 | Loss=1.398974 | CE=1.3863 | Fisher=0.0089 | Subj=0.0760
[Train] Step 100/1000 | Loss=1.165071 | CE=1.1521 | Fisher=0.0091 | Subj=0.0777
[Train] Step 200/1000 | Loss=1.094540 | CE=1.0820 | Fisher=0.0088 | Subj=0.0743
[Train] Step 300/1000 | Loss=1.064069 | CE=1.0514 | Fisher=0.0090 | Subj=0.0741
[Train] Step 400/1000 | Loss=1.010652 | CE=0.9978 | Fisher=0.0091 | Subj=0.0746
[Train] Step 500/1000 | Loss=0.972671 | CE=0.9602 | Fisher=0.0089 | Subj=0.0730
[Train] Step 600/1000 | Loss=0.936695 | CE=0.9230 | Fisher=0.0097 | Subj=0.0794
[Train] Step 700/1000 | Loss=0.948199 | CE=0.9364 | Fisher=0.0083 | Subj=0.0697
[Train] Step 800/1000 | Loss=0.909819 | CE=0.8969 | Fisher=0.0092 | Subj=0.0746
[Train] Step 900/1000 | Loss=0.899344 | CE=0.8872 | Fisher=0.0085 | Subj=0.0719
[Train] Step 1000/1000 | Loss=0.864275 | CE=0.8520 | Fisher=0.0086 | Subj=0.0730
  Subject 2: 82.64%
[Train] Step 1/1000 | Loss=1.399549 | CE=1.3863 | Fisher=0.0092 | Subj=0.0809
[Train] Step 100/1000 | Loss=1.135805 | CE=1.1230 | Fisher=0.0089 | Subj=0.0775
[Train] Step 200/1000 | Loss=1.034946 | CE=1.0223 | Fisher=0.0088 | Subj=0.0757
[Train] Step 300/1000 | Loss=0.970758 | CE=0.9579 | Fisher=0.0091 | Subj=0.0760
[Train] Step 400/1000 | Loss=0.953933 | CE=0.9412 | Fisher=0.0090 | Subj=0.0744
[Train] Step 500/1000 | Loss=0.905491 | CE=0.8929 | Fisher=0.0089 | Subj=0.0744
[Train] Step 600/1000 | Loss=0.848449 | CE=0.8349 | Fisher=0.0095 | Subj=0.0793
[Train] Step 700/1000 | Loss=0.868778 | CE=0.8568 | Fisher=0.0084 | Subj=0.0718
[Train] Step 800/1000 | Loss=0.872073 | CE=0.8592 | Fisher=0.0091 | Subj=0.0746
[Train] Step 900/1000 | Loss=0.791366 | CE=0.7791 | Fisher=0.0086 | Subj=0.0737
[Train] Step 1000/1000 | Loss=0.783685 | CE=0.7712 | Fisher=0.0087 | Subj=0.0756
  Subject 3: 43.06%
[Train] Step 1/1000 | Loss=1.399479 | CE=1.3863 | Fisher=0.0092 | Subj=0.0802
[Train] Step 100/1000 | Loss=1.146035 | CE=1.1334 | Fisher=0.0089 | Subj=0.0762
[Train] Step 200/1000 | Loss=1.054719 | CE=1.0418 | Fisher=0.0091 | Subj=0.0766
[Train] Step 300/1000 | Loss=0.981614 | CE=0.9687 | Fisher=0.0091 | Subj=0.0762
[Train] Step 400/1000 | Loss=0.951096 | CE=0.9384 | Fisher=0.0090 | Subj=0.0741
[Train] Step 500/1000 | Loss=0.908153 | CE=0.8956 | Fisher=0.0088 | Subj=0.0743
[Train] Step 600/1000 | Loss=0.843936 | CE=0.8303 | Fisher=0.0097 | Subj=0.0804
[Train] Step 700/1000 | Loss=0.834757 | CE=0.8228 | Fisher=0.0084 | Subj=0.0713
[Train] Step 800/1000 | Loss=0.852214 | CE=0.8393 | Fisher=0.0092 | Subj=0.0752
[Train] Step 900/1000 | Loss=0.807104 | CE=0.7949 | Fisher=0.0086 | Subj=0.0732
[Train] Step 1000/1000 | Loss=0.788949 | CE=0.7762 | Fisher=0.0089 | Subj=0.0760
  Subject 4: 44.79%
[Train] Step 1/1000 | Loss=1.398865 | CE=1.3863 | Fisher=0.0088 | Subj=0.0754
[Train] Step 100/1000 | Loss=1.140946 | CE=1.1291 | Fisher=0.0083 | Subj=0.0723
[Train] Step 200/1000 | Loss=1.025266 | CE=1.0130 | Fisher=0.0086 | Subj=0.0726
[Train] Step 300/1000 | Loss=0.961841 | CE=0.9498 | Fisher=0.0085 | Subj=0.0719
[Train] Step 400/1000 | Loss=0.962380 | CE=0.9503 | Fisher=0.0085 | Subj=0.0715
[Train] Step 500/1000 | Loss=0.892582 | CE=0.8812 | Fisher=0.0080 | Subj=0.0678
[Train] Step 600/1000 | Loss=0.814578 | CE=0.8015 | Fisher=0.0092 | Subj=0.0774
[Train] Step 700/1000 | Loss=0.794987 | CE=0.7836 | Fisher=0.0079 | Subj=0.0686
[Train] Step 800/1000 | Loss=0.848964 | CE=0.8367 | Fisher=0.0087 | Subj=0.0714
[Train] Step 900/1000 | Loss=0.804348 | CE=0.7927 | Fisher=0.0082 | Subj=0.0699
[Train] Step 1000/1000 | Loss=0.782370 | CE=0.7701 | Fisher=0.0086 | Subj=0.0730
  Subject 5: 40.28%
[Train] Step 1/1000 | Loss=1.399465 | CE=1.3863 | Fisher=0.0092 | Subj=0.0788
[Train] Step 100/1000 | Loss=1.154753 | CE=1.1425 | Fisher=0.0086 | Subj=0.0748
[Train] Step 200/1000 | Loss=1.053225 | CE=1.0404 | Fisher=0.0090 | Subj=0.0753
[Train] Step 300/1000 | Loss=0.999467 | CE=0.9869 | Fisher=0.0089 | Subj=0.0745
[Train] Step 400/1000 | Loss=0.961978 | CE=0.9496 | Fisher=0.0087 | Subj=0.0736
[Train] Step 500/1000 | Loss=0.931847 | CE=0.9197 | Fisher=0.0085 | Subj=0.0721
[Train] Step 600/1000 | Loss=0.856042 | CE=0.8434 | Fisher=0.0089 | Subj=0.0752
[Train] Step 700/1000 | Loss=0.855860 | CE=0.8439 | Fisher=0.0084 | Subj=0.0715
[Train] Step 800/1000 | Loss=0.885937 | CE=0.8731 | Fisher=0.0092 | Subj=0.0744
[Train] Step 900/1000 | Loss=0.812150 | CE=0.8002 | Fisher=0.0084 | Subj=0.0713
[Train] Step 1000/1000 | Loss=0.816620 | CE=0.8036 | Fisher=0.0092 | Subj=0.0759
  Subject 6: 54.51%
[Train] Step 1/1000 | Loss=1.399244 | CE=1.3863 | Fisher=0.0091 | Subj=0.0763
[Train] Step 100/1000 | Loss=1.176174 | CE=1.1638 | Fisher=0.0087 | Subj=0.0749
[Train] Step 200/1000 | Loss=1.090225 | CE=1.0773 | Fisher=0.0092 | Subj=0.0758
[Train] Step 300/1000 | Loss=1.039009 | CE=1.0261 | Fisher=0.0091 | Subj=0.0757
[Train] Step 400/1000 | Loss=0.995004 | CE=0.9826 | Fisher=0.0088 | Subj=0.0730
[Train] Step 500/1000 | Loss=0.972590 | CE=0.9602 | Fisher=0.0088 | Subj=0.0732
[Train] Step 600/1000 | Loss=0.896040 | CE=0.8831 | Fisher=0.0091 | Subj=0.0762
[Train] Step 700/1000 | Loss=0.881367 | CE=0.8693 | Fisher=0.0085 | Subj=0.0721
[Train] Step 800/1000 | Loss=0.910027 | CE=0.8971 | Fisher=0.0092 | Subj=0.0746
[Train] Step 900/1000 | Loss=0.863395 | CE=0.8513 | Fisher=0.0085 | Subj=0.0721
[Train] Step 1000/1000 | Loss=0.873240 | CE=0.8601 | Fisher=0.0093 | Subj=0.0766
  Subject 7: 76.39%
[Train] Step 1/1000 | Loss=1.398444 | CE=1.3863 | Fisher=0.0086 | Subj=0.0716
[Train] Step 100/1000 | Loss=1.191118 | CE=1.1796 | Fisher=0.0081 | Subj=0.0685
[Train] Step 200/1000 | Loss=1.072600 | CE=1.0605 | Fisher=0.0086 | Subj=0.0708
[Train] Step 300/1000 | Loss=1.024744 | CE=1.0128 | Fisher=0.0084 | Subj=0.0702
[Train] Step 400/1000 | Loss=0.960767 | CE=0.9493 | Fisher=0.0081 | Subj=0.0679
[Train] Step 500/1000 | Loss=0.949539 | CE=0.9377 | Fisher=0.0083 | Subj=0.0695
[Train] Step 600/1000 | Loss=0.885119 | CE=0.8732 | Fisher=0.0084 | Subj=0.0708
[Train] Step 700/1000 | Loss=0.886176 | CE=0.8745 | Fisher=0.0082 | Subj=0.0695
[Train] Step 800/1000 | Loss=0.877962 | CE=0.8662 | Fisher=0.0083 | Subj=0.0688
[Train] Step 900/1000 | Loss=0.813083 | CE=0.8017 | Fisher=0.0080 | Subj=0.0681
[Train] Step 1000/1000 | Loss=0.820640 | CE=0.8082 | Fisher=0.0088 | Subj=0.0726
  Subject 8: 65.62%

LOSO Mean = 56.17283980051676 ± 16.98810582553865
'''