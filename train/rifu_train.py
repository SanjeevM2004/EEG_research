"""
Compare pre-aligned (RA) covariances vs. RiFuNetPreAligner refinement
using TSLR / TS-SVM-RBF / MDM / TSA-LDA / CovCNN on LOSO cross-subject evaluation.
"""

import numpy as np
import torch
from time import time

# ---------------------------------------------------------------
# ⬇️ Import your models
# ---------------------------------------------------------------
from models.riemann.rifu import RiFuNetPreAligner
from models.riemann.tslr import RiemannTSLR
from models.riemann.mdm import RiemannMDM
from models.riemann.tsa_lda import TSALDA
from models.riemann.tssvmrbf import RiemannTS_SVM_RBF   # <-- Latest bug-free version

# ============================================================== #
# CONFIG
# ============================================================== #
CACHE_PATH = "./EEG_data/bci_active4.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RIFU_CFG = dict(
    steps=2000,
    batch_size=256,
    lr=1e-3,
    lambda_within=1.0,
    lambda_between=1.0,
    lambda_subj_var=0.1,
    lambda_recon=1e-3,
    device=DEVICE,
)

np.set_printoptions(precision=3, suppress=True)
print("=" * 70)
print(" RiFuNetPreAligner vs Pure RA + CovCNN ")
print(f" Device: {DEVICE}")
print("=" * 70)


# ============================================================== #
# LOAD DATA
# ============================================================== #
data = torch.load(CACHE_PATH, map_location="cpu")

covs_ra = np.stack([c.cpu().numpy() for c in data["ra_covs"]]).astype(np.float32)
labels  = np.asarray(data["labels"]).astype(int)
subjects = np.asarray(data["subj"])

S_ids = np.unique(subjects)
sid_to_int = {sid: i for i, sid in enumerate(S_ids)}
s_int = np.vectorize(sid_to_int.get)(subjects)

S = len(S_ids)
n_classes = int(labels.max() + 1)
N, d, _ = covs_ra.shape
print(f"N={N}, channels={d}, classes={n_classes}, subjects={S}\n")


# ============================================================== #
# HELPERS
# ============================================================== #
def normalize_covs(X):
    Xn = np.empty_like(X)
    for i in range(len(X)):
        tr = np.trace(X[i])
        if tr > 0:
            Xn[i] = X[i] / tr
        else:
            Xn[i] = X[i]
    return Xn.astype(np.float32)


def evaluate_models(X_train, y_train, X_test, y_test, cov_type="RA"):

    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    # ------------------- Riemann Models -------------------
    tssvmrbf = RiemannTS_SVM_RBF(cov_type=cov_type)
    tslr     = RiemannTSLR(cov_type=cov_type)
    mdm      = RiemannMDM(cov_type=cov_type)
    lda      = TSALDA(cov_type=cov_type)

    tssvmrbf.fit(X_train, y_train)
    tslr.fit(X_train, y_train)
    mdm.fit(X_train, y_train)
    lda.fit(X_train, y_train)

    return (
        tssvmrbf.score(X_test, y_test),
        tslr.score(X_test, y_test),
        mdm.score(X_test, y_test),
        lda.score(X_test, y_test),
    )


# ============================================================== #
# LOSO FOLD
# ============================================================== #
def run_fold(test_sid, use_rifu=False):

    train_mask = (subjects != test_sid)
    test_mask  = (subjects == test_sid)

    X_train, y_train, s_train = covs_ra[train_mask], labels[train_mask], s_int[train_mask]
    X_test,  y_test,  s_test  = covs_ra[test_mask],  labels[test_mask],  s_int[test_mask]

    # ------------------- RiFuNet Pre-Alignment -------------------
    if use_rifu:
        print(f"→ Training RiFuNetPreAligner (excluding {test_sid}) on {len(X_train)} trials")

        pre = RiFuNetPreAligner(
            n_channels=d,
            n_actions=n_classes,
            n_subjects=S,
            **RIFU_CFG,
        )

        pre.fit(X_train, y_action=y_train, y_subject=s_train, verbose=True)

        X_train = pre.transform(X_train).cpu().numpy().astype(np.float32)
        X_test  = pre.transform(X_test).cpu().numpy().astype(np.float32)

    # ------------------- Normalize trace -------------------
    X_train = normalize_covs(X_train)
    X_test  = normalize_covs(X_test)

    # ------------------- Evaluate all models -------------------
    return evaluate_models(X_train, y_train, X_test, y_test, cov_type="RA")


# ============================================================== #
# LOSO LOOP
# ============================================================== #
def run_loso(tag, use_rifu=False):

    print(f"\n{'='*70}\n🧠 Running LOSO ({tag})\n{'='*70}")

    acc_svm = []
    acc_tslr = []
    acc_mdm = []
    acc_lda = []

    t0 = time()

    for sid in S_ids:
        a_svm, a_t, a_m, a_l = run_fold(sid, use_rifu)

        acc_svm.append(a_svm)
        acc_tslr.append(a_t)
        acc_mdm.append(a_m)
        acc_lda.append(a_l)

        print(f"  ✅ Subject {sid:>6}: "
              f"SVM={100*a_svm:5.2f}%  "
              f"TSLR={100*a_t:5.2f}%  "
              f"MDM={100*a_m:5.2f}%  "
              f"LDA={100*a_l:5.2f}%  "
        )

    elapsed = time() - t0

    return (
        np.array(acc_svm),
        np.array(acc_tslr),
        np.array(acc_mdm),
        np.array(acc_lda),
        elapsed
    )


# ============================================================== #
# RUN BOTH EXPERIMENTS
# ============================================================== #
acc_svm_rifu, acc_tslr_rifu, acc_mdm_rifu, acc_lda_rifu, time_rifu = run_loso(
    "RiFuNetPreAligner", use_rifu=True
)

acc_svm_ra, acc_tslr_ra, acc_mdm_ra, acc_lda_ra, time_ra = run_loso(
    "Pure RA baseline", use_rifu=False
)


# ============================================================== #
# SUMMARY TABLE
# ============================================================== #
print("\n" + "="*70)
print("🧾 COMPARISON SUMMARY (Mean ± Std)")
print("="*70)

def summary(name, A, B):
    print(f"{name:<12} | RA: {100*A.mean():6.2f}% ± {100*A.std():5.2f}%"
          f" | RiFu+RA: {100*B.mean():6.2f}% ± {100*B.std():5.2f}%")

summary("TS-SVM-RBF", acc_svm_ra, acc_svm_rifu)
summary("TSLR",        acc_tslr_ra, acc_tslr_rifu)
summary("MDM",         acc_mdm_ra,  acc_mdm_rifu)
summary("TSA-LDA",     acc_lda_ra,  acc_lda_rifu)

print("="*70)
print(f"Runtime → RA: {time_ra:.2f}s | RiFu+RA: {time_rifu:.2f}s")
print("="*70)
