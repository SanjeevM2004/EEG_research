"""
Compare pre-aligned (RA) covariances vs. DCRPreAlignerDualFast refinement
using TSLR / TS-SVM-RBF / MDM / TSA-LDA on LOSO cross-subject evaluation.
"""

import numpy as np
import torch
from time import time

# ---------------------------------------------------------------
# ⬇️ Import your models
# ---------------------------------------------------------------
from models.riemann.dcrbifa import DCRPreAlignerDualFast
from models.riemann.tslr import RiemannTSLR
from models.riemann.mdm import RiemannMDM
from models.riemann.tsa_lda import TSALDA
from models.riemann.tssvmrbf import RiemannTS_SVM_RBF   

# ============================================================== #
# Config
# ============================================================== #
CACHE_PATH = "./EEG_data/bci_active4.pt"
DEVICE = "cpu"

DCR_CFG = dict(
    steps=400,
    lr=1e-3,
    gamma_class=1.5,
    delta_subject=0.3,
    beta_reg=4e-4,
    kappa_center=1e-4,
    eps=1e-12,
    device=DEVICE,
    normalize_log=True,
    rewhiten_after=True,
    clip_grad_norm=5.0,
    verbose_every=50,
)

np.set_printoptions(precision=3, suppress=True)
print("=" * 70)
print(" DCRPreAlignerDualFast vs Pure RA (Now with TS-SVM-RBF added) ")
print(f" Device: {DEVICE}")
print("=" * 70)

# ============================================================== #
# Load pre-aligned covariances
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
# Helpers
# ============================================================== #
def normalize_covs(X):
    Xn = np.empty_like(X)
    for i in range(len(X)):
        tr = np.trace(X[i])
        Xn[i] = X[i] / tr if tr > 0 else X[i]
    return Xn.astype(np.float32)

def evaluate_models(X_train, y_train, X_test, y_test, cov_type="RA"):
    """
    Return accuracies for:
      1) TS-SVM-RBF
      2) TSLR
      3) MDM
      4) TSA-LDA
    """
    # Ensure float32 before pyriemann
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

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
# LOSO Cross-Subject Evaluation
# ============================================================== #
def run_fold(test_sid, use_dcr=False):

    train_mask = (subjects != test_sid)
    test_mask  = (subjects == test_sid)

    X_train, y_train, s_train = covs_ra[train_mask], labels[train_mask], s_int[train_mask]
    X_test,  y_test,  s_test  = covs_ra[test_mask],  labels[test_mask],  s_int[test_mask]

    if use_dcr:
        print(f"→ Training DCRPreAlignerDualFast (excluding {test_sid}) on {len(X_train)} trials")
        pre = DCRPreAlignerDualFast(**DCR_CFG)
        X_train = pre.fit_transform(X_train, y_train, s=s_train, verbose=False).astype(np.float32)
        X_test  = pre.transform(X_test).astype(np.float32)

    X_train = normalize_covs(X_train)
    X_test  = normalize_covs(X_test)

    return evaluate_models(X_train, y_train, X_test, y_test, cov_type="RA")


def run_loso(tag, use_dcr=False):
    print(f"\n{'='*70}\n🧠 Running LOSO ({tag})\n{'='*70}")

    accs_svm, accs_tslr, accs_mdm, accs_lda = [], [], [], []
    t0 = time()

    for sid in S_ids:
        a_svm, a_t, a_m, a_l = run_fold(sid, use_dcr)

        accs_svm.append(a_svm)
        accs_tslr.append(a_t)
        accs_mdm.append(a_m)
        accs_lda.append(a_l)

        print(f"  ✅ Subject {sid:>6}: "
              f"TS-SVM-RBF={100*a_svm:5.2f}%  "
              f"TSLR={100*a_t:5.2f}%  "
              f"MDM={100*a_m:5.2f}%  "
              f"TSA-LDA={100*a_l:5.2f}%"
        )

    elapsed = time() - t0
    accs_svm, accs_tslr, accs_mdm, accs_lda = map(np.asarray,
                                                  (accs_svm, accs_tslr, accs_mdm, accs_lda))

    print(f"\n→ LOSO Mean Accuracy ({tag})")
    print(f"   TS-SVM-RBF: {100*accs_svm.mean():.2f}% ± {100*accs_svm.std():.2f}%")
    print(f"   TSLR:       {100*accs_tslr.mean():.2f}% ± {100*accs_tslr.std():.2f}%")
    print(f"   MDM:        {100*accs_mdm.mean():.2f}% ± {100*accs_mdm.std():.2f}%")
    print(f"   TSA-LDA:    {100*accs_lda.mean():.2f}% ± {100*accs_lda.std():.2f}%")
    print(f"⏱️  Time ({tag}): {elapsed:.2f}s\n")

    return accs_svm, accs_tslr, accs_mdm, accs_lda, elapsed

# ============================================================== #
# Run both setups
# ============================================================== #
accs_svm_dcr, accs_tslr_dcr, accs_mdm_dcr, accs_lda_dcr, time_dcr = run_loso(
    "DCRPreAlignerDualFast (Dual Fisher)", use_dcr=True
)

accs_svm_ra, accs_tslr_ra, accs_mdm_ra, accs_lda_ra, time_ra = run_loso(
    "Pure Riemannian Alignment (RA baseline)", use_dcr=False
)

# ============================================================== #
# Summary
# ============================================================== #
print("\n" + "="*70)
print("🧾 COMPARISON SUMMARY (Mean ± Std)")
print("="*70)

def summary_line(name, a_ra, a_dcr):
    mean_ra, std_ra = a_ra.mean(), a_ra.std()
    mean_dcr, std_dcr = a_dcr.mean(), a_dcr.std()
    print(f"{name:<12} | RA: {100*mean_ra:6.2f}% ± {100*std_ra:5.2f}%"
          f" | DCR+RA: {100*mean_dcr:6.2f}% ± {100*std_dcr:5.2f}%")

summary_line("TS-SVM-RBF", accs_svm_ra, accs_svm_dcr)
summary_line("TSLR",       accs_tslr_ra, accs_tslr_dcr)
summary_line("MDM",        accs_mdm_ra,  accs_mdm_dcr)
summary_line("TSA-LDA",    accs_lda_ra,  accs_lda_dcr)

print("="*70)
print(f"Runtime → RA: {time_ra:.2f}s | DCR+RA: {time_dcr:.2f}s")
print("="*70)
