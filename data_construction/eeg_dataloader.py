#import torch
#import sys
#import numpy as np
#from torch.utils.data import DataLoader, Subset
#from .EEGFeatureDataset import EEGFeatureDataset
#
## --- 1️⃣ Load the full dataset ---
#cache_path = "./EEG_data/dataset_sub_desc_cache.pt"
#print(cache_path, flush=True)
#dataset = EEGFeatureDataset(
#    root_dir="./EEG_data/Physionet/",
#    fs=160,
#    tmin=-0.5,
#    tmax=4.0,
#    cache_path=cache_path,
#    rebuild=False
#)
#
## --- 2️⃣ Extract unique subject IDs ---
#unique_subjects = sorted(set(dataset.subject_ids))
#print(f"Found {len(unique_subjects)} unique subjects: {unique_subjects}", flush=True)
#
## --- 3️⃣ Split subjects 80/10/10 ---
#n_subj = len(unique_subjects)
#n_train = int(0.8 * n_subj)
#n_val = int(0.1 * n_subj)
#n_test = n_subj - n_train - n_val  # remainder
#
#np.random.seed(42)
#shuffled = np.random.permutation(unique_subjects)
#train_subjects = shuffled[:n_train]
#val_subjects = shuffled[n_train:n_train + n_val]
#test_subjects = shuffled[n_train + n_val:]
#
#print(f"Train subjects: {train_subjects}", flush=True)
#print(f"Val subjects: {val_subjects}", flush=True)
#print(f"Test subjects: {test_subjects}", flush=True)
#
## --- 4️⃣ Map each subject subset to dataset indices ---
#train_indices = [i for i, sid in enumerate(dataset.subject_ids) if sid in train_subjects]
#val_indices = [i for i, sid in enumerate(dataset.subject_ids) if sid in val_subjects]
#test_indices = [i for i, sid in enumerate(dataset.subject_ids) if sid in test_subjects]
#
#print(f"Train samples: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}", flush=True)
#
## --- 5️⃣ Build subsets ---
#train_ds = Subset(dataset, train_indices)
#val_ds = Subset(dataset, val_indices)
#test_ds = Subset(dataset, test_indices)
#
## --- 6️⃣ Create dataloaders ---
#train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
#val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
#test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
#
#print("DataLoaders ready!", flush=True)
#print(f"Train: {len(train_loader.dataset)} samples | "
#      f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}", flush=True)
#
## --- 7️⃣ Example sanity check ---
#for signals, feats, labels, subj_ids in train_loader:
#    print("Batch signals:", signals.shape, flush=True)
#    print("Batch feats:", feats.shape, flush=True)
#    print("Batch labels:", labels.shape, flush=True)
#    print("Subjects in this batch:", set(subj_ids), flush=True)
#    break
#

#import os
#import torch
#import numpy as np
#from torch.utils.data import DataLoader
#from .EEGCovDataset import EEGCovDataset  # or EEGCovDataset if you use that version
#
#
## =============================== Training Example ===============================
#if __name__ == "__main__":
#    """
#    Example: build all caches OR load a specific one for training.
#    """
#
#    ROOT = "./EEG_data/Physionet/"   # folder containing all Physionet subjects
#    OUT  = "./EEG_data/"                # where cache files will be stored
#
#    # --- 1) Build all four caches (only first time) ---
#    print("Building or verifying cached datasets...")
#    builder = EEGCovDataset(root_dir=ROOT, out_dir=OUT, apply_ea=True, per_epoch_norm=False,rebuild=True)
#
#    # --- 2) Load one specific cache for training ---
#    cache_path = os.path.join(OUT, "imagery_active4.pt")  # choose any one
#    dataset = EEGCovDataset(root_dir=ROOT, cache_path=cache_path)
#
#    # --- 3) Inspect dataset ---
#    print(f"\nTotal samples: {len(dataset)}")
#    sig, cov, label, subj = dataset[0]
#    print(f"Sample:\n  signal={sig.shape}, cov={cov.shape}, label={label}, subject={subj}")
#
#    # --- 4) Wrap in DataLoader ---
#    loader = DataLoader(dataset, batch_size=16, shuffle=True)
#    print("\nTesting DataLoader iteration...")
#    for sigs, covs, labels, subjs in loader:
#        print(f"Batch → signals={sigs.shape}, covs={covs.shape}, labels={labels.shape}")
#        break  # just one batch for demo
#
#    # --- 5) (Optional) Convert to numpy for sklearn/pyRiemann ---
#    X = torch.stack(dataset.covs).numpy()   # shape: (N, C, C)
#    y = np.array(dataset.labels)
#    print(f"\nPrepared for Riemann models: X={X.shape}, y={y.shape}")
#

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from .EEGCovDataset import EEGCovDataset   # your class from bci_dataset_final.py

# =============================== Training / Cache Example ===============================
if __name__ == "__main__":
    """
    Example:
      1) Build and cache both datasets (bci_restactive.pt, bci_active4.pt)
      2) Load one for training / Riemannian modeling
    """

    ROOT = "./EEG_data/Physionet/"    # folder with A01T.gdf ... A09T.gdf
    OUT  = "./EEG_data/"                     # cache output directory
#
    # ================================================================
    # 1️⃣ Build caches (run once)
    # ================================================================
    builder = EEGCovDataset(
        root_dir=ROOT,
        out_dir=OUT,
        per_epoch_norm=False,
        rebuild=True   # set to False after first build
    )

    # ================================================================
    # 2️⃣ Load a specific cache (rest vs active OR 4-class active)
    # ================================================================
    # Choose which dataset to load:
    # cache_name = "bci_restactive.pt"   # binary: rest vs active
    cache_name = "./real_active4.pt"        # 4-class: left, right, feet, tongue

    cache_path = os.path.join(OUT, cache_name)
    print(f"\n📂 Loading cached dataset: {cache_path}")

    dataset = EEGCovDataset(root_dir=ROOT, cache_path=cache_path)
    print(f"✅ Loaded {len(dataset)} samples from {cache_name}")

    # ================================================================
    # 3️⃣ Inspect one sample
    # ================================================================
    sig, cov, ra_cov, ea_cov, lea_cov, label, subj = dataset[0]
    print(f"\nSample info:")
    print(f"  signal → {sig.shape}")
    print(f"  riemann aligned_cov    → {ra_cov.shape}")
    print(f"  euclidean aligned_cov  → {ea_cov.shape}")
    print(f"  log-euclidean aligned_cov → {lea_cov.shape}")
    print(f"  cov    → {cov.shape}")
    print(f"  label  → {label}")
    print(f"  subject→ {subj}")

    # ================================================================
    # 4️⃣ DataLoader for batching
    # ================================================================
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    print("\n🔍 Testing DataLoader...")
    for sigs, covs, ra_covs, ea_covs, lea_covs, labels, subjs in loader:
        print(f"Batch → signals={sigs.shape}, covs={covs.shape}, riemann aligned_covs={ra_covs.shape}, euclidean aligned_covs={ea_covs.shape}, log-euclidean aligned_covs={lea_covs.shape}, labels={labels.shape}")
        break

    # ================================================================
    # 5️⃣ Convert to NumPy for Riemannian models
    # ================================================================
    X = np.stack([c.numpy() for c in dataset.ra_covs])   # (N, C, C)
    y = np.array(dataset.labels)
    print(np.unique(y, return_counts=True))
    print(f"\n📈 Ready for Riemannian models: X={X.shape}, y={y.shape}")
