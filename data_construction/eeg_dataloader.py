import torch
import sys
import numpy as np
from torch.utils.data import DataLoader, Subset
from .EEGFeatureDataset import EEGFeatureDataset

# --- 1️⃣ Load the full dataset ---
cache_path = "./EEG_data/dataset_sub_desc_cache.pt"
print(cache_path, flush=True)
dataset = EEGFeatureDataset(
    root_dir="./EEG_data/Physionet/",
    fs=160,
    tmin=-0.5,
    tmax=4.0,
    cache_path=cache_path,
    rebuild=False
)

# --- 2️⃣ Extract unique subject IDs ---
unique_subjects = sorted(set(dataset.subject_ids))
print(f"Found {len(unique_subjects)} unique subjects: {unique_subjects}", flush=True)

# --- 3️⃣ Split subjects 80/10/10 ---
n_subj = len(unique_subjects)
n_train = int(0.8 * n_subj)
n_val = int(0.1 * n_subj)
n_test = n_subj - n_train - n_val  # remainder

np.random.seed(42)
shuffled = np.random.permutation(unique_subjects)
train_subjects = shuffled[:n_train]
val_subjects = shuffled[n_train:n_train + n_val]
test_subjects = shuffled[n_train + n_val:]

print(f"Train subjects: {train_subjects}", flush=True)
print(f"Val subjects: {val_subjects}", flush=True)
print(f"Test subjects: {test_subjects}", flush=True)

# --- 4️⃣ Map each subject subset to dataset indices ---
train_indices = [i for i, sid in enumerate(dataset.subject_ids) if sid in train_subjects]
val_indices = [i for i, sid in enumerate(dataset.subject_ids) if sid in val_subjects]
test_indices = [i for i, sid in enumerate(dataset.subject_ids) if sid in test_subjects]

print(f"Train samples: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}", flush=True)

# --- 5️⃣ Build subsets ---
train_ds = Subset(dataset, train_indices)
val_ds = Subset(dataset, val_indices)
test_ds = Subset(dataset, test_indices)

# --- 6️⃣ Create dataloaders ---
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

print("DataLoaders ready!", flush=True)
print(f"Train: {len(train_loader.dataset)} samples | "
      f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}", flush=True)

# --- 7️⃣ Example sanity check ---
for signals, feats, labels, subj_ids in train_loader:
    print("Batch signals:", signals.shape, flush=True)
    print("Batch feats:", feats.shape, flush=True)
    print("Batch labels:", labels.shape, flush=True)
    print("Subjects in this batch:", set(subj_ids), flush=True)
    break
