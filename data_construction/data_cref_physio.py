import torch
import numpy as np
from tqdm import tqdm

# -------------------------------------------------------------
# --- Helper: Safe convert to tensor ---------------------------
# -------------------------------------------------------------
def to_tensor(x):
    """Convert list of tensors → single torch.Tensor"""
    if isinstance(x, list):
        return torch.stack([xi if isinstance(xi, torch.Tensor) else torch.tensor(xi) for xi in x])
    return x

# -------------------------------------------------------------
# --- Riemannian (AIRM) mean ----------------------------------
# -------------------------------------------------------------
def riemann_mean_ailogm(C_list, tol=1e-6, max_iter=50):
    """Compute the AIRM mean of SPD matrices."""
    C = C_list.mean(dim=0)
    for _ in range(max_iter):
        w, V = torch.linalg.eigh(C)
        w = torch.clamp(w, min=1e-10)
        C_inv_sqrt = V @ torch.diag(w.rsqrt()) @ V.T

        # Log-maps
        logs = []
        for Ci in C_list:
            tmp = C_inv_sqrt @ Ci @ C_inv_sqrt
            w2, V2 = torch.linalg.eigh(tmp)
            w2 = torch.clamp(w2, min=1e-10)
            logs.append(V2 @ torch.diag(torch.log(w2)) @ V2.T)
        delta = torch.stack(logs).mean(dim=0)

        C_sqrt = V @ torch.diag(torch.sqrt(w)) @ V.T
        C_new = C_sqrt @ torch.linalg.matrix_exp(delta) @ C_sqrt
        if torch.norm(C_new - C) / torch.norm(C) < tol:
            C = C_new
            break
        C = C_new
    return C

# -------------------------------------------------------------
# --- Combine caches and compute per-subject Cref -------------
# -------------------------------------------------------------
def combine_and_add_cref(real_path, imagery_path, save_path=None):
    print(f"📂 Loading caches:\n  1️⃣ {real_path}\n  2️⃣ {imagery_path}")
    real = torch.load(real_path, map_location="cpu")
    imag = torch.load(imagery_path, map_location="cpu")

    # Convert list → tensor if needed
    signals_real = to_tensor(real["signals"])
    signals_imag = to_tensor(imag["signals"])
    ra_covs_real = to_tensor(real["ra_covs"])
    ra_covs_imag = to_tensor(imag["ra_covs"])
    labels_real  = to_tensor(real["labels"])
    labels_imag  = to_tensor(imag["labels"])

    subj_real = np.array(real["subj"])
    subj_imag = np.array(imag["subj"])

    # Merge everything
    signals = torch.cat([signals_real, signals_imag], dim=0)
    ra_covs  = torch.cat([ra_covs_real, ra_covs_imag], dim=0)
    labels   = torch.cat([labels_real, labels_imag], dim=0)
    subj_ids = np.concatenate([subj_real, subj_imag])

    print(f"✅ Combined dataset: {signals.shape[0]} trials, {len(np.unique(subj_ids))} subjects")

    # Compute Riemannian mean per subject
    cref_dict = {}
    print("\n🧮 Computing AIRM mean (Cref) per subject ...")
    for subj in tqdm(np.unique(subj_ids)):
        idx = np.where(subj_ids == subj)[0]
        C_sub = ra_covs[idx]
        cref_dict[subj] = riemann_mean_ailogm(C_sub)

    crefs = torch.stack([cref_dict[sid] for sid in subj_ids])

    # Build final dataset
    combined = {
        "signals": signals,
        "ra_covs": ra_covs,
        "labels": labels,
        "subj": subj_ids.tolist(),
        "crefs": crefs
    }

    if save_path is None:
        save_path = real_path.replace("real_active4.pt", "combined_active4_with_cref.pt")

    torch.save(combined, save_path)
    print(f"\n✅ Saved merged dataset with Cref → {save_path}")
    print(f"Keys: {list(combined.keys())}")

# -------------------------------------------------------------
# --- Example usage -------------------------------------------
# -------------------------------------------------------------
if __name__ == "__main__":
    real_path = "./EEG_data/real_active4.pt"
    imagery_path = "./EEG_data/imagery_active4.pt"
    combine_and_add_cref(real_path, imagery_path)
