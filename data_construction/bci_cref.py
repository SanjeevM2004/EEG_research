import torch
import numpy as np
from tqdm import tqdm

# -------------------------------------------------------------
# Helper: ensure list -> tensor
# -------------------------------------------------------------
def to_tensor(x):
    if isinstance(x, list):
        return torch.stack([xi if isinstance(xi, torch.Tensor) else torch.tensor(xi) for xi in x])
    return x

# -------------------------------------------------------------
# Riemannian mean (AIRM)
# -------------------------------------------------------------
def riemann_mean_ailogm(C_list, tol=1e-6, max_iter=50):
    C = C_list.mean(dim=0)
    for _ in range(max_iter):
        w, V = torch.linalg.eigh(C)
        w = torch.clamp(w, min=1e-10)
        C_inv_sqrt = V @ torch.diag(w.rsqrt()) @ V.T
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
# Compute and attach Cref per subject
# -------------------------------------------------------------
def add_cref_bci(cache_path: str, save_path: str = None):
    print(f"📂 Loading dataset: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")
    print(f"Keys: {list(data.keys())}")

    signals = to_tensor(data["signals"])
    ra_covs  = to_tensor(data["ra_covs"])
    subj_ids = np.array(data["subj"])

    unique_subs = np.unique(subj_ids)
    print(f"Detected {len(unique_subs)} subjects: {unique_subs}")

    cref_dict = {}
    print("\n🧮 Computing AIRM mean (Cref) per subject...")
    for subj in tqdm(unique_subs):
        idx = np.where(subj_ids == subj)[0]
        C_sub = ra_covs[idx]
        cref_dict[subj] = riemann_mean_ailogm(C_sub)

    crefs = torch.stack([cref_dict[sid] for sid in subj_ids])
    data["crefs"] = crefs

    if save_path is None:
        save_path = cache_path.replace(".pt", "_with_cref.pt")

    torch.save(data, save_path)
    print(f"\n✅ Saved dataset with Cref → {save_path}")
    print(f"Keys now: {list(data.keys())}")

# -------------------------------------------------------------
# Example
# -------------------------------------------------------------
if __name__ == "__main__":
    cache_path = "./EEG_data/bci_active4.pt"
    add_cref_bci(cache_path)
