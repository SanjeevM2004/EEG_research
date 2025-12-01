import os, glob, gc, torch, mne
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from preprocessing.riemann_manifold_alignment import (
    riemann_alignment_trace,       # RA
    euclidean_alignment_trace,     # EA
    logeuclidean_alignment_trace   # LEA
)

mne.set_log_level("ERROR")

# =============================== Helpers ===============================
ACTION_LABELS = {"rest": 0, "left_fist": 1, "right_fist": 2, "both_fists": 3, "feet": 4}

def run_is_real(rid):     return rid in [3, 5, 7, 9, 11, 13]
def run_is_imagery(rid):  return rid in [4, 6, 8, 10, 12, 14]

def map_event(run_id: int, event_id: int) -> int:
    """Map PhysioNet event codes to unified action labels."""
    if run_id in [1, 2]:
        return ACTION_LABELS["rest"]
    elif run_id in [3, 4, 7, 8, 11, 12]:
        if event_id == 1:
            return ACTION_LABELS["rest"]
        elif event_id == 2:
            return ACTION_LABELS["left_fist"]
        elif event_id == 3:
            return ACTION_LABELS["right_fist"]
    elif run_id in [5, 6, 9, 10, 13, 14]:
        if event_id == 1:
            return ACTION_LABELS["rest"]
        elif event_id == 2:
            return ACTION_LABELS["both_fists"]
        elif event_id == 3:
            return ACTION_LABELS["feet"]
    raise ValueError(f"Unknown run_id {run_id} or event_id {event_id}")

# ---------------------- Covariance (SPD) ----------------------
def compute_cov(sig: torch.Tensor, eps: float = 1e-6, trace_norm: bool = True) -> torch.Tensor:
    sig = sig - sig.mean(dim=1, keepdim=True)
    cov = sig @ sig.T / max(sig.shape[1] - 1, 1)
    cov = cov + eps * torch.eye(sig.shape[0], device=sig.device)
    if trace_norm:
        tr = torch.trace(cov)
        if tr > 0:
            cov = cov / tr
    return cov

# =============================== Dataset ===============================
class EEGCovDataset(Dataset):
    """
    PhysioNet EEG builder/loader with per-subject EA/RA/LEA alignment.
    """

    def __init__(self,
                 root_dir: str = None,
                 out_dir: str = None,
                 fs: float = 160,
                 tmin: float = -0.5,
                 tmax: float = 4.0,
                 per_epoch_norm: bool = False,
                 rebuild: bool = False,
                 cache_path: str = None):
        self.fs = fs
        self.tmin = tmin
        self.tmax = tmax
        self.per_epoch_norm = per_epoch_norm
        self.rebuild = rebuild
        self.target_len = int((tmax - tmin) * fs) + 1

        # Storage for iteration mode
        self.signals = []
        self.covs = []
        self.ea_covs = []
        self.ra_covs = []
        self.lea_covs = []
        self.labels = []
        self.subj_ids = []

        # BUILD mode
        if root_dir is not None and out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            self._build_all_once(root_dir, out_dir)

        # LOAD mode
        if cache_path is not None:
            self._load_cache(cache_path)

    # ---------------------------------------------------------------
    def _new_bucket(self):
        return {
            "signals": [],
            "covs": [],
            "ea_covs": [],
            "ra_covs": [],
            "lea_covs": [],
            "labels": [],
            "subj": []
        }

    # ---------------------------------------------------------------
    def _build_all_once(self, root_dir: str, out_dir: str):
        """
        Parse EDFs → epochs → signals/covs/labels/subj into 4 buckets.
        Compute EA/RA/LEA per subject within each bucket and save .pt files.
        """
        buckets = {
            "real_active4":       self._new_bucket(),
            "real_restactive":    self._new_bucket(),
            "imagery_active4":    self._new_bucket(),
            "imagery_restactive": self._new_bucket(),
        }

        subjects = sorted({
            os.path.basename(f).split("R")[0]
            for f in glob.glob(os.path.join(root_dir, "**/*.edf"), recursive=True)
        })
        for sid in subjects:
            print(f"\n=== Subject {sid} ===")
            files = sorted(glob.glob(os.path.join(root_dir, sid, "*.edf")))
            if not files:
                print(f"[warn] no EDF for {sid}")
                continue

            for f in files:
                base = os.path.basename(f)
                try:
                    rid = int(base.split("R")[1][:2])
                except Exception:
                    print(f"[skip {f}] cannot parse run id")
                    continue

                kind = "real" if run_is_real(rid) else "imagery" if run_is_imagery(rid) else None
                if kind is None:
                    continue

                try:
                    raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
                    sfreq = raw.info["sfreq"]
                    nyquist = sfreq / 2.0
                    raw.filter(1, min(79, 0.99 * nyquist), fir_design="firwin", verbose=False)
                    raw.notch_filter(60, verbose=False)
                    if abs(sfreq - self.fs) > 1e-3:
                        raw.resample(self.fs)
                except Exception as e:
                    print(f"[skip {f}] {e}")
                    continue

                events, _ = mne.events_from_annotations(raw, verbose=False)
                if len(events) == 0:
                    continue

                epochs = mne.Epochs(
                    raw, events, tmin=self.tmin, tmax=self.tmax,
                    baseline=None, preload=True, verbose=False,
                )
                data = epochs.get_data().astype("float32")  # (N,C,T)
                ev = epochs.events[:, -1]
                labels = [map_event(rid, e) for e in ev]

                for x_np, lab in zip(data, labels):
                    if self.per_epoch_norm:
                        x_np = (x_np - x_np.mean(axis=-1, keepdims=True)) / (
                            x_np.std(axis=-1, keepdims=True) + 1e-8
                        )

                    x = torch.tensor(x_np)
                    x = self._pad(x)
                    cov = compute_cov(x)

                    # Assign to buckets
                    if kind == "real":
                        b = buckets["real_restactive"]
                        b["signals"].append(x.cpu())
                        b["covs"].append(cov.cpu())
                        b["labels"].append(0 if lab == 0 else 1)
                        b["subj"].append(sid)

                        if lab != 0:
                            b2 = buckets["real_active4"]
                            b2["signals"].append(x.cpu())
                            b2["covs"].append(cov.cpu())
                            b2["labels"].append(lab - 1)
                            b2["subj"].append(sid)

                    elif kind == "imagery":
                        b = buckets["imagery_restactive"]
                        b["signals"].append(x.cpu())
                        b["covs"].append(cov.cpu())
                        b["labels"].append(0 if lab == 0 else 1)
                        b["subj"].append(sid)

                        if lab != 0:
                            b2 = buckets["imagery_active4"]
                            b2["signals"].append(x.cpu())
                            b2["covs"].append(cov.cpu())
                            b2["labels"].append(lab - 1)
                            b2["subj"].append(sid)

            gc.collect()
            print(f"{sid} processed.")

        # -------------------- Compute EA/RA/LEA and save --------------------
        for name, b in buckets.items():
            path = os.path.join(out_dir, f"{name}.pt")
            if os.path.exists(path) and not self.rebuild:
                print(f"[Skip existing] {path}")
                continue

            subs = b["subj"]
            covs = b["covs"]
            if len(subs) == 0:
                print(f"[warn] {name}: empty bucket.")
                torch.save(b, path)
                continue

            subj_groups = sorted(set(subs))
            ea_covs = [None] * len(covs)
            ra_covs = [None] * len(covs)
            lea_covs = [None] * len(covs)

            for sid in tqdm(subj_groups, desc=f"Aligning {name}"):
                idxs = [i for i, s in enumerate(subs) if s == sid]
                subj_covs = [covs[i] for i in idxs]
                try:
                    ea_subj = euclidean_alignment_trace(subj_covs)
                except Exception as e:
                    print(f"  ⚠️ EA fail {sid}: {e} (fallback=identity)")
                    ea_subj = subj_covs
                try:
                    ra_subj = riemann_alignment_trace(subj_covs)
                except Exception as e:
                    print(f"  ⚠️ RA fail {sid}: {e} (fallback=identity)")
                    ra_subj = subj_covs
                try:
                    lea_subj = logeuclidean_alignment_trace(subj_covs)
                except Exception as e:
                    print(f"  ⚠️ LEA fail {sid}: {e} (fallback=identity)")
                    lea_subj = subj_covs

                for j, idx in enumerate(idxs):
                    ea_covs[idx]  = ea_subj[j].cpu()
                    ra_covs[idx]  = ra_subj[j].cpu()
                    lea_covs[idx] = lea_subj[j].cpu()

            b["ea_covs"] = ea_covs
            b["ra_covs"] = ra_covs
            b["lea_covs"] = lea_covs

            torch.save(b, path)
            print(f"[✓] Saved {name}: {len(b['labels'])} samples → {path}")

    # ---------------------------------------------------------------
    def _load_cache(self, cache_path: str):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(cache_path)
        d = torch.load(cache_path, map_location="cpu")
        for key in ["signals", "covs", "ea_covs", "ra_covs", "lea_covs", "labels", "subj"]:
            if key not in d:
                raise KeyError(f"Missing key '{key}' in cache: {cache_path}")
        n = len(d["labels"])
        assert len(d["signals"])==len(d["covs"])==len(d["ea_covs"])==len(d["ra_covs"])==len(d["lea_covs"])==len(d["subj"])==n
        self.signals  = d["signals"]
        self.covs     = d["covs"]
        self.ea_covs  = d["ea_covs"]
        self.ra_covs  = d["ra_covs"]
        self.lea_covs = d["lea_covs"]
        self.labels   = d["labels"]
        self.subj_ids = d["subj"]
        print(f"Loaded cache: {cache_path} → {n} samples")

    # ---------------------------------------------------------------
    def _pad(self, s: torch.Tensor) -> torch.Tensor:
        c, t = s.shape
        if t < self.target_len:
            return F.pad(s, (0, self.target_len - t))
        else:
            return s[:, :self.target_len]

    # -------------------- PyTorch Dataset API --------------------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.signals[idx],
            self.covs[idx],
            self.ra_covs[idx],
            self.ea_covs[idx],
            self.lea_covs[idx],
            torch.tensor(self.labels[idx]).long(),
            self.subj_ids[idx],
        )

# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1) BUILD caches
    builder = EEGCovDataset(
        root_dir="./EEG_raw/physionet/",
        out_dir="./EEG_data/physionet_caches/",
        fs=160,
        tmin=-0.5,
        tmax=4.0,
        per_epoch_norm=False,
        rebuild=True,
        cache_path=None,
    )

    # 2) LOAD one bucket to iterate (e.g., imagery_active4)
    ds = EEGCovDataset(
        cache_path="./EEG_data/physionet_caches/imagery_active4.pt"
    )
    print("Samples loaded:", len(ds))
    if len(ds) > 0:
        s, cov, ra, ea, lea, y, subj = ds[0]
        print("Shapes:", s.shape, cov.shape, ra.shape, ea.shape, lea.shape,
              "label=", y.item(), "subj=", subj)
    else:
        print("⚠️ No samples found. Check dataset/event mapping.")
