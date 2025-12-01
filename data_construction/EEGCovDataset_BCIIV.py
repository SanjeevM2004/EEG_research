import os, glob, gc, warnings
import numpy as np
import torch, mne
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.linalg

# ------------------------------------------------------------------
# FINAL universal NumPy hot-patch: fromstring(binary) → frombuffer
# ------------------------------------------------------------------
import numpy as np

if not hasattr(np, "_fromstring_patched"):
    _orig_fromstring = np.fromstring

    def _patched_fromstring(string, dtype=float, count=-1, sep=""):
        """
        Universal fix for NumPy>=2.0 when libraries (e.g., MNE GDF) still use
        np.fromstring on binary blobs. We first try the original behavior; on
        ValueError (binary mode removed) we convert *anything* to bytes and use
        np.frombuffer. This covers str/bytes/bytearray/memoryview/np.void/
        0-D or N-D np.ndarray, etc.
        """
        try:
            # fast path: if this is a normal text case, keep default semantics
            return _orig_fromstring(string, dtype=dtype, count=count, sep=sep)
        except ValueError as e:
            # fallback path: coerce to raw bytes and use frombuffer
            # 1) normalize to bytes
            if isinstance(string, bytes):
                b = string
            elif isinstance(string, bytearray):
                b = bytes(string)
            elif isinstance(string, memoryview):
                b = string.tobytes()
            elif isinstance(string, np.ndarray):
                b = string.tobytes()
            elif isinstance(string, np.void):
                b = bytes(string)
            elif isinstance(string, str):
                # GDF header fields may arrive as "binary" in a Python str;
                # preserve byte values 0..255 via latin-1 (1:1 mapping)
                b = string.encode("latin1", errors="ignore")
            else:
                # last resort
                try:
                    b = bytes(string)
                except Exception:
                    raise e
            # 2) safe binary parse
            return np.frombuffer(b, dtype=dtype, count=count)

    np.fromstring = _patched_fromstring
    np._fromstring_patched = True
    print("✅ Patched numpy.fromstring → universal fallback to frombuffer.")

# Silence the harmless duplicate channel-name warning
warnings.filterwarnings("ignore", message="Channel names are not unique")

from pyriemann.estimation import Covariances
from preprocessing.riemann_manifold_alignment import (
    riemann_alignment_trace, euclidean_alignment_trace, logeuclidean_alignment_trace
)

# ================================================================
# Helpers / constants
# ================================================================
def compute_cov_pyr(sig_np: np.ndarray, estimator: str = "oas") -> torch.Tensor:
    """Compute covariance (OAS) with per-channel centering + trace normalization."""
    sig_np = sig_np - sig_np.mean(axis=1, keepdims=True)
    cov = Covariances(estimator=estimator).transform(sig_np[None, ...])[0]  # (C,C)
    cov = cov / np.trace(cov)  # trace-normalize
    return torch.from_numpy(cov.astype(np.float32))

def _pad_to(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or truncate (C,T) to target_len"""
    c, t = x.shape
    return F.pad(x, (0, target_len - t)) if t < target_len else x[:, :target_len]

# ================================================================
# Dataset
# ================================================================
class EEGCovDataset(Dataset):
    """
    BCI Competition IV-2a builder (TRAIN files only: A0*T.gdf)

    - Extracts 4-class motor imagery epochs (left/right/feet/tongue).
    - Uses robust event mapping via MNE's event_dict (works across MNE versions).
    - Applies band-pass + notch, optional per-epoch z-norm.
    - Outputs raw covs plus RA/EA/LogEA aligned covs.
    - Saves ONE cache: bci_active4.pt  (labels 0..3).

    You can extend similarly to add a 'rest' cache if needed.
    """

    def __init__(self, root_dir, out_dir=None,
                 fs=250, mi_tmin=0.5, mi_tmax=3.5,
                 bp_lo=8, bp_hi=35, notch=50,
                 drop_eog=True, per_epoch_norm=False,
                 cache_path=None, rebuild=False):
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.fs = fs
        self.mi_tmin, self.mi_tmax = mi_tmin, mi_tmax
        self.bp_lo, self.bp_hi, self.notch = bp_lo, bp_hi, notch
        self.drop_eog, self.per_epoch_norm, self.rebuild = drop_eog, per_epoch_norm, rebuild
        self.target_len = int((mi_tmax - mi_tmin) * fs)

        if cache_path:
            d = torch.load(cache_path, map_location="cpu")
            self.signals  = d["signals"]
            self.covs     = d["covs"]
            self.ra_covs  = d["ra_covs"]
            self.ea_covs  = d["ea_covs"]
            self.lea_covs = d["lea_covs"]
            self.labels   = d["labels"]
            self.subj_ids = d["subj"]
        else:
            os.makedirs(out_dir, exist_ok=True)
            self._build_all_once(out_dir)

    # ---------------------------------------------------------------
    def _build_all_once(self, out_dir):
        # buckets[name] = (signals, covs, ra_covs, ea_covs, lea_covs, labels, subj)
        buckets = {"bci_active4": ([], [], [], [], [], [], [])}

        # ONLY TRAINING FILES (labels present)
        files = sorted(glob.glob(os.path.join(self.root_dir, "A0*T.gdf")))
        if not files:
            raise FileNotFoundError(f"No training .gdf files found in {self.root_dir}")

        for f in files:
            sid = os.path.basename(f).split(".")[0]
            print(f"\n=== Subject {sid} ===")

            # -------- Read raw safely --------
            raw = mne.io.read_raw_gdf(f, preload=True, verbose=False)

            # Drop/keep channels by type (avoid name assumptions)
            raw.pick(eeg=True, eog=not self.drop_eog)

            # Basic filtering
            raw.filter(self.bp_lo, self.bp_hi, fir_design="firwin", verbose=False)
            raw.notch_filter(self.notch, verbose=False)

            # -------- Events & robust mapping --------
            events, event_dict = mne.events_from_annotations(raw, verbose=False)
            # Example (varies by MNE): {'1023':1,'1072':2,'768':6,'769':7,'770':8,'771':9,'772':10}
            print("  Annotation keys found:", list(event_dict.keys())[:20])
            print("  Found", len(events), "events with IDs:", np.unique(events[:, -1]))

            # Find MNE IDs that correspond to "769,770,771,772"
            wanted_str = ['769', '770', '771', '772']
            mi_ids = [event_dict[s] for s in wanted_str if s in event_dict]
            if not mi_ids:
                print("  → No MI IDs present in event_dict:", event_dict)
                print(f"[warn] {sid}: no valid MI events → skipped MI part")
                continue

            # Filter events to MI only (using MNE's numeric IDs)
            mi_mask = np.isin(events[:, -1], mi_ids)
            mi_events = events[mi_mask]
            if mi_events.size == 0:
                print("  → No MI events found after filtering with IDs:", mi_ids)
                print(f"[warn] {sid}: no valid MI events → skipped MI part")
                continue

            # Build inverse map to recover the original '769'/'770'/... label
            inv_event_dict = {v: k for k, v in event_dict.items()}
            code_to_label = {'769': 1, '770': 2, '771': 3, '772': 4}

            # -------- Epoch & feature extraction --------
            subj_covs = []
            for e in mi_events:
                code_str = inv_event_dict[int(e[-1])]  # e.g., '769'
                lab = code_to_label.get(code_str, None)
                if lab is None:
                    continue  # skip anything unexpected

                # Single-epoch extraction around the cue
                epochs = mne.Epochs(
                    raw, np.array([e]),
                    tmin=self.mi_tmin, tmax=self.mi_tmax,
                    baseline=None, preload=True, verbose=False
                )
                x_np = epochs.get_data()[0].astype("float32")  # (C,T)

                if self.per_epoch_norm:
                    x_np = (x_np - x_np.mean(axis=-1, keepdims=True)) / (x_np.std(axis=-1, keepdims=True) + 1e-8)

                x = torch.tensor(x_np)
                x = _pad_to(x, self.target_len)
                cov = compute_cov_pyr(x.numpy())  # (C,C)
                subj_covs.append(cov)

                # Accumulate (signals, covs, _, _, _, labels(0..3), subject)
                buckets["bci_active4"][0].append(x)
                buckets["bci_active4"][1].append(cov)
                buckets["bci_active4"][5].append(lab - 1)  # shift to 0..3
                buckets["bci_active4"][6].append(sid)

            # -------- Subject-wise alignment(s) --------
            if len(subj_covs) > 1:
                ra_covs  = riemann_alignment_trace(subj_covs)
                ea_covs  = euclidean_alignment_trace(subj_covs)
                lea_covs = logeuclidean_alignment_trace(subj_covs)
            else:
                ra_covs = ea_covs = lea_covs = subj_covs

            for ra_cov in ra_covs:
                buckets["bci_active4"][2].append(ra_cov)
            for ea_cov in ea_covs:
                buckets["bci_active4"][3].append(ea_cov)
            for lea_cov in lea_covs:
                buckets["bci_active4"][4].append(lea_cov)

            gc.collect(); torch.cuda.empty_cache()
            print(f"{sid} processed: {len(subj_covs)} trials.")

        # -------- Save cache --------
        name = "bci_active4"
        sigs, covs, ra_covs, ea_covs, lea_covs, labs, subs = buckets[name]
        path = os.path.join(out_dir, f"{name}.pt")
        torch.save(
            {"signals": sigs, "covs": covs, "ra_covs": ra_covs, "ea_covs": ea_covs,
             "lea_covs": lea_covs, "labels": labs, "subj": subs},
            path
        )
        print(f"[✓] Saved {path}  (N={len(labs)})")

    # ---------------------------------------------------------------
    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.signals[idx], self.covs[idx],
            self.ra_covs[idx], self.ea_covs[idx], self.lea_covs[idx],
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.subj_ids[idx],
        )
