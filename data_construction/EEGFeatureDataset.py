import os, glob, warnings, gc, psutil, torch
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset
import torch.nn.functional as F
import mne
from sklearn.exceptions import ConvergenceWarning

from preprocessing.filters import bandpass_filter, notch_filter
from preprocessing.epoching import create_epochs
from preprocessing.normalize import zscore_normalize
from feature_extraction.builder import build_feature_vector


# ---------------- Label Maps ----------------
ACTION_LABELS = {
    "rest": 0,
    "left_fist": 1,
    "right_fist": 2,
    "both_fists": 3,
    "feet": 4,
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def map_event_to_action(run_id: int, event_id: int) -> int:
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


# =====================================================================
# EEGFeatureDataset
# =====================================================================
class EEGFeatureDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 fs: float = 160,
                 tmin: float = -0.5,
                 tmax: float = 4.0,
                 bands: Optional[Dict[str, Tuple[float, float]]] = None,
                 device: str = None,
                 cache_path: Optional[str] = None,
                 rebuild: bool = False,
                 auto_build: bool = True):

        self.root_dir = root_dir
        self.fs = fs
        self.tmin = tmin
        self.tmax = tmax
        self.bands = bands
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_path = cache_path
        self.rebuild = rebuild

        self.target_len = int((self.tmax - self.tmin) * self.fs) + 1
        self.signals, self.features, self.labels, self.subject_ids = [], [], [], []

        # ================================================================
        # Load from cache or build
        # ================================================================
        if auto_build:
            if cache_path and os.path.exists(cache_path) and not rebuild:
                print(f"Loading dataset from cache: {cache_path}", flush=True)
                cache = torch.load(cache_path, map_location="cpu")
                self.signals = cache["signals"]
                self.features = cache["features"]
                self.labels = cache["labels"]
                self.subject_ids = cache.get("subject_ids", ["unknown"] * len(self.labels))
                print(f"Loaded {len(self.labels)} samples from cache.", flush=True)
            else:
                print("Building dataset from raw EDF files...", flush=True)
                file_paths = glob.glob(os.path.join(root_dir, "**/*.edf"), recursive=True)
                if not file_paths:
                    raise RuntimeError(f"No EDF files found in {root_dir}")

                sigs, feats, labels, subj_ids = self._build_dataset(file_paths)
                self.signals, self.features, self.labels, self.subject_ids = sigs, feats, labels, subj_ids

                if cache_path:
                    torch.save({
                        "signals": self.signals,
                        "features": self.features,
                        "labels": self.labels,
                        "subject_ids": self.subject_ids,
                    }, cache_path)
                    print(f"💾 Dataset cached at {cache_path}", flush=True)
        else:
            print("Skipping auto build — ready for per-subject processing.", flush=True)

    # =================================================================
    # Utility: pad/truncate
    # =================================================================
    def _pad_or_truncate(self, signal: torch.Tensor) -> torch.Tensor:
        cur_len = signal.shape[1]
        if cur_len < self.target_len:
            return F.pad(signal, (0, self.target_len - cur_len))
        elif cur_len > self.target_len:
            return signal[:, :self.target_len]
        return signal

    # =================================================================
    # Process a single EDF file
    # =================================================================
    def _process_file(self, fpath: str) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        basename = os.path.basename(fpath)
        subj_id = basename.split("R")[0]
        run_id = int(basename.split("R")[1][:2])

        raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
        raw = bandpass_filter(raw, 1, min(79, self.fs / 2 - 1))
        raw = notch_filter(raw, 60)

        epochs = create_epochs(raw, tmin=self.tmin, tmax=self.tmax)
        if len(epochs) == 0:
            return [], [], [], []

        data = epochs.get_data().astype("float32", copy=False)
        events = epochs.events[:, -1]
        labels = [map_event_to_action(run_id, e) for e in events]
        data = zscore_normalize(data)

        data_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
        sig_list, feat_list = [], []
        for sig in data_tensor:
            sig = self._pad_or_truncate(sig)
            sig_list.append(sig.cpu())
            f = build_feature_vector(sig.unsqueeze(0), fs=self.fs, bands=self.bands, device=self.device)
            feat_list.append(f.squeeze(0).cpu())

        torch.cuda.empty_cache()
        return sig_list, feat_list, labels, [subj_id] * len(labels)

    # =================================================================
    # Build dataset across all subjects (memory-safe + per-subject caching)
    # =================================================================
    def _build_dataset(self, file_paths: List[str]):
        subjects = sorted(set(os.path.basename(f).split("R")[0] for f in file_paths))

        # ✅ tmp_cache placed beside final cache_path
        cache_dir = os.path.dirname(os.path.abspath(self.cache_path)) if self.cache_path else "."
        tmp_cache_dir = os.path.join(cache_dir, "tmp_cache")
        os.makedirs(tmp_cache_dir, exist_ok=True)

        for subj in subjects:
            print(f"\nProcessing subject: {subj}", flush=True)
            sigs, feats, labels, subj_ids = self.build_from_subject(subj)
            print(f"Built {len(labels)} epochs for {subj}", flush=True)

            # Save intermediate cache per subject
            subj_cache_path = os.path.join(tmp_cache_dir, f"{subj}.pt")
            torch.save({
                "signals": sigs,
                "features": feats,
                "labels": labels,
                "subject_ids": subj_ids
            }, subj_cache_path)
            print(f"[Saved] {subj_cache_path}", flush=True)

            # Free memory after each subject
            del sigs, feats, labels, subj_ids
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Memory usage: {psutil.virtual_memory().percent:.1f}%", flush=True)

        # Merge all subjects into one cache
        print("\nMerging subject-level caches into final dataset...", flush=True)
        merged_sigs, merged_feats, merged_labels, merged_subj_ids = [], [], [], []
        for f in sorted(glob.glob(os.path.join(tmp_cache_dir, "*.pt"))):
            print(f"Loading {f}", flush=True)
            d = torch.load(f, map_location="cpu")
            merged_sigs.extend(d["signals"])
            merged_feats.extend(d["features"])
            merged_labels.extend(d["labels"])
            merged_subj_ids.extend(d["subject_ids"])
            del d
            gc.collect()

        torch.save({
            "signals": merged_sigs,
            "features": merged_feats,
            "labels": merged_labels,
            "subject_ids": merged_subj_ids
        }, self.cache_path)
        print(f"Final dataset cached at {self.cache_path} ({len(merged_labels)} samples)", flush=True)

        import shutil
        shutil.rmtree(tmp_cache_dir, ignore_errors=True)
        print("Temporary caches deleted.", flush=True)

        return merged_sigs, merged_feats, merged_labels, merged_subj_ids

    # =================================================================
    # Dataset API
    # =================================================================
    def build_from_subject(self, subject_id: str):
        subj_dir = os.path.join(self.root_dir, subject_id)
        file_paths = sorted(glob.glob(os.path.join(subj_dir, "*.edf")))
        if not file_paths:
            raise RuntimeError(f"No EDF files found for {subject_id}")

        all_sigs, all_feats, all_labels, all_subj_ids = [], [], [], []
        for fpath in file_paths:
            sigs, feats, labels, subj_ids = self._process_file(fpath)
            if sigs:
                all_sigs.extend(sigs)
                all_feats.extend(feats)
                all_labels.extend(labels)
                all_subj_ids.extend(subj_ids)
        return all_sigs, all_feats, all_labels, all_subj_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig = self.signals[idx]              # keep on CPU
        feats = self.features[idx]           # keep on CPU
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        subj_id = self.subject_ids[idx]
        return sig, feats, label, subj_id
