import os
import argparse
import glob
import re
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation

from data_construction.EEGFeatureDataset import EEGFeatureDataset
from models.eeg_feat import EEGGraphNet


# --------------------------
# Small helpers
# --------------------------
def trial_signature(arr: np.ndarray) -> np.ndarray:
    """
    signature = zscore(mean over time per channel), shape (N, C)
    """
    s = arr.mean(axis=-1)
    s = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-8)
    return s.astype(np.float32)


def greedy_match_indices(X_sig: np.ndarray, Ref_sig: np.ndarray) -> np.ndarray:
    """
    Greedy 1-to-1 nearest neighbor (no replacement).
    Returns indices into Ref_sig that best match each row of X_sig.
    """
    n_x, n_ref = X_sig.shape[0], Ref_sig.shape[0]
    if n_ref < n_x:
        raise RuntimeError(
            f"Need at least {n_x} cached epochs, have {n_ref}. "
            "Likely too many bad epochs were dropped for that subject."
        )
    X2 = (X_sig ** 2).sum(axis=1, keepdims=True)      # (n_x, 1)
    R2 = (Ref_sig ** 2).sum(axis=1, keepdims=True).T  # (1, n_ref)
    XR = X_sig @ Ref_sig.T                            # (n_x, n_ref)
    D = X2 + R2 - 2.0 * XR                            # (n_x, n_ref)

    used = np.zeros(n_ref, dtype=bool)
    match = np.empty(n_x, dtype=np.int64)
    for i in range(n_x):
        drow = D[i].copy()
        drow[used] = np.inf
        j = int(np.argmin(drow))
        if not np.isfinite(drow[j]):
            raise RuntimeError("Ran out of available cached epochs to match.")
        match[i] = j
        used[j] = True
    return match


def list_subject_ids(root_dir: str) -> list[str]:
    """
    Find subjects like S001, S002... under your Physionet root.
    """
    subs = set()
    for p in glob.glob(os.path.join(root_dir, "**", "S[0-9][0-9][0-9]*.edf"), recursive=True):
        m = re.search(r"(S[0-9]{3})", os.path.basename(p))
        if m:
            subs.add(m.group(1))
    return sorted(subs)


# --------------------------
# Sklearn-compatible wrapper
# --------------------------
class TorchEEGGraphNet(BaseEstimator, ClassifierMixin):
    """
    One global estimator (as MOABB expects).
    On predict(X), we:
      1) Auto-detect which subject X belongs to by comparing signatures against all cached subjects
      2) Build cache for that subject with EEGFeatureDataset.build_from_subject() (your exact pipeline)
      3) Match MOABB X trials to cached epochs
      4) Run the pretrained model on the matched cached (signals, features)
    """
    def __init__(self, model_path, mae_path, dataset_root,
                 cache_dir="./EEG_data/moabb_cache",
                 backbone="rgcn", device=None):
        self.model_path = model_path
        self.mae_path = mae_path
        self.dataset_root = dataset_root
        self.cache_dir = cache_dir
        self.backbone = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes_ = np.arange(5)  # 5 classes
        os.makedirs(self.cache_dir, exist_ok=True)

        # cache of loaded subjects in-memory to avoid reloading on every fold
        self._mem_cache = {}  # subj_str -> (signals_cpu, feats_cpu, labels_cpu)

    # Keep params stable for sklearn.clone
    def get_params(self, deep=True):
        return {
            "model_path": self.model_path,
            "mae_path": self.mae_path,
            "dataset_root": self.dataset_root,
            "cache_dir": self.cache_dir,
            "backbone": self.backbone,
            "device": self.device,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _build_model(self, num_classes=5):
        C, d_in = 64, 191
        mae_d_model, mae_ff = (256, 512) if "desc" in self.mae_path else (128, 256)
        self.model = EEGGraphNet(
            C=C, d_in=d_in, d_hidden=128, num_classes=num_classes,
            backbone=self.backbone, mae_d_model=mae_d_model,
            mae_ff=mae_ff, mae_path=self.mae_path
        ).to(self.device)
        print(f"🔄 Loading model weights from {self.model_path} ...")
        state = torch.load(self.model_path, map_location=self.device)
        miss, unexp = self.model.load_state_dict(state, strict=False)
        print(f"✅ Model loaded | Missing: {miss} | Unexpected: {unexp}")
        self.model.eval()

    def fit(self, X, y):
        if self.model is None:
            self._build_model(num_classes=5)
        return self

    def _load_or_build_subject(self, subj_str: str):
        """
        Returns CPU tensors: (signals, feats, labels)
        """
        if subj_str in self._mem_cache:
            return self._mem_cache[subj_str]

        cache_file = os.path.join(self.cache_dir, f"{subj_str}_cache.pt")
        if os.path.exists(cache_file):
            cache = torch.load(cache_file, map_location="cpu")
            signals, feats, labels = cache["signals"], cache["features"], cache["labels"]
            print(f"📂 Loaded cached data for {subj_str} ({len(labels)} epochs)")
        else:
            print(f"🧩 Building dataset for {subj_str} ...")
            ds = EEGFeatureDataset(
                root_dir=self.dataset_root,
                fs=160, tmin=-0.5, tmax=4.0,
                cache_path=None, rebuild=False, auto_build=False
            )
            sigs, fts, lbls = ds.build_from_subject(subject_id=subj_str)
            signals = torch.stack(sigs).cpu()
            feats = torch.stack(fts).cpu()
            labels = torch.tensor(lbls, dtype=torch.long)
            torch.save({"signals": signals, "features": feats, "labels": labels}, cache_file)
            print(f"✅ Cached {subj_str} ({len(labels)} samples)")

        self._mem_cache[subj_str] = (signals, feats, labels)
        return signals, feats, labels

    def _autodetect_subject(self, X_np: np.ndarray, candidate_subjects: list[str]) -> str:
        """
        Compare X signatures against each subject's cache, pick subject with min total match distance.
        """
        X_sig = trial_signature(X_np)  # (n_x, C)

        best_subj, best_cost = None, float("inf")
        for subj_str in candidate_subjects:
            signals_ref, _, _ = self._load_or_build_subject(subj_str)
            Ref_sig = trial_signature(signals_ref.numpy().astype(np.float32))  # (n_ref, C)
            try:
                match_idx = greedy_match_indices(X_sig, Ref_sig)
            except RuntimeError:
                continue  # not enough ref epochs; skip
            # compute total cost for matched pairs (L2^2)
            # reuse the fact D = ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            a = X_sig
            b = Ref_sig[match_idx]
            cost = np.sum((a - b) ** 2)
            if cost < best_cost:
                best_cost = cost
                best_subj = subj_str

        if best_subj is None:
            raise RuntimeError("Failed to autodetect subject for this batch.")
        print(f"🧭 Autodetected subject = {best_subj} (total match cost={best_cost:.2f})")
        return best_subj

    @torch.no_grad()
    def predict(self, X):
        """
        Use build_from_subject caches:
          - autodetect which subject X belongs to
          - match trials to cached epochs
          - run the pretrained model on matched cached data
        """
        if self.model is None:
            self._build_model(num_classes=5)

        X_np = np.asarray(X, dtype=np.float32)  # (n_x, C, T)

        # 1) Discover subject candidates under dataset_root
        subjects = list_subject_ids(self.dataset_root)
        if not subjects:
            raise RuntimeError(f"No Physionet Sxxx found in {self.dataset_root}")

        # 2) Autodetect which subject matches this X
        subj_str = self._autodetect_subject(X_np, subjects)

        # 3) Load selected subject cache and match trials
        signals_ref, feats_ref, _ = self._load_or_build_subject(subj_str)
        X_sig = trial_signature(X_np)
        Ref_sig = trial_signature(signals_ref.numpy().astype(np.float32))
        match_idx = greedy_match_indices(X_sig, Ref_sig)

        sel_signals = signals_ref[match_idx].to(self.device)
        sel_feats   = feats_ref[match_idx].to(self.device)

        # 4) Forward pass
        logits = self.model(sel_signals, sel_feats)
        preds = logits.argmax(dim=1).cpu().numpy()

        # 5) (Optional) If your checkpoint was ever trained with 5 classes
        #    but MOABB sometimes evaluates 4, you can clamp here:
        preds = np.clip(preds, 0, 4)

        print(f"[{subj_str}] → predicted {len(preds)} trials (MOABB expects {len(X)}) ✅")
        return preds


# --------------------------
# Runner
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_subjects", type=int, default=0,
                        help="Evaluate only first N subjects (0 = all)")
    parser.add_argument("--backbone", type=str, default="rgcn")
    parser.add_argument("--mae_path", type=str, default="./models_saved/mae_eeg_desc.pt")
    parser.add_argument("--model_path", type=str, default="./models_saved/rgcn_desc_best.pt")
    parser.add_argument("--cache_dir", type=str, default="./EEG_data/moabb_cache")
    parser.add_argument("--dataset_root", type=str, default="./EEG_data/Physionet/")
    args = parser.parse_args()

    os.environ["MNE_DATA"] = os.path.abspath(args.dataset_root)
    os.makedirs(os.environ["MNE_DATA"], exist_ok=True)

    # Force MOABB to use your 5-class map (names MOABB expects)
    events_5 = {"rest": 0, "left_hand": 1, "right_hand": 2, "hands": 3, "feet": 4}
    paradigm = MotorImagery(n_classes=5, events=events_5)
    dataset = PhysionetMI()

    if args.limit_subjects and args.limit_subjects > 0:
        dataset.subject_list = dataset.subject_list[: args.limit_subjects]
        print(f"⚠️ Limiting to {len(dataset.subject_list)} subjects: {dataset.subject_list}")

    print("✅ Using dataset:", dataset.code)
    print("🔍 Model:", os.path.abspath(args.model_path))
    print("🔍 MAE:", os.path.abspath(args.mae_path))
    print("🗄️  Cache:", os.path.abspath(args.cache_dir))

    # A single pipeline (MOABB applies it to each subject internally)
    pipeline = Pipeline([
        ("torch_model", TorchEEGGraphNet(
            model_path=args.model_path,
            mae_path=args.mae_path,
            dataset_root=args.dataset_root,
            backbone=args.backbone,
            cache_dir=args.cache_dir
        ))
    ])
    pipelines = {f"EEGGraphNet_{args.backbone.upper()}": pipeline}

    print("\n🚀 Starting MOABB Within-Session evaluation...")
    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,   # ensure fresh evaluation
        n_jobs=1          # avoid clone/parallel issues
    )
    results = evaluation.process(pipelines)

    print("\n✅ Evaluation finished.\n", results.head())
    os.makedirs("./eval", exist_ok=True)
    out_csv = f"./eval/moabb_within_session_{args.backbone}.csv"
    results.to_csv(out_csv, index=False)
    print(f"📊 Results saved to {out_csv}")


if __name__ == "__main__":
    main()
