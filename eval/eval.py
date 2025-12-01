#!/usr/bin/env python3
import argparse
import os
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------- Your project imports --------
from models.eeg_feat import EEGGraphNet
from data_construction.EEGFeatureDataset import EEGFeatureDataset
# -------------------------------------


# =========================
# Small helpers (labels)
# =========================
def _auto_class_names(y_true: Sequence[int], y_pred: Sequence[int], num_classes: Optional[int] = None) -> List[str]:
    if num_classes is not None:
        return [str(i) for i in range(num_classes)]
    classes = sorted(set(list(map(int, y_true)) + list(map(int, y_pred))))
    return [str(c) for c in classes]


# =========================
# Confusion matrices (sklearn, blue-white)
# =========================
def plot_confusion_matrix_blues(
    y_true,
    y_pred,
    class_names=None,
    normalize=None,  # None | 'true' | 'pred' | 'all'
    title="Confusion Matrix",
    save_path="confusion_matrix.png",
    dpi=300,
):
    """
    Plots a confusion matrix using sklearn's ConfusionMatrixDisplay
    with a blue-white colormap only.
    """
    # Bigger slide-friendly fonts
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        cmap=plt.cm.Blues,         # blue-white
        normalize=normalize,       # None, 'true', 'pred', or 'all'
        colorbar=True,
        include_values=True
    )
    ax = disp.ax_
    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Make ticks angled for long class names
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


# =========================
# Classification report visuals (matplotlib, blue-white)
# =========================
def plot_classification_report_heatmap(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[List[str]] = None,
    title: str = "Classification Report",
    save_path: str = "classification_report.png",
    dpi: int = 300,
    zero_division: int = 0,
):
    """
    Visualizes precision/recall/F1/support as a blue-white heatmap.
    (Sklearn has no built-in visual; we render with matplotlib.)
    """
    report: Dict[str, Any] = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=zero_division
    )

    keys = ["precision", "recall", "f1-score", "support"]
    rows = []
    row_labels = []

    # Per-class rows
    for cname in class_names:
        rows.append([report[cname][k] for k in keys])
        row_labels.append(cname)

    # Accuracy row
    acc = report.get("accuracy", 0.0)
    total_support = sum(int(report[c]["support"]) for c in class_names)
    rows.append([np.nan, np.nan, acc, total_support])
    row_labels.append("accuracy")

    # Macro & weighted avg rows
    for agg in ["macro avg", "weighted avg"]:
        if agg in report:
            rows.append([report[agg][k] for k in keys])
            row_labels.append(agg)

    rows = np.array(rows, dtype=float)

    # Build a display matrix scaled to [0,1] for colormap, keep annotations separately
    rows_plot = rows.copy()
    for j, k in enumerate(keys):
        if k == "support":
            col = rows[:, j]
            m = np.nanmax(col) if np.isfinite(col).any() else 1.0
            rows_plot[:, j] = np.nan_to_num(col / max(m, 1.0), nan=0.0)
        else:
            rows_plot[:, j] = np.nan_to_num(rows[:, j], nan=0.0)

    fig, ax = plt.subplots(figsize=(10.5, 0.6 + 0.5 * len(row_labels)))
    im = ax.imshow(rows_plot, aspect="auto", cmap=plt.cm.Blues, vmin=0, vmax=1)

    ax.set_title(title, pad=12, fontsize=14)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=12)

    # Light grid for readability
    ax.set_xticks(np.arange(-.5, len(keys), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1]):
            v_plot = rows_plot[i, j]  # 0..1 for color
            color = "white" if v_plot >= 0.5 else "black"
            if np.isnan(rows[i, j]):
                text = ""
            else:
                if keys[j] == "support":
                    text = f"{int(round(rows[i, j]))}"
                else:
                    text = f"{rows[i, j]:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=12, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_f1_bar_blue(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[List[str]] = None,
    title: str = "Per-Class F1 Scores",
    save_path: str = "per_class_f1.png",
    dpi: int = 300,
    zero_division: int = 0,
):
    """
    Per-class F1 bar chart (blue series).
    """
    rep = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=zero_division
    )
    f1s = np.array([rep[c]["f1-score"] for c in class_names], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    bars = ax.bar(range(len(class_names)), f1s, color=plt.cm.Blues(np.linspace(0.4, 0.9, len(class_names))))
    ax.set_title(title, pad=12, fontsize=14)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_ylim(0, 1.05)

    # Annotate values
    for rect, val in zip(bars, f1s):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=11, color="black")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================
# Eval + metrics helpers
# ======================
@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    per_class_correct = Counter()
    per_class_total = Counter()
    all_labels, all_preds = [], []

    # NOTE: dataset yields (signals, feats, labels, extra) per your code
    for signals, feats, labels, _ in tqdm(loader, desc="Evaluating"):
        signals, feats, labels = signals.to(device), feats.to(device), labels.to(device)
        outputs = model(signals, feats)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        for l, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            per_class_total[int(l)] += 1
            if int(l) == int(p):
                per_class_correct[int(l)] += 1

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds .extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    per_class_acc = {c: per_class_correct[c] / per_class_total[c] for c in per_class_total}

    cm = confusion_matrix(all_labels, all_preds)
    cls_report_str = classification_report(all_labels, all_preds, digits=4)

    return avg_loss, acc, per_class_acc, cm, cls_report_str, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="gcn or rgcn")
    parser.add_argument("--weights", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--root_dir", type=str, default="./EEG_data/Physionet/")
    parser.add_argument("--cache_path", type=str, default="./EEG_data/dataset_desc_cache.pt")
    parser.add_argument("--mae_path", type=str, default="./models_saved/mae_eeg_desc.pt")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    parser.add_argument("--class_names", type=str, default="", help="Comma-separated names in label order, e.g., 'rest,left,right,both_fists,both_legs'")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --------------------------
    # Load Dataset
    # --------------------------
    dataset = EEGFeatureDataset(
        root_dir=args.root_dir,
        fs=160,
        tmin=-0.5,
        tmax=4.0,
        cache_path=args.cache_path,
        rebuild=False,
    )

    # dataset returns (signals, feats, label, extra)
    labels = [int(lbl.cpu().item()) if torch.is_tensor(lbl) else int(lbl) for _, _, lbl, _ in dataset]
    num_classes = max(labels) + 1

    # Stratified split for evaluation
    idx = np.arange(len(labels))
    train_idx, test_idx, y_train, y_test = train_test_split(
        idx, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        test_idx, y_test, test_size=0.5, random_state=42, stratify=y_test
    )

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(test_idx),
        shuffle=False
    )

    # --------------------------
    # Build Model
    # --------------------------
    sample_sigs, sample_feats, _, _ = dataset[0]
    C, d_in = sample_feats.shape

    if "desc" in args.mae_path:
        mae_d_model, mae_ff = 256, 512
    else:
        mae_d_model, mae_ff = 128, 256

    if args.model == "gcn":
        model = EEGGraphNet(
            C=C, d_in=d_in, d_hidden=256, num_classes=num_classes,
            backbone="gcn", mae_d_model=mae_d_model, mae_ff=mae_ff,
            mae_path=args.mae_path
        ).to(device)
    elif args.model == "rgcn":
        model = EEGGraphNet(
            C=C, d_in=d_in, d_hidden=128, num_classes=num_classes,
            backbone="rgcn", mae_d_model=mae_d_model, mae_ff=mae_ff,
            mae_path=args.mae_path
        ).to(device)
    else:
        raise ValueError("args.model must be 'gcn' or 'rgcn'")

    # --------------------------
    # Load Checkpoint
    # --------------------------
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    print(f"Loading weights from {args.weights}")
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()

    # --------------------------
    # Evaluate on Test Set
    # --------------------------
    test_loss, test_acc, test_per_class, cm, report_str, y_true, y_pred = evaluate(model, test_loader, device, criterion)
    print(f"\n[Test Set] Loss={test_loss:.4f}, Accuracy={test_acc*100:.2f}%")
    print("Per-class accuracy:", test_per_class)
    print("\nClassification Report:\n", report_str)
    print("Confusion Matrix:\n", cm)

    # --------------------------
    # Visuals + exports
    # --------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Class names
    if args.class_names.strip():
        class_names = [s.strip() for s in args.class_names.split(",")]
        if len(class_names) != (max(y_true) + 1):
            print("[WARN] --class_names length does not match detected classes; continuing with provided names.")
    else:
        class_names = _auto_class_names(y_true, y_pred, num_classes=num_classes)

    # Save text report
    (out_dir / f"classification_report_{args.model}.txt").write_text(report_str, encoding="utf-8")

    # Confusion matrices (blue-white, via sklearn)
    plot_confusion_matrix_blues(
        y_true, y_pred, class_names=class_names,
        normalize=None,
        title="Confusion Matrix (Counts)",
        save_path=str(out_dir / f"cm_counts_{args.model}.png")
    )
    plot_confusion_matrix_blues(
        y_true, y_pred, class_names=class_names,
        normalize="true",
        title="Confusion Matrix (Row-Normalized / Recall)",
        save_path=str(out_dir / f"cm_true_norm_{args.model}.png")
    )
    plot_confusion_matrix_blues(
        y_true, y_pred, class_names=class_names,
        normalize="pred",
        title="Confusion Matrix (Column-Normalized / Precision)",
        save_path=str(out_dir / f"cm_pred_norm_{args.model}.png")
    )

    # Classification report visual + per-class F1 bars (both in blue-white theme)
    plot_classification_report_heatmap(
        y_true, y_pred, class_names=class_names,
        title="Classification Report (Precision / Recall / F1 / Support)",
        save_path=str(out_dir / f"classification_report_{args.model}.png")
    )
    plot_f1_bar_blue(
        y_true, y_pred, class_names=class_names,
        title="Per-Class F1 Scores",
        save_path=str(out_dir / f"per_class_f1_{args.model}.png")
    )

    print(f"\nSaved slide-ready outputs to: {out_dir.resolve()}")
    print(" - cm_counts_*.png")
    print(" - cm_true_norm_*.png")
    print(" - cm_pred_norm_*.png")
    print(" - classification_report_*.png")
    print(" - per_class_f1_*.png")
    print(" - classification_report_*.txt")


if __name__ == "__main__":
    main()
