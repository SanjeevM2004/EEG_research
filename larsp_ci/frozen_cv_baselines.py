#!/usr/bin/env python3
"""Expression-free frozen image baselines for DLPFC LODO evaluation.

Baselines:
1. Frozen DINOv2 ViT-S/14 image encoder + multinomial logistic regression.
2. Frozen DeepLabV3-ResNet50 dense segmentation encoder + multinomial logistic regression.

Only H&E patches and manual labels from training donors are used. The checkpoint is
loaded from the validated image-only dataset builder; no expression matrix or gene
field is accessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

import cv_only_runner as core


class PatchTensorDataset(Dataset):
    def __init__(self, patches: np.ndarray) -> None:
        self.patches = patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> torch.Tensor:
        x = torch.from_numpy(self.patches[index]).permute(2, 0, 1).float() / 255.0
        return x


def batches(patches: np.ndarray, batch_size: int) -> Iterable[torch.Tensor]:
    return DataLoader(PatchTensorDataset(patches), batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=False)


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def load_dinov2(device: torch.device) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def extract_dinov2(model: torch.nn.Module, patches: np.ndarray, batch_size: int,
                    device: torch.device) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for x in batches(patches, batch_size):
        x = F.interpolate(x.to(device), size=(224, 224), mode="bilinear", align_corners=False)
        x = normalize_imagenet(x)
        z = model(x)
        outputs.append(z.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def load_deeplab(device: torch.device) -> torch.nn.Module:
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.backbone


@torch.inference_mode()
def extract_deeplab(backbone: torch.nn.Module, patches: np.ndarray, batch_size: int,
                     device: torch.device) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for x in batches(patches, batch_size):
        x = F.interpolate(x.to(device), size=(224, 224), mode="bilinear", align_corners=False)
        x = normalize_imagenet(x)
        dense = backbone(x)["out"]
        # Dense image-segmentation representation pooled only after the frozen
        # per-pixel feature map has been computed.
        z = F.adaptive_avg_pool2d(dense, output_size=1).flatten(1)
        outputs.append(z.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def fit_probe(x: np.ndarray, y: np.ndarray, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed,
                           solver="lbfgs", multi_class="auto"),
    ).fit(x, y)


def metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    valid = y >= 0
    y = y[valid]
    pred = pred[valid]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(7), zero_division=0)),
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "ordinal_mae": float(np.mean(np.abs(y - pred))),
    }


def plot_map(section: core.Section, pred: np.ndarray, output: Path, method: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    axes[0].imshow(section.image)
    axes[0].set_title("H&E input")
    axes[1].imshow(section.image)
    axes[1].scatter(section.coords_image[:, 0], section.coords_image[:, 1],
                    c=core.PALETTE[np.clip(section.labels, 0, 6)], s=6, linewidths=0)
    axes[1].set_title("Manual labels")
    axes[2].imshow(section.image)
    axes[2].scatter(section.coords_image[:, 0], section.coords_image[:, 1],
                    c=core.PALETTE[pred], s=6, linewidths=0)
    axes[2].set_title(method)
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(fig)


def split_features(sections: List[core.Section], values: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    offset = 0
    for section in sections:
        out[section.sample_id] = values[offset:offset + len(section.labels)]
        offset += len(section.labels)
    return out


def evaluate_model(name: str, features: Dict[str, np.ndarray], sections: List[core.Section],
                   seeds: List[int], output_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for seed in seeds:
        for heldout in core.DONOR_MAP:
            train = [s for s in sections if s.donor != heldout]
            test = [s for s in sections if s.donor == heldout]
            x_train = np.concatenate([features[s.sample_id] for s in train])
            y_train = np.concatenate([s.labels for s in train])
            valid = y_train >= 0
            probe = fit_probe(x_train[valid], y_train[valid], seed)
            for section in test:
                pred = probe.predict(features[section.sample_id]).astype(int)
                row: Dict[str, object] = {
                    "method": name, "seed": seed, "heldout_donor": heldout,
                    "sample_id": section.sample_id,
                }
                row.update(metrics(section.labels, pred))
                rows.append(row)
                for barcode, true, value in zip(section.barcodes, section.labels, pred):
                    prediction_rows.append({
                        "method": name, "seed": seed, "heldout_donor": heldout,
                        "sample_id": section.sample_id, "barcode": str(barcode),
                        "true_label": int(true), "pred_label": int(value),
                    })
                plot_map(section, pred,
                         output_dir / "maps" / f"seed_{seed}" / name / f"{section.sample_id}.png",
                         name)
    pd.DataFrame(prediction_rows).to_csv(output_dir / f"predictions_{name}.csv", index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 73])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sections: List[core.Section] = payload["sections"]
    patches = np.concatenate([s.patches for s in sections])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    audit = {
        "expression_access": False,
        "inputs": ["H&E patches", "training-donor manual labels"],
        "folds": "leave-one-donor-out",
        "seeds": args.seeds,
        "device": str(device),
        "models": ["frozen DINOv2 ViT-S/14", "frozen DeepLabV3-ResNet50 dense encoder"],
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))

    dino = load_dinov2(device)
    dino_features = split_features(sections, extract_dinov2(dino, patches, args.batch_size, device))
    del dino

    deeplab = load_deeplab(device)
    dense_features = split_features(sections, extract_deeplab(deeplab, patches, args.batch_size, device))
    del deeplab

    rows = []
    rows.extend(evaluate_model("Frozen-DINOv2-Linear", dino_features, sections, args.seeds, output_dir))
    rows.extend(evaluate_model("Frozen-DeepLabV3-Dense-Linear", dense_features, sections, args.seeds, output_dir))
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "section_metrics_frozen_baselines.csv", index=False)
    summary = frame.groupby(["method", "seed"], as_index=False)[
        ["accuracy", "macro_f1", "ari", "nmi", "ordinal_mae"]
    ].mean()
    summary.to_csv(output_dir / "seed_summary_frozen_baselines.csv", index=False)
    aggregate = summary.groupby("method").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        ari_mean=("ari", "mean"), ari_std=("ari", "std"),
        nmi_mean=("nmi", "mean"), nmi_std=("nmi", "std"),
    ).reset_index()
    aggregate.to_csv(output_dir / "aggregate_frozen_baselines.csv", index=False)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
