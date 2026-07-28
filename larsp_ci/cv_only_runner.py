#!/usr/bin/env python3
"""LaRSP-CV: image-only DLPFC cortical-layer annotation.

The expression matrix is never read or used. The only model inputs are:
  1) H&E pixels from a patch centred at each Visium spot, and
  2) spot coordinates used to construct the within-section spatial graph.

Training labels from reference donors are used only for supervised contrastive
fine-tuning and construction/evaluation of semantic layer prototypes. Held-out
labels are never used during fitting, alignment, clustering, or prediction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as graph_laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import SpectralClustering
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix, f1_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import ResNet18_Weights, resnet18

LABELS = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]
DONOR_MAP = {
    "Br5292": ["151507", "151508", "151509", "151510"],
    "Br5595": ["151669", "151670", "151671", "151672"],
    "Br8100": ["151673", "151674", "151675", "151676"],
}
SAMPLE_TO_DONOR = {sid: donor for donor, ids in DONOR_MAP.items() for sid in ids}
ZENODO_RECORD = "18852117"
IMAGE_ROOT = "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/images"
DEFAULT_LOWRES_SCALE = 0.0450045
PALETTE = np.asarray([
    [240, 2, 127], [55, 126, 184], [77, 175, 74], [152, 78, 163],
    [255, 215, 0], [255, 127, 0], [26, 26, 26],
], dtype=np.float32) / 255.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_label(value: object) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -1
    text = str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {
        "1": 0, "l1": 0, "layer1": 0,
        "2": 1, "l2": 1, "layer2": 1,
        "3": 2, "l3": 2, "layer3": 2,
        "4": 3, "l4": 3, "layer4": 3,
        "5": 4, "l5": 4, "layer5": 4,
        "6": 5, "l6": 5, "layer6": 5,
        "wm": 6, "whitematter": 6,
    }
    return aliases.get(text, -1)


@dataclass
class Section:
    sample_id: str
    donor: str
    coords_fullres: np.ndarray
    coords_image: np.ndarray
    labels: np.ndarray
    barcodes: np.ndarray
    image: np.ndarray
    patches: np.ndarray
    scale_factor: float


def download_file(url: str, target: Path, min_bytes: int = 128) -> None:
    if target.exists() and target.stat().st_size >= min_bytes:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    session = requests.Session()
    headers = {"User-Agent": "LaRSP-CV/1.0"}
    for attempt in range(8):
        try:
            with session.get(url, stream=True, timeout=(30, 600), headers=headers) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if tmp.stat().st_size < min_bytes:
                raise RuntimeError(f"Downloaded file is too small: {target.name}")
            tmp.replace(target)
            return
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            if attempt == 7:
                raise RuntimeError(f"Unable to download {url}: {exc}") from exc
            time.sleep(min(90, 3 * (2 ** attempt)))


def download_dataset(data_dir: Path) -> None:
    """Download coordinate containers, ground truth, and H&E images.

    H5AD is used only for spot barcodes and obsm['spatial']. adata.X,
    layers, var, genes, and all expression values are deliberately untouched.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    api = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=120)
    api.raise_for_status()
    files = {entry["key"]: entry for entry in api.json()["files"]}
    for sid in SAMPLE_TO_DONOR:
        for name in [f"{sid}_filtered_feature_bc_matrix.h5ad", f"{sid}_truth.txt"]:
            if name not in files:
                raise KeyError(f"Zenodo record missing {name}")
            entry = files[name]
            url = entry.get("links", {}).get("content") or entry.get("links", {}).get("self")
            if not url:
                url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"
            download_file(url, data_dir / name)
        download_file(f"{IMAGE_ROOT}/{sid}_tissue_lowres_image.png", data_dir / f"{sid}_tissue_lowres_image.png")


def read_truth(path: Path) -> pd.Series:
    frame = pd.read_csv(path, sep="\t", header=None, index_col=0)
    return frame.iloc[:, -1]


def extract_patches(image: np.ndarray, coords_xy: np.ndarray, patch_size: int) -> np.ndarray:
    radius = patch_size // 2
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    patches = np.empty((coords_xy.shape[0], patch_size, patch_size, 3), dtype=np.uint8)
    for i, (x, y) in enumerate(coords_xy):
        cx = int(round(float(x))) + radius
        cy = int(round(float(y))) + radius
        x0, y0 = cx - radius, cy - radius
        crop = padded[y0:y0 + patch_size, x0:x0 + patch_size]
        if crop.shape[:2] != (patch_size, patch_size):
            canvas = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
            h, w = crop.shape[:2]
            canvas[:h, :w] = crop
            crop = canvas
        patches[i] = crop
    return patches


def load_sections(data_dir: Path, patch_size: int, image_scale: float) -> List[Section]:
    sections: List[Section] = []
    for sid, donor in SAMPLE_TO_DONOR.items():
        adata = ad.read_h5ad(data_dir / f"{sid}_filtered_feature_bc_matrix.h5ad", backed="r")
        if "spatial" not in adata.obsm:
            raise KeyError(f"{sid}: obsm['spatial'] missing")
        coords_full = np.asarray(adata.obsm["spatial"][:, :2], dtype=np.float64)
        barcodes = adata.obs_names.astype(str).to_numpy()
        try:
            adata.file.close()
        except Exception:
            pass
        truth = read_truth(data_dir / f"{sid}_truth.txt")
        truth.index = truth.index.astype(str)
        labels = np.asarray([canonical_label(v) for v in truth.reindex(barcodes).to_numpy()], dtype=np.int64)
        image = np.asarray(Image.open(data_dir / f"{sid}_tissue_lowres_image.png").convert("RGB"), dtype=np.uint8)
        scale = float(image_scale)
        coords_img = coords_full * scale
        h, w = image.shape[:2]
        if coords_img[:, 0].max(initial=0) > 1.05 * w or coords_img[:, 1].max(initial=0) > 1.05 * h:
            scale = float(min((w - 1) / max(coords_full[:, 0].max(), 1.0),
                              (h - 1) / max(coords_full[:, 1].max(), 1.0)))
            coords_img = coords_full * scale
        patches = extract_patches(image, coords_img, patch_size)
        print(f"[load-cv] {sid} donor={donor} spots={len(barcodes)} labelled={(labels >= 0).sum()} "
              f"image={w}x{h} scale={scale:.7f}; expression_read=False")
        sections.append(Section(sid, donor, coords_full, coords_img, labels, barcodes, image, patches, scale))
    return sections


class PatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray | None = None):
        self.patches = patches
        self.labels = labels

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.patches[index].copy()).permute(2, 0, 1).float() / 255.0
        if self.labels is None:
            return x
        return x, int(self.labels[index])


def batch_augment(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    b = len(out)
    flip_h = torch.rand(b, device=out.device) < 0.5
    flip_v = torch.rand(b, device=out.device) < 0.5
    out[flip_h] = torch.flip(out[flip_h], dims=[3])
    out[flip_v] = torch.flip(out[flip_v], dims=[2])
    rotations = torch.randint(0, 4, (b,), device=out.device)
    for k in (1, 2, 3):
        mask = rotations == k
        if mask.any():
            out[mask] = torch.rot90(out[mask], k=k, dims=[2, 3])
    brightness = 1.0 + 0.20 * torch.randn(b, 1, 1, 1, device=out.device)
    contrast = 1.0 + 0.20 * torch.randn(b, 1, 1, 1, device=out.device)
    mean = out.mean(dim=(2, 3), keepdim=True)
    out = (out - mean) * contrast + mean
    out = out * brightness
    out = out * (1.0 + 0.10 * torch.randn(b, 3, 1, 1, device=out.device))
    if random.random() < 0.5:
        out = F.avg_pool2d(F.pad(out, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
    return (out + 0.025 * torch.randn_like(out)).clamp(0.0, 1.0)


class TinyVisualEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(96), nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisualBYOL(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.online_encoder = TinyVisualEncoder(embedding_dim)
        self.online_projector = MLP(embedding_dim, 128, 64)
        self.predictor = MLP(64, 128, 64)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, momentum: float = 0.99) -> None:
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
        for online, target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)


def negative_cosine(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return 2.0 - 2.0 * (F.normalize(p, dim=1) * F.normalize(z.detach(), dim=1)).sum(dim=1).mean()


def train_byol(patches: np.ndarray, embedding_dim: int, epochs: int, batch_size: int,
               device: torch.device) -> Tuple[VisualBYOL, List[float]]:
    model = VisualBYOL(embedding_dim).to(device)
    loader = DataLoader(PatchDataset(patches), batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=False, drop_last=True)
    optimizer = torch.optim.AdamW(list(model.online_encoder.parameters()) +
                                  list(model.online_projector.parameters()) +
                                  list(model.predictor.parameters()), lr=1e-3, weight_decay=1e-4)
    history: List[float] = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for x in loader:
            x = x.to(device)
            v1, v2 = batch_augment(x), batch_augment(x)
            p1 = model.predictor(model.online_projector(model.online_encoder(v1)))
            p2 = model.predictor(model.online_projector(model.online_encoder(v2)))
            with torch.no_grad():
                z1 = model.target_projector(model.target_encoder(v1))
                z2 = model.target_projector(model.target_encoder(v2))
            loss = 0.5 * (negative_cosine(p1, z2) + negative_cosine(p2, z1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_target(0.99)
            losses.append(float(loss.detach().cpu()))
        value = float(np.mean(losses)); history.append(value)
        print(f"[visual-BYOL] epoch={epoch + 1}/{epochs} loss={value:.5f}")
    return model, history


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.12) -> torch.Tensor:
    z = F.normalize(features, dim=1)
    logits = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(eye, -1e9)
    same = labels[:, None].eq(labels[None, :]) & ~eye
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positives = same.sum(dim=1)
    valid = positives > 0
    if not valid.any():
        return torch.zeros((), device=z.device, requires_grad=True)
    return -(log_prob * same).sum(dim=1)[valid].div(positives[valid]).mean()


def finetune_encoder(model: VisualBYOL, patches: np.ndarray, labels: np.ndarray,
                     epochs: int, batch_size: int, device: torch.device) -> List[float]:
    valid = labels >= 0
    patches, labels = patches[valid], labels[valid]
    counts = np.bincount(labels, minlength=7).astype(np.float64)
    weights = 1.0 / np.maximum(counts[labels], 1.0)
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(labels), replacement=True)
    loader = DataLoader(PatchDataset(patches, labels), batch_size=batch_size, sampler=sampler,
                        num_workers=2, drop_last=True)
    classifier = nn.Linear(model.online_encoder.head.out_features, 7).to(device)
    optimizer = torch.optim.AdamW(list(model.online_encoder.parameters()) + list(classifier.parameters()),
                                  lr=2e-4, weight_decay=1e-4)
    history: List[float] = []
    for epoch in range(epochs):
        model.online_encoder.train(); classifier.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            e1 = model.online_encoder(batch_augment(x))
            e2 = model.online_encoder(batch_augment(x))
            ce = 0.5 * (F.cross_entropy(classifier(e1), y) + F.cross_entropy(classifier(e2), y))
            supcon = supervised_contrastive_loss(torch.cat([e1, e2]), torch.cat([y, y]))
            loss = ce + 0.25 * supcon
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        value = float(np.mean(losses)); history.append(value)
        print(f"[visual-SupCon] epoch={epoch + 1}/{epochs} loss={value:.5f}")
    model.online_encoder.eval()
    for p in model.online_encoder.parameters():
        p.requires_grad_(False)
    return history


@torch.no_grad()
def encode_patches(model: VisualBYOL, patches: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.online_encoder.eval()
    loader = DataLoader(PatchDataset(patches), batch_size=batch_size, shuffle=False, num_workers=2)
    outputs = [model.online_encoder(x.to(device)).cpu().numpy() for x in loader]
    return np.concatenate(outputs).astype(np.float64)


@torch.no_grad()
def resnet18_features(patches: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity(); model.eval().to(device)
    loader = DataLoader(PatchDataset(patches), batch_size=batch_size, shuffle=False, num_workers=2)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    out = []
    for x in loader:
        x = F.interpolate(x.to(device), size=(96, 96), mode="bilinear", align_corners=False)
        out.append(model((x - mean) / std).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def handcrafted_features(patches: np.ndarray) -> np.ndarray:
    features = []
    for patch in patches:
        p = patch.astype(np.float32) / 255.0
        means, stds = p.mean(axis=(0, 1)), p.std(axis=(0, 1))
        q25, q75 = np.quantile(p, 0.25, axis=(0, 1)), np.quantile(p, 0.75, axis=(0, 1))
        gray = np.dot(p[..., :3], np.asarray([0.299, 0.587, 0.114]))
        h = hog(gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
        features.append(np.concatenate([means, stds, q25, q75, h]))
    return np.asarray(features, dtype=np.float64)


def fit_logistic(x: np.ndarray, y: np.ndarray, seed: int) -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200, class_weight="balanced", C=1.0,
                                                               random_state=seed, solver="lbfgs")).fit(x, y)


def classifier_probabilities(model: object, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = np.asarray(model[-1].classes_)
    out = np.full((len(x), 7), 1e-8, dtype=np.float64)
    out[:, classes.astype(int)] = raw
    return out / out.sum(axis=1, keepdims=True)


def build_affinity(embeddings: np.ndarray, coords: np.ndarray, k: int = 12, temperature: float = 0.18) -> csr_matrix:
    n = len(embeddings)
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    scale = float(np.median(distances[:, 1:])) + 1e-8
    z = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    rows, cols, vals = [], [], []
    for i in range(n):
        for d, j in zip(distances[i, 1:], indices[i, 1:]):
            cosine = float(np.clip(z[i] @ z[j], -1.0, 1.0))
            value = max(math.exp((cosine - 1.0) / temperature) * math.exp(-0.5 * (float(d) / scale) ** 2), 1e-8)
            rows.extend([i, int(j)]); cols.extend([int(j), i]); vals.extend([value, value])
    matrix = csr_matrix((vals, (rows, cols)), shape=(n, n)).maximum(csr_matrix((vals, (rows, cols)), shape=(n, n)).T)
    matrix.setdiag(1.0)
    return matrix


def choose_cluster_count(affinity: csr_matrix, minimum: int = 5, maximum: int = 7) -> int:
    n = affinity.shape[0]
    max_eval = min(maximum + 2, n - 1)
    if max_eval <= minimum:
        return min(maximum, max(2, n // 20))
    lap = graph_laplacian(affinity, normed=True)
    try:
        values = np.sort(eigsh(lap, k=max_eval, which="SM", return_eigenvectors=False))
        gaps = [(float(values[k] - values[k - 1]), k) for k in range(minimum, min(maximum, len(values) - 1) + 1)]
        return int(max(gaps)[1])
    except Exception:
        return maximum


def cluster_fiedler_order(clusters: np.ndarray, affinity: csr_matrix) -> np.ndarray:
    k = len(np.unique(clusters))
    block = np.zeros((k, k), dtype=np.float64)
    coo = affinity.tocoo()
    for i, j, value in zip(coo.row, coo.col, coo.data):
        a, b = int(clusters[i]), int(clusters[j])
        if a != b:
            block[a, b] += float(value)
    block = 0.5 * (block + block.T); np.fill_diagonal(block, 0.0)
    if k <= 2 or np.allclose(block, 0):
        return np.arange(k)
    lap = np.diag(block.sum(axis=1)) - block
    vals, vecs = np.linalg.eigh(lap)
    return np.argsort(vecs[:, np.argsort(vals)[1]])


def partition_clusters(embeddings: np.ndarray, coords: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, int, csr_matrix]:
    affinity = build_affinity(embeddings, coords)
    n_clusters = choose_cluster_count(affinity)
    clusters = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans",
                                  random_state=seed, n_init=20).fit_predict(affinity.toarray())
    return clusters.astype(int), cluster_fiedler_order(clusters, affinity).astype(int), int(n_clusters), affinity


def graph_smooth_probabilities(affinity: csr_matrix, probabilities: np.ndarray, alpha: float = 0.65, steps: int = 10) -> np.ndarray:
    transition = sp.diags(1.0 / np.maximum(np.asarray(affinity.sum(axis=1)).ravel(), 1e-12)) @ affinity
    initial, current = probabilities.copy(), probabilities.copy()
    for _ in range(steps):
        current = alpha * transition.dot(current) + (1.0 - alpha) * initial
        current = np.maximum(current, 1e-12); current /= current.sum(axis=1, keepdims=True)
    return current


def majority_refine(labels: np.ndarray, coords: np.ndarray, k: int = 8, steps: int = 2) -> np.ndarray:
    current = labels.copy()
    indices = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords).kneighbors(return_distance=False)
    for _ in range(steps):
        updated = current.copy()
        for i, neigh in enumerate(indices):
            neigh = neigh[neigh != i]
            votes = np.bincount(current[neigh], minlength=7)
            winner = int(np.argmax(votes))
            if votes[winner] > len(neigh) / 2:
                updated[i] = winner
        current = updated
    return current


def regularized_covariance(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        center = x.mean(axis=0, keepdims=True) if len(x) else np.zeros((1, x.shape[1]))
        x = np.vstack([x, center + 1e-4, center - 1e-4])
    cov = LedoitWolf().fit(x).covariance_
    cov = 0.5 * (cov + cov.T)
    floor = max(1e-5, 1e-5 * float(np.trace(cov)) / max(len(cov), 1))
    return cov + floor * np.eye(len(cov))


def sym_power(a: np.ndarray, power: float) -> np.ndarray:
    values, vectors = eigh(0.5 * (a + a.T))
    values = np.clip(values, 1e-10, None)
    return (vectors * (values ** power)) @ vectors.T


def airm_distance(a: np.ndarray, b: np.ndarray) -> float:
    invsqrt = sym_power(a, -0.5)
    values = np.clip(np.linalg.eigvalsh(0.5 * (invsqrt @ b @ invsqrt + (invsqrt @ b @ invsqrt).T)), 1e-10, None)
    return float(np.linalg.norm(np.log(values)))


def riemannian_mean(covariances: Sequence[np.ndarray], max_iter: int = 40, tol: float = 1e-7) -> np.ndarray:
    if not covariances:
        raise ValueError("No covariances supplied")
    mean = np.mean(np.stack(covariances), axis=0)
    mean = 0.5 * (mean + mean.T) + 1e-6 * np.eye(mean.shape[0])
    for _ in range(max_iter):
        sqrt, invsqrt = sym_power(mean, 0.5), sym_power(mean, -0.5)
        tangent = np.zeros_like(mean)
        for cov in covariances:
            whitened = invsqrt @ cov @ invsqrt
            vals, vecs = eigh(0.5 * (whitened + whitened.T))
            tangent += (vecs * np.log(np.clip(vals, 1e-10, None))) @ vecs.T
        tangent /= len(covariances)
        norm = float(np.linalg.norm(tangent, "fro"))
        vals, vecs = eigh(0.5 * (tangent + tangent.T))
        mean = sqrt @ ((vecs * np.exp(vals)) @ vecs.T) @ sqrt
        mean = 0.5 * (mean + mean.T) + 1e-8 * np.eye(mean.shape[0])
        if norm < tol:
            break
    return mean


def align_covariance(covariance: np.ndarray, reference: np.ndarray) -> np.ndarray:
    invsqrt = sym_power(reference, -0.5)
    aligned = invsqrt @ covariance @ invsqrt
    return 0.5 * (aligned + aligned.T) + 1e-8 * np.eye(aligned.shape[0])


def cluster_covariances(z: np.ndarray, clusters: np.ndarray) -> List[np.ndarray]:
    return [regularized_covariance(z[clusters == idx]) for idx in range(int(clusters.max()) + 1)]


def make_layer_prototypes(train_sections: Sequence[Section], embeddings: Mapping[str, np.ndarray],
                          donor_refs: Mapping[str, np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, Dict[str, int]]]:
    by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    counts: Dict[str, Dict[str, int]] = {}
    for section in train_sections:
        z = embeddings[section.sample_id]
        section_counts: Dict[str, int] = {}
        for label in range(7):
            mask = section.labels == label
            section_counts[LABELS[label]] = int(mask.sum())
            if mask.sum() >= 3:
                by_class[label].append(align_covariance(regularized_covariance(z[mask]), donor_refs[section.donor]))
        counts[section.sample_id] = section_counts
    fallback = riemannian_mean([c for values in by_class.values() for c in values])
    return [riemannian_mean(by_class[label]) if by_class[label] else fallback.copy() for label in range(7)], counts


def cost_matrix(covariances: Sequence[np.ndarray], prototypes: Sequence[np.ndarray]) -> np.ndarray:
    return np.asarray([[airm_distance(cov, proto) for proto in prototypes] for cov in covariances])


def costs_to_probabilities(costs: np.ndarray) -> np.ndarray:
    positive = costs[costs > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    logits = -costs / max(scale, 1e-6); logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def monotone_assignment(costs: np.ndarray, order: np.ndarray) -> np.ndarray:
    def solve(block_order: np.ndarray) -> Tuple[np.ndarray, float]:
        k, c = len(block_order), costs.shape[1]
        dp = np.full((k, c), np.inf); back = np.full((k, c), -1, dtype=int)
        dp[0] = costs[block_order[0]]
        for i in range(1, k):
            for label in range(1, c):
                prev = int(np.argmin(dp[i - 1, :label]))
                dp[i, label] = dp[i - 1, prev] + costs[block_order[i], label]
                back[i, label] = prev
        last = int(np.argmin(dp[-1])); mapping = np.zeros(k, dtype=int)
        for i in range(k - 1, -1, -1):
            mapping[block_order[i]] = last
            last = back[i, last] if i > 0 else last
        return mapping, float(np.min(dp[-1]))
    forward, fc = solve(order); reverse, rc = solve(order[::-1])
    return forward if fc <= rc else reverse


def boundary_f1(true: np.ndarray, pred: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    if len(true) < 3:
        return float("nan")
    neigh = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords).kneighbors(return_distance=False)
    true_b = np.zeros(len(true), dtype=bool); pred_b = np.zeros(len(true), dtype=bool)
    for i, ns in enumerate(neigh):
        ns = ns[ns != i]
        true_b[i] = np.any(true[ns] != true[i]); pred_b[i] = np.any(pred[ns] != pred[i])
    tp, fp, fn = np.sum(true_b & pred_b), np.sum(~true_b & pred_b), np.sum(true_b & ~pred_b)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def evaluate(true: np.ndarray, pred: np.ndarray, coords: np.ndarray) -> Dict[str, float]:
    valid = true >= 0; y, p = true[valid], pred[valid]
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=np.arange(7), zero_division=0)),
        "ari": float(adjusted_rand_score(y, p)),
        "nmi": float(normalized_mutual_info_score(y, p)),
        "boundary_f1": boundary_f1(y, p, coords[valid]),
        "ordinal_mae": float(np.mean(np.abs(y - p))),
    }


def plot_section(section: Section, spectral: np.ndarray, pred: np.ndarray, output: Path, method: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    axes[0].imshow(section.image); axes[0].set_title("H&E input")
    axes[1].imshow(section.image); axes[1].scatter(section.coords_image[:, 0], section.coords_image[:, 1], c=PALETTE[np.clip(section.labels, 0, 6)], s=6, linewidths=0)
    axes[1].set_title("Manual labels (evaluation only)")
    axes[2].imshow(section.image); axes[2].scatter(section.coords_image[:, 0], section.coords_image[:, 1], c=spectral, cmap="tab10", s=6, linewidths=0)
    axes[2].set_title("Unsupervised spectral blocks")
    axes[3].imshow(section.image); axes[3].scatter(section.coords_image[:, 0], section.coords_image[:, 1], c=PALETTE[pred], s=6, linewidths=0)
    axes[3].set_title(f"{method} prediction")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=210, bbox_inches="tight"); plt.close(fig)


def split_by_section(sections: Sequence[Section], values: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}; offset = 0
    for section in sections:
        out[section.sample_id] = values[offset:offset + len(section.labels)]
        offset += len(section.labels)
    return out


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_dataset(data_dir)
    sections = load_sections(data_dir, args.patch_size, args.image_scale)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rows: List[Dict[str, object]] = []; diagnostics: Dict[str, object] = {}; histories: Dict[str, object] = {}
    started = time.time()

    for heldout in DONOR_MAP:
        fold_start = time.time()
        train = [s for s in sections if s.donor != heldout]; test = [s for s in sections if s.donor == heldout]
        train_patches = np.concatenate([s.patches for s in train]); train_labels = np.concatenate([s.labels for s in train])
        test_patches = np.concatenate([s.patches for s in test]); valid_train = train_labels >= 0

        print(f"[fold={heldout}] extracting HOG/color baseline")
        handcrafted_train, handcrafted_test = handcrafted_features(train_patches), handcrafted_features(test_patches)
        handcrafted_model = fit_logistic(handcrafted_train[valid_train], train_labels[valid_train], args.seed)
        handcrafted_prob_all = classifier_probabilities(handcrafted_model, handcrafted_test)

        resnet_prob_all = None
        if not args.skip_resnet:
            print(f"[fold={heldout}] extracting frozen ImageNet ResNet18 baseline")
            res_train, res_test = resnet18_features(train_patches, args.batch_size, device), resnet18_features(test_patches, args.batch_size, device)
            resnet_prob_all = classifier_probabilities(fit_logistic(res_train[valid_train], train_labels[valid_train], args.seed), res_test)

        model, byol_history = train_byol(train_patches, args.embedding_dim, args.byol_epochs, args.batch_size, device)
        supcon_history = finetune_encoder(model, train_patches, train_labels, args.finetune_epochs, args.batch_size, device)
        z_train_all, z_test_all = encode_patches(model, train_patches, args.batch_size, device), encode_patches(model, test_patches, args.batch_size, device)
        visual_probe_all = classifier_probabilities(fit_logistic(z_train_all[valid_train], train_labels[valid_train], args.seed), z_test_all)

        from sklearn.decomposition import PCA
        cov_dim = min(args.cov_dim, z_train_all.shape[1], z_train_all.shape[0] - 1)
        cov_pca = PCA(n_components=cov_dim, svd_solver="randomized", random_state=args.seed).fit(z_train_all)
        z_train, z_train_cov = split_by_section(train, z_train_all), split_by_section(train, cov_pca.transform(z_train_all))
        z_test, z_test_cov = split_by_section(test, z_test_all), split_by_section(test, cov_pca.transform(z_test_all))
        handcrafted_prob, visual_probe_prob = split_by_section(test, handcrafted_prob_all), split_by_section(test, visual_probe_all)
        resnet_prob = split_by_section(test, resnet_prob_all) if resnet_prob_all is not None else {}

        train_partitions: Dict[str, Dict[str, object]] = {}; blocks_by_donor: Dict[str, List[np.ndarray]] = defaultdict(list)
        for section in train:
            clusters, order, n_clusters, affinity = partition_clusters(z_train[section.sample_id], section.coords_fullres, args.seed)
            blocks = cluster_covariances(z_train_cov[section.sample_id], clusters)
            blocks_by_donor[section.donor].extend(blocks)
            train_partitions[section.sample_id] = {"clusters": clusters, "order": order, "n_clusters": n_clusters, "affinity": affinity, "blocks": blocks}
        donor_refs = {donor: riemannian_mean(blocks) for donor, blocks in blocks_by_donor.items()}
        prototypes, class_counts = make_layer_prototypes(train, z_train_cov, donor_refs)

        test_partitions: Dict[str, Dict[str, object]] = {}; test_blocks_all: List[np.ndarray] = []
        for section in test:
            clusters, order, n_clusters, affinity = partition_clusters(z_test[section.sample_id], section.coords_fullres, args.seed)
            blocks = cluster_covariances(z_test_cov[section.sample_id], clusters); test_blocks_all.extend(blocks)
            test_partitions[section.sample_id] = {"clusters": clusters, "order": order, "n_clusters": n_clusters, "affinity": affinity, "blocks": blocks}
        test_ref = riemannian_mean(test_blocks_all)

        diagnostics[heldout] = {
            "input_modalities": ["H&E pixels", "spot coordinates"], "gene_expression_used": False,
            "training_donor_block_counts": {d: len(v) for d, v in blocks_by_donor.items()},
            "test_donor_block_count": len(test_blocks_all),
            "training_section_cluster_counts": {sid: int(v["n_clusters"]) for sid, v in train_partitions.items()},
            "test_section_cluster_counts": {sid: int(v["n_clusters"]) for sid, v in test_partitions.items()},
            "training_reference_condition_numbers": {d: float(np.linalg.cond(r)) for d, r in donor_refs.items()},
            "test_reference_condition_number": float(np.linalg.cond(test_ref)),
            "class_spot_counts": class_counts,
            "alignment": "C_RA = R_d^{-1/2} C R_d^{-1/2}; visual class means computed after RA",
        }
        histories[heldout] = {"visual_byol": byol_history, "visual_supcon": supcon_history}

        for section in test:
            part = test_partitions[section.sample_id]
            clusters, order, affinity = np.asarray(part["clusters"]), np.asarray(part["order"]), part["affinity"]
            aligned_blocks = [align_covariance(cov, test_ref) for cov in part["blocks"]]
            costs = cost_matrix(aligned_blocks, prototypes); probs_block = costs_to_probabilities(costs)
            pred_mdm = np.argmin(costs, axis=1).astype(int)[clusters]
            pred_ordered = monotone_assignment(costs, order)[clusters]
            graph_prob = graph_smooth_probabilities(affinity, probs_block[clusters], alpha=0.65, steps=10)
            pred_graph = np.argmax(graph_prob, axis=1).astype(int)
            pred_final = majority_refine(pred_graph, section.coords_fullres, k=args.majority_k, steps=args.majority_steps)
            handcrafted_graph = graph_smooth_probabilities(affinity, handcrafted_prob[section.sample_id], alpha=0.65, steps=10)
            visual_probe_graph = graph_smooth_probabilities(affinity, visual_probe_prob[section.sample_id], alpha=0.65, steps=10)

            predictions: Dict[str, np.ndarray] = {
                "HOG-Color-Linear": np.argmax(handcrafted_prob[section.sample_id], axis=1).astype(int),
                "HOG-Color-Graph": np.argmax(handcrafted_graph, axis=1).astype(int),
                "Visual-BYOL-Linear": np.argmax(visual_probe_prob[section.sample_id], axis=1).astype(int),
                "Visual-BYOL-Graph": np.argmax(visual_probe_graph, axis=1).astype(int),
                "LaRSP-CV-RA-MDM": pred_mdm,
                "LaRSP-CV-RA-OrderedMDM": pred_ordered,
                "LaRSP-CV-RA-Graph": pred_graph,
                "LaRSP-CV-Final": pred_final,
            }
            if resnet_prob_all is not None:
                res_graph = graph_smooth_probabilities(affinity, resnet_prob[section.sample_id], alpha=0.65, steps=10)
                predictions["ImageNet-ResNet18-Linear"] = np.argmax(resnet_prob[section.sample_id], axis=1).astype(int)
                predictions["ImageNet-ResNet18-Graph"] = np.argmax(res_graph, axis=1).astype(int)

            for method, pred in predictions.items():
                rows.append({"seed": args.seed, "heldout_donor": heldout, "sample_id": section.sample_id,
                             "method": method, **evaluate(section.labels, pred, section.coords_fullres)})

            section_dir = output_dir / "sections" / section.sample_id; section_dir.mkdir(parents=True, exist_ok=True)
            plot_section(section, clusters, pred_final, section_dir / f"{section.sample_id}_cv_only_map.png", "LaRSP-CV-Final")
            pd.DataFrame({
                "barcode": section.barcodes, "x_fullres": section.coords_fullres[:, 0], "y_fullres": section.coords_fullres[:, 1],
                "x_image": section.coords_image[:, 0], "y_image": section.coords_image[:, 1], "true": section.labels,
                "spectral_block": clusters, **{f"pred_{name}": pred for name, pred in predictions.items()},
            }).to_csv(section_dir / f"{section.sample_id}_cv_only_predictions.csv", index=False)
            pd.DataFrame(confusion_matrix(section.labels[section.labels >= 0], pred_final[section.labels >= 0], labels=np.arange(7)),
                         index=LABELS, columns=LABELS).to_csv(section_dir / f"{section.sample_id}_cv_only_confusion.csv")

        print(f"[fold={heldout}] finished in {(time.time() - fold_start) / 60:.1f} min")

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"section_metrics_cv_seed_{args.seed}.csv", index=False)
    summary = frame.groupby("method")[["accuracy", "macro_f1", "ari", "nmi", "boundary_f1", "ordinal_mae"]].mean()
    summary.to_csv(output_dir / f"method_summary_cv_seed_{args.seed}.csv")
    frame.groupby(["method", "heldout_donor"])[["accuracy", "macro_f1", "ari", "nmi"]].mean().to_csv(output_dir / f"donor_summary_cv_seed_{args.seed}.csv")
    (output_dir / f"diagnostics_cv_seed_{args.seed}.json").write_text(json.dumps(diagnostics, indent=2))
    (output_dir / f"training_history_cv_seed_{args.seed}.json").write_text(json.dumps(histories, indent=2))
    metadata = {
        "seed": args.seed, "elapsed_seconds": time.time() - started, "device": str(device),
        "input_modalities": ["H&E pixels", "spot coordinates"], "gene_expression_used": False,
        "expression_matrix_accessed": False, "patch_size": args.patch_size,
        "lowres_scale_default": args.image_scale,
        "protocol": "3-fold leave-one-donor-out semantic layer annotation",
        "test_adaptation": "unlabelled held-out donor RA reference from visual spectral blocks",
        "labels": LABELS,
    }
    (output_dir / f"metadata_cv_seed_{args.seed}.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="larsp_ci/data")
    parser.add_argument("--output-dir", default="larsp_ci/results_cv")
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--image-scale", type=float, default=DEFAULT_LOWRES_SCALE)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--cov-dim", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--byol-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--majority-k", type=int, default=8)
    parser.add_argument("--majority-steps", type=int, default=2)
    parser.add_argument("--skip-resnet", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
