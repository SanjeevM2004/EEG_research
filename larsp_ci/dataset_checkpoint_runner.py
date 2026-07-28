#!/usr/bin/env python3
"""Build a validated image-only DLPFC checkpoint and train only from it.

Only H&E pixels, spot coordinates, spot identifiers, donor/section metadata and
manual layer labels are stored. Gene-expression matrices are never accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image

import cv_only_runner as core


class DLPFCCVDatasetBuilder:
    def __init__(self, data_dir: Path, checkpoint: Path, patch_size: int, image_scale: float):
        self.data_dir = data_dir
        self.checkpoint = checkpoint
        self.patch_size = patch_size
        self.image_scale = image_scale

    @staticmethod
    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _truth_valid(path: Path) -> bool:
        if not path.exists() or path.stat().st_size < 128:
            return False
        raw = path.read_bytes()
        if b"\x00" in raw or raw.count(b"\n") < 100:
            return False
        try:
            s = core.read_truth(path)
        except Exception:
            return False
        return len(s) >= 100 and sum(core.canonical_label(v) >= 0 for v in s.to_numpy()) >= 100

    def _download_atomic(self, url: str, target: Path, expected_md5: str | None, min_bytes: int) -> None:
        def valid(path: Path) -> bool:
            return (
                path.exists()
                and path.stat().st_size >= min_bytes
                and (not expected_md5 or self._md5(path).lower() == expected_md5.lower())
            )

        if valid(target):
            return
        target.unlink(missing_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(8):
            tmp = target.with_suffix(target.suffix + f".part.{os.getpid()}")
            tmp.unlink(missing_ok=True)
            try:
                with requests.get(
                    url, stream=True, timeout=(30, 900), allow_redirects=True,
                    headers={"User-Agent": "LaRSP-CV-dataset-builder/2.1"},
                ) as response:
                    response.raise_for_status()
                    with tmp.open("wb") as f:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            if chunk:
                                f.write(chunk)
                if not valid(tmp):
                    raise RuntimeError(
                        f"download validation failed for {target.name}: "
                        f"size={tmp.stat().st_size if tmp.exists() else 0}, "
                        f"md5={self._md5(tmp) if tmp.exists() else 'missing'}"
                    )
                tmp.replace(target)
                return
            except Exception:
                tmp.unlink(missing_ok=True)
                if attempt == 7:
                    raise
                time.sleep(min(60, 2 ** attempt))

    def _recover_truth_from_h5ad(self, sid: str, h5ad_path: Path, truth_path: Path) -> str:
        """Recover manual labels from spot-level h5ad metadata, never from expression."""
        adata = core.ad.read_h5ad(h5ad_path, backed="r")
        try:
            obs = adata.obs.copy()
            barcodes = adata.obs_names.astype(str)
        finally:
            try:
                adata.file.close()
            except Exception:
                pass

        preferred = [
            "ground_truth", "layer_guess_reordered", "layer_guess", "spatialLIBD",
            "manual_layer", "manual_annotation", "annotation", "layer", "label",
        ]
        candidates = []
        for col in preferred + [str(c) for c in obs.columns if str(c) not in preferred]:
            if col not in obs.columns:
                continue
            values = obs[col].astype(str).to_numpy()
            recognised = int(sum(core.canonical_label(v) >= 0 for v in values))
            candidates.append((recognised, col, values))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if not candidates or candidates[0][0] < 100:
            scored = [(n, c) for n, c, _ in candidates[:20]]
            raise ValueError(
                f"{sid}: no usable manual layer column in h5ad obs; best candidates={scored}"
            )

        recognised, column, values = candidates[0]
        frame = pd.DataFrame({0: barcodes, 1: values})
        frame = frame[[core.canonical_label(v) >= 0 for v in frame[1].to_numpy()]]
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = truth_path.with_suffix(truth_path.suffix + ".recovered")
        frame.to_csv(tmp, sep="\t", header=False, index=False)
        tmp.replace(truth_path)
        if not self._truth_valid(truth_path):
            raise ValueError(f"{sid}: recovered truth file did not validate")
        print(f"[dataset] recovered {truth_path.name} from obs[{column!r}] ({recognised} labels)")
        return f"h5ad_obs:{column}"

    def acquire_sources(self) -> Dict[str, str]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        response = requests.get(f"https://zenodo.org/api/records/{core.ZENODO_RECORD}", timeout=120)
        response.raise_for_status()
        entries = {item["key"]: item for item in response.json()["files"]}
        checksums: Dict[str, str] = {}

        for sid in core.SAMPLE_TO_DONOR:
            h5name = f"{sid}_filtered_feature_bc_matrix.h5ad"
            truth_name = f"{sid}_truth.txt"
            for name in (h5name, truth_name):
                if name not in entries:
                    raise KeyError(f"Zenodo record missing required file: {name}")
                checksum = str(entries[name].get("checksum", ""))
                expected = checksum.split(":", 1)[1] if checksum.lower().startswith("md5:") else None
                url = f"https://zenodo.org/records/{core.ZENODO_RECORD}/files/{name}?download=1"
                self._download_atomic(url, self.data_dir / name, expected, 128)

            truth_path = self.data_dir / truth_name
            provenance = "zenodo_truth"
            if not self._truth_valid(truth_path):
                provenance = self._recover_truth_from_h5ad(
                    sid, self.data_dir / h5name, truth_path
                )
            checksums[h5name] = self._md5(self.data_dir / h5name)
            checksums[truth_name] = self._md5(truth_path)
            checksums[f"{truth_name}:source"] = provenance

            image_name = f"{sid}_{core.IMAGE_SUFFIX}"
            image_path = self.data_dir / image_name
            self._download_atomic(f"{core.IMAGE_ROOT}/{image_name}", image_path, None, 10_000)
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                image_path.unlink(missing_ok=True)
                self._download_atomic(f"{core.IMAGE_ROOT}/{image_name}", image_path, None, 10_000)
                with Image.open(image_path) as image:
                    image.verify()
            checksums[image_name] = self._md5(image_path)
        return checksums

    def validate_sections(self, sections: List[core.Section]) -> None:
        expected = set(core.SAMPLE_TO_DONOR)
        observed = {s.sample_id for s in sections}
        if observed != expected or len(sections) != len(expected):
            raise ValueError(f"Section mismatch: expected={sorted(expected)}, observed={sorted(observed)}")
        keys = set()
        for section in sections:
            n = len(section.barcodes)
            if n < 100:
                raise ValueError(f"{section.sample_id}: implausibly few spots ({n})")
            if section.coords_fullres.shape != (n, 2) or section.coords_image.shape != (n, 2):
                raise ValueError(f"{section.sample_id}: coordinate shape mismatch")
            if section.patches.shape != (n, self.patch_size, self.patch_size, 3):
                raise ValueError(f"{section.sample_id}: patch shape mismatch {section.patches.shape}")
            if section.labels.shape != (n,) or not np.isin(section.labels, np.arange(-1, 7)).all():
                raise ValueError(f"{section.sample_id}: invalid label vector")
            if int((section.labels >= 0).sum()) < 100:
                raise ValueError(f"{section.sample_id}: too few labelled spots")
            if not np.isfinite(section.coords_fullres).all() or not np.isfinite(section.coords_image).all():
                raise ValueError(f"{section.sample_id}: non-finite coordinates")
            for barcode in section.barcodes:
                key = (section.sample_id, str(barcode))
                if key in keys:
                    raise ValueError(f"Duplicate spot key: {key}")
                keys.add(key)

    def build(self) -> None:
        checksums = self.acquire_sources()
        sections = core.load_sections(self.data_dir, self.patch_size, self.image_scale)
        self.validate_sections(sections)
        payload = {
            "format": "larsp-dlpfc-cv-v2.1",
            "labels": list(core.LABELS),
            "label_to_id": {name: i for i, name in enumerate(core.LABELS)},
            "patch_size": self.patch_size,
            "image_scale": self.image_scale,
            "sample_to_donor": dict(core.SAMPLE_TO_DONOR),
            "source_md5": checksums,
            "sections": sections,
        }
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.checkpoint.parent, delete=False) as f:
            tmp = Path(f.name)
        try:
            torch.save(payload, tmp)
            loaded = torch.load(tmp, map_location="cpu", weights_only=False)
            self.validate_sections(loaded["sections"])
            tmp.replace(self.checkpoint)
        finally:
            tmp.unlink(missing_ok=True)
        manifest = {
            "format": payload["format"],
            "checkpoint": str(self.checkpoint),
            "checkpoint_bytes": self.checkpoint.stat().st_size,
            "sections": len(sections),
            "spots": int(sum(len(s.barcodes) for s in sections)),
            "labelled_spots": int(sum((s.labels >= 0).sum() for s in sections)),
            "samples": [s.sample_id for s in sections],
            "donors": sorted({s.donor for s in sections}),
            "source_md5": checksums,
        }
        self.checkpoint.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))


def train_from_checkpoint(args: argparse.Namespace) -> None:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sections = payload["sections"]
    builder = DLPFCCVDatasetBuilder(
        Path(args.data_dir), Path(args.checkpoint), int(payload["patch_size"]),
        float(payload.get("image_scale", 0.0)),
    )
    builder.validate_sections(sections)
    core.download_dataset = lambda _data_dir: None
    core.load_sections = lambda _data_dir, _patch_size, _image_scale: sections
    args.skip_download = True
    core.run(args)


def core_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="larsp_ci/data")
    p.add_argument("--output-dir", default="larsp_ci/results_cv")
    p.add_argument("--patch-size", type=int, default=48)
    p.add_argument("--image-scale", type=float, default=core.DEFAULT_IMAGE_SCALE)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--cov-dim", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--byol-epochs", type=int, default=12)
    p.add_argument("--finetune-epochs", type=int, default=8)
    p.add_argument("--majority-k", type=int, default=8)
    p.add_argument("--majority-steps", type=int, default=2)
    p.add_argument("--skip-resnet", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--data-dir", default="larsp_ci/data")
    build.add_argument("--checkpoint", default="larsp_ci/checkpoints/dlpfc_cv_v2.pt")
    build.add_argument("--patch-size", type=int, default=48)
    build.add_argument("--image-scale", type=float, default=core.DEFAULT_IMAGE_SCALE)
    train = sub.add_parser("train", parents=[core_arg_parser()], add_help=False)
    train.add_argument("--checkpoint", default="larsp_ci/checkpoints/dlpfc_cv_v2.pt")
    args = parser.parse_args()
    if args.command == "build":
        DLPFCCVDatasetBuilder(
            Path(args.data_dir), Path(args.checkpoint), args.patch_size, args.image_scale
        ).build()
    else:
        train_from_checkpoint(args)


if __name__ == "__main__":
    main()
