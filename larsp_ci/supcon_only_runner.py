#!/usr/bin/env python3
"""Run LaRSP-CV with a strictly contrastive representation-learning stage.

Protocol:
  1. BYOL self-supervised pretraining.
  2. Supervised contrastive adaptation of the encoder and a temporary
     projection head, using training-donor labels only.
  3. Discard the projection head and freeze the encoder.
  4. Extract embeddings once; downstream classification cannot update it.

No classifier or cross-entropy term exists in stage 2.
"""
from __future__ import annotations

import json
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import cv_only_runner as core
import dataset_checkpoint_runner as checkpoint_runner


class SupConProjectionHead(nn.Module):
    """Temporary projection head used only by the SupCon objective."""

    def __init__(self, input_dim: int, output_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def contrastive_only_adapt_encoder(
    model: core.VisualBYOL,
    patches: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> List[float]:
    """Adapt the BYOL encoder using SupCon only, then freeze it.

    This deliberately replaces the previous CE + 0.25*SupCon objective.
    The temporary projection head is discarded after this function returns.
    """
    valid = labels >= 0
    patches = patches[valid]
    labels = labels[valid]
    if len(labels) == 0:
        raise ValueError("No labelled training spots are available for SupCon adaptation")

    class_counts = np.bincount(labels, minlength=7).astype(np.float64)
    sample_weights = 1.0 / np.maximum(class_counts[labels], 1.0)
    sampler = WeightedRandomSampler(
        sample_weights.tolist(), num_samples=len(labels), replacement=True
    )
    loader = DataLoader(
        core.PatchDataset(patches, labels),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        drop_last=True,
    )

    embedding_dim = int(model.online_encoder.head.out_features)
    projector = SupConProjectionHead(embedding_dim, output_dim=64).to(device)

    # Only the encoder and temporary contrastive projector are trainable.
    # No classifier is instantiated and no cross-entropy is computed.
    for parameter in model.online_encoder.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        list(model.online_encoder.parameters()) + list(projector.parameters()),
        lr=2e-4,
        weight_decay=1e-4,
    )

    audit = {
        "stage": "post_BYOL_representation_adaptation",
        "objective": "supervised_contrastive_loss_only",
        "cross_entropy_used": False,
        "classifier_instantiated": False,
        "encoder_updated_during_supcon": True,
        "encoder_frozen_before_downstream_classification": True,
        "projection_head_discarded_after_adaptation": True,
        "trainable_encoder_parameters": int(
            sum(p.numel() for p in model.online_encoder.parameters())
        ),
        "trainable_projector_parameters": int(sum(p.numel() for p in projector.parameters())),
    }
    print("[SupCon-only audit] " + json.dumps(audit, sort_keys=True))

    history: List[float] = []
    for epoch in range(epochs):
        model.online_encoder.train()
        projector.train()
        losses = []
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            view_one = core.batch_augment(images)
            view_two = core.batch_augment(images)
            embedding_one = model.online_encoder(view_one)
            embedding_two = model.online_encoder(view_two)
            projection_one = projector(embedding_one)
            projection_two = projector(embedding_two)

            loss = core.supervised_contrastive_loss(
                torch.cat([projection_one, projection_two], dim=0),
                torch.cat([targets, targets], dim=0),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite SupCon loss: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if not losses:
            raise RuntimeError("SupCon loader produced no batches")
        value = float(np.mean(losses))
        history.append(value)
        print(
            f"[visual-SupCon-only] epoch={epoch + 1}/{epochs} "
            f"loss={value:.5f} cross_entropy=False"
        )

    # Downstream stages can read embeddings but cannot alter the encoder.
    model.online_encoder.eval()
    for parameter in model.online_encoder.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.online_encoder.parameters()):
        raise AssertionError("Encoder was not fully frozen before downstream evaluation")
    print("[SupCon-only audit] encoder_frozen=True; temporary_projector_discarded=True")
    return history


# dataset_checkpoint_runner imports this same cv_only_runner module object.
# Replacing the function here therefore changes the training path without
# duplicating the dataset/checkpoint implementation.
core.finetune_encoder = contrastive_only_adapt_encoder


if __name__ == "__main__":
    checkpoint_runner.main()
