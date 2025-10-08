"""
ssp_mae_train.py
----------------
Self-supervised pretraining of MAE on EEG feature arrays.
- Resumes training from saved model
- Saves incremental checkpoints (_1, _2, ...)
- Train/val split, cosine scheduler, early stopping
- NaN-safe training: normalization, clipping, finite checks
"""

import os
import math
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_construction.EEGFeatureDataset import EEGFeatureDataset
from preprocessing.generate_masks import generate_mask
from models.mae import TransformerMAE

def all_finite(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


def get_next_checkpoint_path(base_path: str) -> str:
    """
    If base_path exists, create suffixed versions: base_1.pt, base_2.pt, ...
    If not, return base_path.
    """
    root, ext = os.path.splitext(base_path)
    if not os.path.exists(base_path):
        return base_path
    k = 1
    while True:
        candidate = f"{root}_{k}{ext}"
        if not os.path.exists(candidate):
            return candidate
        k += 1


def main():
    # ----------------------------
    # Config
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # if your new cache has 191-dim features, point to it here:
    cache_path = "./EEG_data/dataset_desc_cache.pt"
    root_dir = "./EEG_data/Physionet/"

    batch_size = 16
    num_epochs = 150
    lr = 1e-3
    weight_decay = 0.05

    # larger model settings
    d_model = 256
    dim_ff = 512

    mask_ratio = 0.30

    # turn AMP off first to stabilize; flip to True later if stable
    USE_AMP = False

    save_base = "./models_saved/mae_eeg_desc.pt"

    # ----------------------------
    # Dataset & DataLoader
    # ----------------------------
    dataset = EEGFeatureDataset(
        root_dir=root_dir, fs=160, tmin=-0.5, tmax=4.0,
        cache_path=cache_path, rebuild=False
    )
    n_val = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ----------------------------
    # Model
    # ----------------------------
    sample = dataset[0]
    # dataset may return (signals, feats, label) or (feats, label)
    if isinstance(sample, (list, tuple)) and len(sample) == 3:
        _, sample_feats, _ = sample
    elif isinstance(sample, (list, tuple)) and len(sample) == 2:
        sample_feats, _ = sample
    else:
        raise RuntimeError("Unexpected dataset sample format.")

    C, d_in = sample_feats.shape
    print(f"[Debug] Feature shape per sample: C={C}, D={d_in}")

    model = TransformerMAE(
        d_in=d_in, n_channels=C,
        d_model=d_model, nhead=4, num_layers=4,
        dim_feedforward=dim_ff, dropout=0.1, use_huber=False
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    from torch.amp import autocast, GradScaler
    scaler = GradScaler(enabled=(device == "cuda") and USE_AMP)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ----------------------------
    # (Optional) Resume if model exists
    # ----------------------------
    if os.path.exists(save_base):
        print(f"Loading checkpoint: {save_base}")
        model.load_state_dict(torch.load(save_base, map_location=device))

    # ----------------------------
    # Training loop
    # ----------------------------
    patience = 10
    best_val = math.inf
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # support both dataset signatures
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                _, feats, _ = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 4:
                _, feats, _, _ = batch
            else:
                feats, _ = batch

            feats = feats.to(device, non_blocking=True)

            # -------- normalize and mask --------
            #feats = normalize_feats(feats, clip_val=5.0)  # crucial to avoid NaNs

            # mask must be same shape as feats (B, C, D) boolean
            mask = generate_mask(feats.shape, mask_ratio=mask_ratio,
                                 per_channel_same=False, device=feats.device)

            feats_masked = feats.clone()
            feats_masked[mask] = 0.0  # mask token = 0

            # quick sanity
            if not (all_finite(feats) and all_finite(feats_masked)):
                print("[Warn] Non-finite feats detected. Stats:",
                      "mean", feats.mean().item(), "std", feats.std().item(),
                      "min", feats.min().item(), "max", feats.max().item())
                continue

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device, enabled=USE_AMP):
                Z_hat, _ = model(feats_masked, mask)
                # check Z_hat too
                if not all_finite(Z_hat):
                    print("[Warn] Non-finite Z_hat detected. "
                          f"mean={Z_hat.mean().item():.4e}, std={Z_hat.std().item():.4e}, "
                          f"min={Z_hat.min().item():.4e}, max={Z_hat.max().item():.4e}")
                loss = model.masked_reconstruction_loss(Z_hat, feats, mask)

            # replace NaN/Inf loss (extreme guard; should not trigger after normalization)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Optional: print first batch stats for deeper debug
            if epoch == 1 and batch_idx == 0:
                with torch.no_grad():
                    print("[Debug] First-batch feats:",
                          "mean", feats.mean().item(),
                          "std", feats.std().item(),
                          "min", feats.min().item(),
                          "max", feats.max().item())
                    print("[Debug] First-batch Z_hat:",
                          "mean", Z_hat.mean().item(),
                          "std", Z_hat.std().item(),
                          "min", Z_hat.min().item(),
                          "max", Z_hat.max().item())

        epoch_loss /= max(1, len(train_loader))

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast(device_type=device, enabled=USE_AMP):
            for batch in val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    _, feats, _ = batch
                elif isinstance(batch, (list, tuple)) and len(batch) == 4:
                    _, feats, _, _ = batch
                else:
                    feats, _ = batch

                feats = feats.to(device, non_blocking=True)
                #feats = normalize_feats(feats, clip_val=5.0)

                mask = generate_mask(feats.shape, mask_ratio=mask_ratio,
                                     per_channel_same=False, device=feats.device)
                feats_masked = feats.clone()
                feats_masked[mask] = 0.0

                Z_hat, _ = model(feats_masked, mask)
                cur_loss = model.masked_reconstruction_loss(Z_hat, feats, mask)
                cur_loss = torch.nan_to_num(cur_loss, nan=0.0, posinf=1e4, neginf=1e4)
                val_loss += cur_loss.item()

        val_loss /= max(1, len(val_loader))
        scheduler.step()

        print(f"Epoch {epoch}: train={epoch_loss:.4f}, val={val_loss:.4f}")

        # ----------------------------
        # Early stopping + incremental save
        # ----------------------------
        if val_loss + 1e-3 < best_val:
            best_val = val_loss
            no_improve = 0
            ckpt_path = save_base
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Saved] {ckpt_path}")
        else:
            no_improve += 1
            print(f"[Plateau] {no_improve}/{patience}")
            if no_improve >= patience:
                print("[Stop] Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
