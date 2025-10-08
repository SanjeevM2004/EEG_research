import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter

from adjacency_edges.entropy_mi import mutual_info_adjacency_psd
from adjacency_edges.riemann import riemann_log_euclidean
from adjacency_edges.spearman import spearman_adjacency_psd
from models.eeg import EEGGraphNet
from data_construction.EEGFeatureDataset import EEGFeatureDataset

# -------------------------
# Evaluate on a loader (AMP enabled to save VRAM)
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, criterion, amp_enabled: bool):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    per_class_correct = Counter()
    per_class_total = Counter()

    autocast = torch.cuda.amp.autocast if (amp_enabled and device == "cuda") else torch.autocast
    # torch.autocast for CPU is available in newer torch; if absent, we'll just not use context.

    for batch in loader:
        if len(batch) == 3:
            signals, feats, labels = batch
        else:
            signals, feats, labels = batch[0], batch[1], batch[2]

        signals = signals.to(device, non_blocking=True)
        feats   = feats.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        if amp_enabled and device == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(signals, feats)
                loss = criterion(outputs, labels)
        else:
            outputs = model(signals, feats)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # per-class
        lcpu = labels.detach().cpu().numpy()
        pcpu = preds.detach().cpu().numpy()
        for l, p in zip(lcpu, pcpu):
            per_class_total[l] += 1
            if l == p:
                per_class_correct[l] += 1

        # free per-iter tensors
        del outputs, loss, preds, signals, feats, labels

        # Helps on tight VRAM; if this slows you down, call every N steps instead
        if device == "cuda":
            torch.cuda.empty_cache()

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    per_class_acc = {c: per_class_correct[c] / per_class_total[c] for c in per_class_total}
    return avg_loss, acc, per_class_acc

# -------------------------
# Training Loop with AMP + Gradient Accumulation + cleanup
# -------------------------
def train_model(train_loader, val_loader, test_loader, model, device, optimizer, scheduler,
                criterion, num_epochs, patience, save_path, amp_enabled: bool, accum_steps: int):

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device == "cuda"))

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for step, batch in enumerate(pbar, start=1):
            if len(batch) == 3:
                signals, feats, labels = batch
            else:
                signals, feats, labels = batch[0], batch[1], batch[2]

            signals = signals.to(device, non_blocking=True)
            feats   = feats.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            if amp_enabled and device == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model(signals, feats)
                    loss = criterion(outputs, labels) / accum_steps
            else:
                outputs = model(signals, feats)
                loss = criterion(outputs, labels) / accum_steps

            # backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # track stats on unscaled loss
            batch_loss = loss.item() * accum_steps
            epoch_loss += batch_loss * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # step optimizer every accum_steps
            if step % accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # free per-iter tensors
            del outputs, loss, preds, signals, feats, labels
            if device == "cuda":
                torch.cuda.empty_cache()

        train_loss = epoch_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # ✅ Validation (AMP on to save VRAM)
        val_loss, val_acc, val_per_class = evaluate(model, val_loader, device, criterion, amp_enabled)

        msg = (f"Epoch {epoch}: "
               f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
               f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        print(msg)
        print("Val per-class acc:", val_per_class)

        scheduler.step()

        # Early stopping
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            no_improve = 0
            print("New best val accuracy, saving model...")
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{patience}")
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

        if device == "cuda":
            torch.cuda.empty_cache()

    # ✅ Final Test Eval
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc, test_per_class = evaluate(model, test_loader, device, criterion, amp_enabled)
    print(f" Final Test Loss={test_loss:.4f}, Test Accuracy={test_acc:.4f}")
    print("Test per-class acc:", test_per_class)
    return best_val_acc, test_acc

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gcn", "rgcn"], required=True,
                        help="Choose model: gcn or rgcn")
    parser.add_argument("--batch_size", type=int, default=16, help="micro-batch size per step")
    parser.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--amp", action="store_true", help="enable mixed precision (FP16) on CUDA")
    args = parser.parse_args()

    # Slightly faster matmul on new GPUs; optional
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if device == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Initial VRAM (MB): alloc=", round(torch.cuda.memory_allocated()/1024**2, 1),
              " reserved=", round(torch.cuda.memory_reserved()/1024**2, 1))

    dataset = EEGFeatureDataset(
        root_dir="./EEG_data/Physionet/", fs=160, tmin=-0.5, tmax=4.0,
        cache_path="./EEG_data/dataset_desc_cache.pt", rebuild=False
    )
    mae_path = "./models_saved/mae_eeg_desc.pt"

    # --------------------------
    # Extract labels
    # --------------------------
    labels = [int(lbl.cpu().item()) if torch.is_tensor(lbl) else int(lbl) for _, _, lbl in dataset]
    num_classes = max(labels) + 1

    # --------------------------
    # Stratified train/val/test split (NOTE: trial-level split; for research claims use subject-level)
    # --------------------------
    idx = np.arange(len(labels))
    train_idx, test_idx, y_train, y_test = train_test_split(
        idx, labels, test_size=0.2, random_state=42, stratify=labels)
    val_idx, test_idx, y_val, y_test = train_test_split(
        test_idx, y_test, test_size=0.5, random_state=42, stratify=y_test)

    # --------------------------
    # Weighted Sampler for balancing (train only)
    # --------------------------
    class_counts = np.bincount(y_train)
    class_weights = 1. / np.maximum(class_counts, 1)
    sample_weights = np.array([class_weights[y] for y in y_train], dtype=np.float32)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # --------------------------
    # DataLoaders (pin_memory + non_blocking transfers)
    # --------------------------
    # Important: Train uses the full dataset with a sampler over train_idx weights.
    # Val/Test use SubsetRandomSampler over their indices.
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=False,   # 🔴 disable
        num_workers=0
    )

    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(val_idx),
        pin_memory=False,   # 🔴 disable
        num_workers=0
    )

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(test_idx),
        pin_memory=False,   # 🔴 disable
        num_workers=0
    )


    print("Train size:", len(train_idx), "Val size:", len(val_idx), "Test size:", len(test_idx))

    # --------------------------
    # Model
    # --------------------------
    sample_sigs, sample_feats, _ = dataset[0]
    C, d_in = sample_feats.shape
    if "desc" in mae_path:
        mae_d_model, mae_ff = 256, 512
    else:
        mae_d_model, mae_ff = 128, 256

    if args.model == "gcn":
        model = EEGGraphNet(C=C, d_in=d_in, d_hidden=256,
                            num_classes=num_classes, backbone="gcn",
                            mae_d_model=mae_d_model, mae_ff=mae_ff, mae_path=mae_path).to(device)
    else:
        model = EEGGraphNet(C=C, d_in=d_in, d_hidden=128,
                            num_classes=num_classes, backbone="rgcn",
                            mae_d_model=mae_d_model, mae_ff=mae_ff, mae_path=mae_path).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    # ✅ Balanced Loss
    class_weights_torch = torch.tensor(1.0 / np.maximum(np.bincount(labels), 1), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_torch)

    save_path = f"./models_saved/{args.model}_best.pt"
    import os
    if os.path.exists(save_path):
        print(f"Resuming from checkpoint {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("Starting training from scratch")
        
    best_val_acc, test_acc = train_model(
        train_loader, val_loader, test_loader, model,
        device, optimizer, scheduler, criterion,
        num_epochs=150, patience=10, save_path=save_path,
        amp_enabled=args.amp, accum_steps=max(1, args.accum_steps)
    )

    print(f"Training finished. Best Val Acc={best_val_acc:.4f}, Test Acc={test_acc:.4f}")
    if device == "cuda":
        print("Final VRAM (MB): alloc=", round(torch.cuda.memory_allocated()/1024**2, 1),
              " reserved=", round(torch.cuda.memory_reserved()/1024**2, 1))

if __name__ == "__main__":
    main()
