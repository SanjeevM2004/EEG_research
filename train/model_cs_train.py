import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import numpy as np
from tqdm import tqdm
from collections import Counter
import os
from torch.amp import autocast, GradScaler
from models.eeg_feat import EEGGraphNet
from data_construction.EEGFeatureDataset import EEGFeatureDataset


@torch.no_grad()
def evaluate(model, loader, device, criterion, amp_enabled: bool):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    per_class_correct = Counter()
    per_class_total = Counter()

    for batch in loader:
        if len(batch) == 3:
            signals, feats, labels = batch
        else:
            signals, feats, labels = batch[0], batch[1], batch[2]

        signals = signals.to(device, non_blocking=True)
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if amp_enabled and device == "cuda":
            with autocast(enabled=True):
                outputs = model(signals, feats)
                loss = criterion(outputs, labels)
        else:
            outputs = model(signals, feats)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        lcpu = labels.detach().cpu().numpy()
        pcpu = preds.detach().cpu().numpy()
        for l, p in zip(lcpu, pcpu):
            per_class_total[l] += 1
            if l == p:
                per_class_correct[l] += 1

        del outputs, loss, preds, signals, feats, labels
        if device == "cuda":
            torch.cuda.empty_cache()

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    per_class_acc = {c: per_class_correct[c] / per_class_total[c] for c in per_class_total}
    return avg_loss, acc, per_class_acc


def current_lr(optimizer):
    return [g["lr"] for g in optimizer.param_groups]


def train_model(train_loader, val_loader, test_loader, model, device, optimizer,
                plateau_scheduler, criterion, num_epochs, es_patience,
                save_path, amp_enabled: bool, accum_steps: int, best_val_acc = None):

    scaler = GradScaler(enabled=(amp_enabled and device == "cuda"))
    best_val_acc = 0.0 if best_val_acc is None else best_val_acc
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
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if amp_enabled and device == "cuda":
                with autocast(enabled=True):
                    outputs = model(signals, feats)
                    loss = criterion(outputs, labels) / accum_steps
            else:
                outputs = model(signals, feats)
                loss = criterion(outputs, labels) / accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            batch_loss = loss.item() * accum_steps
            epoch_loss += batch_loss * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

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

            del outputs, loss, preds, signals, feats, labels
            if device == "cuda":
                torch.cuda.empty_cache()

        train_loss = epoch_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation
        val_loss, val_acc, val_per_class = evaluate(model, val_loader, device, criterion, amp_enabled)
        plateau_scheduler.step(val_loss)

        print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        print("Val per-class acc:", val_per_class)
        print("LR now:", [f"{lr:.6g}" for lr in current_lr(optimizer)])

        # Early stopping on val_acc
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            no_improve = 0
            print("New best val accuracy, saving model...")
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{es_patience}")
            if no_improve >= es_patience:
                print("Early stopping triggered.")
                break

        if device == "cuda":
            torch.cuda.empty_cache()

    # Final test
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc, test_per_class = evaluate(model, test_loader, device, criterion, amp_enabled)
    print(f"Final Test Loss={test_loss:.4f}, Test Accuracy={test_acc:.4f}")
    print("Test per-class acc:", test_per_class)
    return best_val_acc, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gcn", "rgcn"], required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--plateau_patience", type=int, default=2)
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--es_patience", type=int, default=10)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = EEGFeatureDataset(
        root_dir="./EEG_data/Physionet/",
        fs=160, tmin=-0.5, tmax=4.0,
        cache_path="./EEG_data/dataset_sub_desc_cache.pt",
        rebuild=False
    )

    mae_path = "./models_saved/mae_eeg_desc.pt"
    labels = [int(lbl.cpu().item()) if torch.is_tensor(lbl) else int(lbl) for _, _, lbl, _ in dataset]
    num_classes = max(labels) + 1

    # Subject splits
    unique_subjects = sorted(set(dataset.subject_ids))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(unique_subjects)

    n_subj = len(unique_subjects)
    n_train = int(0.8 * n_subj)
    n_val = int(0.1 * n_subj)
    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:n_train + n_val]
    test_subjects = unique_subjects[n_train + n_val:]

    train_idx = [i for i, sid in enumerate(dataset.subject_ids) if sid in train_subjects]
    val_idx = [i for i, sid in enumerate(dataset.subject_ids) if sid in val_subjects]
    test_idx = [i for i, sid in enumerate(dataset.subject_ids) if sid in test_subjects]

    # Weighted sampler for balanced training
    train_labels = [labels[i] for i in train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = np.array([class_weights[lbl] for lbl in train_labels], dtype=np.float32)

    # ⚙️ Shrink epoch size: only sample a subset per epoch
    epoch_fraction = 0.25        # 25% of total samples per epoch
    num_samples = int(len(sample_weights) * epoch_fraction)

    sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)


    # DataLoaders
    train_ds, val_ds, test_ds = Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    sample_sigs, sample_feats, _, _ = dataset[0]
    C, d_in = sample_feats.shape
    mae_d_model, mae_ff = (256, 512) if "desc" in mae_path else (128, 256)
    d_hidden = 256 if args.model == "gcn" else 128

    model = EEGGraphNet(C=C, d_in=d_in, d_hidden=d_hidden,
                        num_classes=num_classes, backbone=args.model,
                        mae_d_model=mae_d_model, mae_ff=mae_ff,
                        mae_path=mae_path).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=0.05)

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.plateau_factor,
        patience=args.plateau_patience, threshold=1e-4,
        min_lr=args.min_lr
    )

    class_weights_torch = torch.tensor(1.0 / np.maximum(np.bincount(labels), 1),
                                       dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_torch)

    save_path = f"./models_saved/{args.model}_cs_best.pt"
    val_acc = None
    # ✅ Resume if checkpoint exists
    if os.path.exists(save_path):
        print(f"Checkpoint found: {save_path}. Loading weights...")
        model.load_state_dict(torch.load(save_path, map_location=device))

        # Evaluate current checkpoint before resuming
        print("Evaluating checkpoint before resuming training...")
        val_loss, val_acc, val_per_class = evaluate(model, val_loader, device, criterion, amp_enabled=args.amp)
        print(f"Validation before resume — Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print("Per-class acc:", val_per_class)
    else:
        print("No checkpoint found. Training from scratch...")


    best_val_acc, test_acc = train_model(
        train_loader, val_loader, test_loader, model, device,
        optimizer, plateau_scheduler, criterion,
        num_epochs=150, es_patience=args.es_patience,
        save_path=save_path, amp_enabled=args.amp,
        accum_steps=max(1, args.accum_steps), best_val_acc=val_acc
    )

    print(f"Training finished. Best Val Acc={best_val_acc:.4f}, Test Acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
