import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import numpy as np
from tqdm import tqdm

from data_construction.cref_dataset_wrapper import SimpleEEGDataset
from models.neuronet import NeuroGraphNet


# ============================================================
#   EVALUATION (with subject mean-centering)
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, criterion, subj_to_idx, subj_xref):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for signals, ra_covs, cref, labels, subj_ids in loader:

        signals = signals.to(device)
        ra_covs = ra_covs.to(device)
        cref    = cref.to(device)
        labels  = labels.to(device)

        # Domain indices
        dom_y = torch.tensor([subj_to_idx[int(s.item())] for s in subj_ids],
                             dtype=torch.long, device=device)

        # ----------------------
        # Subject mean-centering
        # ----------------------
        Xref_batch = torch.stack(
            [subj_xref[int(s.item())] for s in subj_ids],
            dim=0
        ).to(device)   # (B,C,T)

        signals_proc = signals - Xref_batch

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, _, _ = model(signals_proc, ra_covs, cref)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total



# ============================================================
#     PHYSIONET CROSS-SUBJECT TRAINING (AMP + Scaler)
# ============================================================
def train_physionet(
    cache_path="./EEG_data/combined_active4_with_cref.pt",
    batch_size=16,
    lr=1e-3,
    num_epochs=150,
    test_interval=5,
    es_patience=10,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -----------------------------
    # Load dataset
    # -----------------------------
    data = torch.load(cache_path)
    signals     = data["signals"]
    ra_covs     = data["ra_covs"]
    crefs       = data["crefs"]
    labels      = data["labels"]
    subject_ids = data["subj"]

    dataset = SimpleEEGDataset(
        signals=signals,
        ra_covs=ra_covs,
        cref=crefs,
        labels=labels,
        subject_ids=subject_ids,
    )

    subject_ids = dataset.subject_ids
    labels_arr = np.array(dataset.labels)
    num_classes = len(set(labels_arr))

    # -----------------------------
    # Subject split
    # -----------------------------
    uniq = sorted(set(subject_ids))
    np.random.seed(42)
    np.random.shuffle(uniq)

    n = len(uniq)
    train_sub = uniq[:int(0.6*n)]
    val_sub   = uniq[int(0.6*n):int(0.8*n)]
    test_sub  = uniq[int(0.8*n):]

    train_idx = [i for i,s in enumerate(subject_ids) if s in train_sub]
    val_idx   = [i for i,s in enumerate(subject_ids) if s in val_sub]
    test_idx  = [i for i,s in enumerate(subject_ids) if s in test_sub]

    # -----------------------------
    # Weighted sampler
    # -----------------------------
    train_labels = labels_arr[train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = np.array([class_weights[l] for l in train_labels], dtype=np.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

    train_dl = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, sampler=sampler)
    val_dl   = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    print(f"Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # -----------------------------
    # Domain mapping
    # -----------------------------
    subj_to_idx = {s: i for i,s in enumerate(uniq)}
    num_domains = len(uniq)

    # -------------------------------------------------
    # Precompute subject-level Euclidean mean signals
    # -------------------------------------------------
    print("Computing subject-level mean signals (Xref)...")

    signals_tensor = torch.as_tensor(signals, dtype=torch.float32)  # (N,C,T)
    subj_xref = {}  # domain_idx -> (C,T) tensor

    for subj in uniq:
        dom = subj_to_idx[subj]
        idxs = [i for i,s in enumerate(subject_ids) if s == subj]
        subj_sigs = signals_tensor[idxs]       # (#trials,C,T)
        Xref = subj_sigs.mean(dim=0)           # (C,T)
        subj_xref[dom] = Xref

    print("Xref computed for all subjects.")

    # -----------------------------
    # Model + optimizer
    # -----------------------------
    model = NeuroGraphNet(
        num_classes=num_classes,
        num_domains=num_domains,
        lstm_hidden=64,
        gcn_hidden=128,
        gcn_layers=2,
        global_hidden=128,
        lambda_local=0.1,
        lambda_global=0.1,
        whiten_signals=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    no_improve = 0

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    for epoch in range(1, num_epochs + 1):

        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for signals, ra_covs, cref, labels, subj_ids in tqdm(train_dl, desc=f"Epoch {epoch}"):

            signals = signals.to(device)
            ra_covs = ra_covs.to(device)
            cref    = cref.to(device)
            labels  = labels.to(device)

            # Domain indices (int → domain_idx)
            dom_y = torch.tensor([subj_to_idx[int(s.item())] for s in subj_ids],
                                 dtype=torch.long, device=device)

            # ----------------------------------
            # SUBJECT MEAN-CENTERING HERE
            # ----------------------------------
            Xref_batch = torch.stack(
                [subj_xref[int(s.item())] for s in subj_ids],
                dim=0
            ).to(device)

            signals_proc = signals - Xref_batch

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, dom_loc, dom_glob = model(signals_proc, ra_covs, cref)

                loss_cls        = criterion(logits, labels)
                loss_dom_local  = criterion(dom_loc, dom_y)
                loss_dom_global = criterion(dom_glob, dom_y)

                loss = (
                    loss_cls
                    + model.lambda_local * loss_dom_local
                    + model.lambda_global * loss_dom_global
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # -----------------------------
        # VALIDATION
        # -----------------------------
        val_loss, val_acc = evaluate(model, val_dl, device, criterion, subj_to_idx, subj_xref)

        print(f"Epoch {epoch} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), "./neurograph_physio_best.pt")
        else:
            no_improve += 1

        if no_improve >= es_patience:
            print("Early stopping triggered.")
            break

        # -----------------------------
        # TEST every 5 epochs
        # -----------------------------
        if epoch % test_interval == 0:
            _, test_acc = evaluate(model, test_dl, device, criterion, subj_to_idx, subj_xref)
            print(f"[TEST @ epoch {epoch}] Acc = {test_acc:.4f}")

    # ============================================================
    # FINAL TEST
    # ============================================================
    print("\nLoading BEST model...")
    model.load_state_dict(torch.load("./neurograph_physio_best.pt", map_location=device))

    final_loss, final_acc = evaluate(model, test_dl, device, criterion, subj_to_idx, subj_xref)
    print("\n========== FINAL PHYSIONET TEST ACCURACY ==========")
    print(f"Test Accuracy = {final_acc:.4f}")
    print("===================================================")

    return final_acc



if __name__ == "__main__":
    train_physionet()
