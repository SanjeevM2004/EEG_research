import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import numpy as np
from tqdm import tqdm

from data_construction.cref_dataset_wrapper import SimpleEEGDataset
from models.neuronet import NeuroGraphNet


# ============================================================
#   EVALUATION  (with X - Xref preprocessing)
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

        subj_cpu = [str(s) for s in subj_ids]
        dom_y = torch.tensor([subj_to_idx[s] for s in subj_cpu], 
                             device=device, dtype=torch.long)

        # -------- X - Xref ----------
        Xref_batch = torch.stack(
            [subj_xref[subj_to_idx[str(s.item())]] for s in subj_ids],
            dim=0
        ).to(device)

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
#   BCI-IV 2a — LOSO TRAINING
# ============================================================
def train_bci_loso(
    cache_path="./EEG_data/bci_active4_with_cref.pt",
    batch_size=8,
    lr=1e-2,
    num_epochs=120,
    test_interval=5,
    es_patience=10,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    data = torch.load(cache_path)

    signals     = data["signals"]
    ra_covs     = data["ra_covs"]
    cref        = data["crefs"]
    labels      = data["labels"]
    subject_ids = [str(s) for s in data["subj"]]  # SAFE strings

    full_dataset = SimpleEEGDataset(
        signals=signals,
        ra_covs=ra_covs,
        cref=cref,
        labels=labels,
        subject_ids=subject_ids,
    )

    labels_arr  = np.array(full_dataset.labels)
    num_classes = len(set(labels_arr))
    unique_subjects = sorted(set(subject_ids))

    print("Subjects found:", unique_subjects)

    # Convert entire signals list into tensor ONCE (for Xref)
    signals_tensor = torch.stack(
        [torch.tensor(s, dtype=torch.float32) for s in signals],
        dim=0
    )   # (N, C, T)
    print("signals_tensor =", signals_tensor.shape)

    all_test_acc = {}

    # ============================================================
    # LOSO LOOP
    # ============================================================
    for test_sub in unique_subjects:

        print("\n===================================================")
        print(f"   LOSO Training – Holding out {test_sub}")
        print("===================================================\n")

        base_idx = np.arange(len(subject_ids))

        train_mask = np.array([s != test_sub for s in subject_ids])
        test_mask  = np.array([s == test_sub for s in subject_ids])

        train_all_idx = base_idx[train_mask]
        test_idx      = base_idx[test_mask]

        # validation split (10% of train)
        rng = np.random.RandomState(42)
        perm = rng.permutation(train_all_idx)
        val_split = max(1, int(0.1 * len(perm)))

        val_idx   = perm[:val_split]
        train_idx = perm[val_split:]

        # -----------------------------
        # Weighted sampler
        # -----------------------------
        train_labels = labels_arr[train_idx]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = np.array([class_weights[l] for l in train_labels],
                                  dtype=np.float32)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True,
        )

        train_dl = DataLoader(Subset(full_dataset, train_idx), 
                              batch_size=batch_size, sampler=sampler)
        val_dl   = DataLoader(Subset(full_dataset, val_idx),
                              batch_size=batch_size, shuffle=False)
        test_dl  = DataLoader(Subset(full_dataset, test_idx),
                              batch_size=batch_size, shuffle=False)

        print(f"Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

        # -----------------------------
        # Domain mapping for fold
        # -----------------------------
        uniq_domains = sorted(set(subject_ids[i] for i in train_idx))
        subj_to_idx = {s: i for i, s in enumerate(uniq_domains)}
        print("Domain mapping:", subj_to_idx)

        # ---------------------------------------------------------
        # Compute Xref for each training subject   (C,T)
        # ---------------------------------------------------------
        print("Computing Xref ...")
        subj_xref = {}
        for s in uniq_domains:
            dom = subj_to_idx[s]
            idxs = [i for i, ss in enumerate(subject_ids) if ss == s]
            Xref = signals_tensor[idxs].mean(dim=0)  # (C,T)
            subj_xref[dom] = Xref
        print("Xref ready.")

        # ---------------------------------------------------------
        # Build model
        # ---------------------------------------------------------
        model = NeuroGraphNet(
            num_classes=num_classes,
            num_domains=len(uniq_domains),
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
        no_improve   = 0
        save_path = f"./neurograph_bci_best_{test_sub}.pt"

        # ============================================================
        # TRAINING
        # ============================================================
        for epoch in range(1, num_epochs + 1):

            model.train()
            total_loss, correct, total = 0.0, 0, 0

            for signals_b, ra_b, cref_b, y_b, subj_b in tqdm(
                train_dl, desc=f"[LOSO {test_sub}] Epoch {epoch}"
            ):
                signals_b = signals_b.to(device)
                ra_b      = ra_b.to(device)
                cref_b    = cref_b.to(device)
                y_b       = y_b.to(device)

                # domain ids
                dom_y = torch.tensor(
                    [subj_to_idx[str(s)] for s in subj_b],
                    device=device, dtype=torch.long
                )

                # -------- X - Xref ----------
                Xref_batch = torch.stack(
                    [subj_xref[subj_to_idx[str(s.item())]] for s in subj_b],
                    dim=0
                ).to(device)
                signals_proc = signals_b - Xref_batch

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, dom_loc, dom_glob = model(signals_proc, ra_b, cref_b)

                    loss_cls = criterion(logits, y_b)
                    loss_dom_l = criterion(dom_loc, dom_y)
                    loss_dom_g = criterion(dom_glob, dom_y)

                    loss = (
                        loss_cls
                        + model.lambda_local * loss_dom_l
                        + model.lambda_global * loss_dom_g
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                preds = logits.argmax(1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)
                total_loss += loss.item() * y_b.size(0)

            train_loss = total_loss / total
            train_acc  = correct / total

            # -----------------------------
            # VALIDATION
            # -----------------------------
            val_loss, val_acc = evaluate(
                model, val_dl, device, criterion, subj_to_idx, subj_xref
            )

            print(f"[{test_sub}] Epoch {epoch} | "
                  f"Train {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val {val_loss:.4f}/{val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                torch.save(model.state_dict(), save_path)
            else:
                no_improve += 1

            if no_improve >= es_patience:
                print(f"[{test_sub}] Early stopping.")
                break

            if epoch % test_interval == 0:
                _, tmp_acc = evaluate(
                    model, test_dl, device, criterion, subj_to_idx, subj_xref
                )
                print(f"[{test_sub}] TEST @ {epoch}: {tmp_acc:.4f}")

        # ============================================================
        # FINAL TEST
        # ============================================================
        print(f"\nLoading best model for {test_sub} ...")
        model.load_state_dict(torch.load(save_path, map_location=device))

        _, final_acc = evaluate(
            model, test_dl, device, criterion, subj_to_idx, subj_xref
        )
        print(f"[{test_sub}] FINAL TEST ACC = {final_acc:.4f}\n")

        all_test_acc[test_sub] = final_acc

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n========== LOSO SUMMARY ==========")
    for s, acc in all_test_acc.items():
        print(f"Subject {s}: {acc:.4f}")

    print("Mean Accuracy =", np.mean(list(all_test_acc.values())))
    print("=================================\n")

    return all_test_acc



if __name__ == "__main__":
    train_bci_loso()
