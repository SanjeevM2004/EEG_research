import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from collections import Counter
import os

from models.eeg import EEGGraphNet
from data_construction.EEGFeatureDataset import EEGFeatureDataset

# -------------------------
# Evaluation utility
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    per_class_correct = Counter()
    per_class_total = Counter()
    all_labels, all_preds = [], []

    for signals, feats, labels in tqdm(loader, desc="Evaluating"):
        signals, feats, labels = (
            signals.to(device),
            feats.to(device),
            labels.to(device)
        )
        outputs = model(signals, feats)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # Per-class stats
        for l, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            per_class_total[l] += 1
            if l == p:
                per_class_correct[l] += 1

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    per_class_acc = {c: per_class_correct[c] / per_class_total[c] for c in per_class_total}

    cm = confusion_matrix(all_labels, all_preds)
    cls_report = classification_report(all_labels, all_preds, digits=4)

    return avg_loss, acc, per_class_acc, cm, cls_report


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="gcn or rgcn")
    parser.add_argument("--weights", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --------------------------
    # Load Dataset
    # --------------------------
    dataset = EEGFeatureDataset(
        root_dir="./EEG_data/Physionet/",
        fs=160,
        tmin=-0.5,
        tmax=4.0,
        cache_path="./EEG_data/dataset_desc_cache.pt",
        rebuild=False,
    )

    labels = [int(lbl.cpu().item()) if torch.is_tensor(lbl) else int(lbl) for _, _, lbl in dataset]
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

    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --------------------------
    # Build Model
    # --------------------------
    sample_sigs, sample_feats, _ = dataset[0]
    C, d_in = sample_feats.shape
    mae_path = "./models_saved/mae_eeg_desc.pt"

    if "desc" in mae_path:
        mae_d_model, mae_ff = 256, 512
    else:
        mae_d_model, mae_ff = 128, 256

    if args.model == "gcn":
        model = EEGGraphNet(
            C=C, d_in=d_in, d_hidden=256, num_classes=num_classes,
            backbone="gcn", mae_d_model=mae_d_model, mae_ff=mae_ff,
            mae_path=mae_path
        ).to(device)
    else:
        model = EEGGraphNet(
            C=C, d_in=d_in, d_hidden=128, num_classes=num_classes,
            backbone="rgcn", mae_d_model=mae_d_model, mae_ff=mae_ff,
            mae_path=mae_path
        ).to(device)

    # --------------------------
    # Load Checkpoint
    # --------------------------
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    print(f"Loading weights from {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device))

    criterion = nn.CrossEntropyLoss()

    # --------------------------
    # Evaluate on Test Set
    # --------------------------
    test_loss, test_acc, test_per_class, cm, report = evaluate(model, test_loader, device, criterion)
    print(f"\n[Test Set] Loss={test_loss:.4f}, Accuracy={test_acc*100:.2f}%")
    print("Per-class accuracy:", test_per_class)
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    # --------------------------
    # Evaluate on Entire Dataset
    # --------------------------
    #full_loss, full_acc, _, _, _ = evaluate(model, full_loader, device, criterion)
    #print(f"\n[Full Dataset] Accuracy={full_acc*100:.2f}%, Loss={full_loss:.4f}")

if __name__ == "__main__":
    main()
