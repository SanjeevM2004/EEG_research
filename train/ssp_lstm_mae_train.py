import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
from data_construction.cref_dataset_wrapper import SimpleEEGDataset
from tqdm import tqdm

# ------------------------------------------------------------------
# Gradient Reversal Layer
# ------------------------------------------------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

# ------------------------------------------------------------------
# Subject-Invariant LSTM Autoencoder
# ------------------------------------------------------------------
class SubjectInvariantLSTMAE(nn.Module):
    def __init__(self,
                 n_channels,
                 seq_len,
                 d_emb=64,
                 enc_hidden=128,
                 dec_hidden=128,
                 num_layers=1,
                 n_subjects=15,
                 dann_lambda=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.d_emb = d_emb
        self.dann_lambda = dann_lambda

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_channels,
            hidden_size=enc_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        enc_out_dim = enc_hidden * 2

        self.to_emb = nn.Sequential(
            nn.Linear(enc_out_dim, d_emb),
            nn.ReLU()
        )

        # DANN subject classifier
        self.subj_classifier = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, n_subjects)
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=d_emb,
            hidden_size=dec_hidden,
            num_layers=num_layers,
            batch_first=True
        )
        self.out_proj = nn.Linear(dec_hidden, n_channels)

    def encode(self, x):
        # x: (B, C, T) -> (B, T, C)
        if x.shape[1] == self.n_channels:
            x = x.transpose(1, 2)

        enc_out, _ = self.encoder(x)
        last = enc_out[:, -1, :]
        z = self.to_emb(last)
        return z

    def decode(self, z):
        B = z.size(0)
        z_seq = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder(z_seq)
        x_hat = self.out_proj(dec_out).transpose(1, 2)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        z_rev = grad_reverse(z, self.dann_lambda)
        subj_logits = self.subj_classifier(z_rev)
        x_hat = self.decode(z)
        return x_hat, z, subj_logits

# ------------------------------------------------------------------
# Training Epoch
# ------------------------------------------------------------------
def train_epoch(model, train_dl, optimizer, device):
    model.train()
    tot_loss = tot_rec = tot_adv = 0

    for signals, ra_covs, crefs, labels, subj_ids in train_dl:
        signals = signals.to(device)
        subj_ids = subj_ids.to(device)

        optimizer.zero_grad()

        x_hat, z, subj_logits = model(signals)

        rec_loss = F.mse_loss(x_hat, signals)
        adv_loss = F.cross_entropy(subj_logits, subj_ids)
        loss = rec_loss + adv_loss

        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        tot_rec += rec_loss.item()
        tot_adv += adv_loss.item()

    n = len(train_dl)
    return tot_loss/n, tot_rec/n, tot_adv/n

# ------------------------------------------------------------------
# Validation Epoch
# ------------------------------------------------------------------
def validate_epoch(model, val_dl, device):
    model.eval()
    tot_loss = tot_rec = tot_adv = 0

    with torch.no_grad():
        for signals, ra_covs, crefs, labels, subj_ids in val_dl:
            signals = signals.to(device)
            subj_ids = subj_ids.to(device)

            x_hat, z, subj_logits = model(signals)

            rec_loss = F.mse_loss(x_hat, signals)
            adv_loss = F.cross_entropy(subj_logits, subj_ids)

            loss = rec_loss + adv_loss

            tot_loss += loss.item()
            tot_rec += rec_loss.item()
            tot_adv += adv_loss.item()

    n = len(val_dl)
    return tot_loss/n, tot_rec/n, tot_adv/n

# ------------------------------------------------------------------
# Main Training Function
# ------------------------------------------------------------------
def run_training(cache_path,
                 save_path="./models_saved/lstm_mae.pt",
                 batch_size=32,
                 d_emb=64,
                 dann_lambda=0.3,
                 lr=1e-3,
                 epochs=50):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -----------------------------
    # Load data
    # -----------------------------
    data = torch.load(cache_path)
    dataset = SimpleEEGDataset(
        signals=data["signals"],
        ra_covs=data["ra_covs"],
        cref=data["crefs"],
        labels=data["labels"],
        subject_ids=data["subj"],
    )

    subject_ids = dataset.subject_ids
    labels_arr = np.array(dataset.labels)

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

    print(f"Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

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

    # -----------------------------
    # Model
    # -----------------------------
    sample_signal = dataset.signals[0]
    C, T = sample_signal.shape

    model = SubjectInvariantLSTMAE(
        n_channels=C,
        seq_len=T,
        d_emb=d_emb,
        enc_hidden=128,
        dec_hidden=128,
        num_layers=1,
        n_subjects=len(uniq),
        dann_lambda=dann_lambda,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------------
    # Training
    # -----------------------------
    best_val = float("inf")

    for ep in range(1, epochs+1):
        tr_loss, tr_rec, tr_adv = train_epoch(model, train_dl, optimizer, device)
        va_loss, va_rec, va_adv = validate_epoch(model, val_dl, device)

        print(f"Epoch {ep:03d} | "
              f"Train={tr_loss:.4f} (rec={tr_rec:.4f}, adv={tr_adv:.4f}) | "
              f"Val={va_loss:.4f} (rec={va_rec:.4f}, adv={va_adv:.4f})")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")

    print("\nTraining completed.")
    print(f"Final saved model: {save_path}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    run_training(
        cache_path="./EEG_data/bci_active4_with_cref.pt",
        save_path="./models_saved/lstm_mae.pt",
        batch_size=32,
        d_emb=64,
        dann_lambda=0.3,
        lr=1e-3,
        epochs=50,
    )
