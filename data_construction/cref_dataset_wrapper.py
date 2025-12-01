import torch
import torch.utils.data

class SimpleEEGDataset(torch.utils.data.Dataset):
    def __init__(self, signals, ra_covs, cref, labels, subject_ids):
        self.signals = signals
        self.ra_covs = ra_covs
        self.cref = cref
        self.labels = labels
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.signals[idx].float(),      # (C, T)
            self.ra_covs[idx].float(),      # (C, C)
            self.cref[idx].float(),         # (C, C)
            self.labels[idx],
            self.subject_ids[idx],
        )

