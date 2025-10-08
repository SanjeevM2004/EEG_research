import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .w_adj import WeightedAdjacency
from adjacency_edges.riemann import riemann_log_euclidean
from adjacency_edges.entropy_mi import mutual_info_adjacency_psd    
from adjacency_edges.spearman import spearman_adjacency_psd
from .mae import TransformerMAE
from .gcn import GCN
from .rgcn import ShallowGraphTransformer


class EEGGraphNet(nn.Module):
    def __init__(self, C: int, d_in: int, d_hidden: int,
                 num_classes: int, backbone: str = "gcn",
                 fs: float = 160,
                 mae_d_model: int = 128, mae_ff: int = 256,
                 num_layers: int = 2, dropout: float = 0.1, mae_path: str = "./models_saved/mae_eeg.pt"):
        """
        Args:
            C: number of EEG channels
            d_in: feature dimension per channel
            d_hidden: hidden dimension for graph model
            num_classes: number of output classes
            backbone: "gcn" or "rgcn" (ShallowGraphTransformer)
            fs: sampling frequency
        """
        super().__init__()
        self.fs = fs
        self.backbone_type = backbone

        # 1. MAE encoder
        self.mae = TransformerMAE(
            d_in=d_in, n_channels=C,
            d_model=mae_d_model, nhead=4, num_layers=4,
            dim_feedforward=mae_ff, dropout=dropout, use_huber=False
        )
        self.mae.load_state_dict(torch.load(mae_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        print("Loaded MAE checkpoint.")
        # 🔒 Freeze all parameters of MAE
        for param in self.mae.parameters():
            param.requires_grad = False
        # 2. Graph model selection
        if backbone == "gcn":
            self.graph_model = GCN(
                d_in=mae_d_model, d_hidden=d_hidden,
                num_layers=num_layers, dropout=dropout
            )
            out_dim = d_hidden
        elif backbone == "rgcn":
            self.graph_model = ShallowGraphTransformer(
                d_in=mae_d_model, d_model=d_hidden,
                d_ff=mae_ff, num_layers=num_layers
            )
            out_dim = d_hidden
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 3. Weighted adjacency fusion
        self.adj_fusion = WeightedAdjacency(num_relations=3)

        # 4. Classification head
        self.classifier = nn.Linear(out_dim, num_classes)

    def compute_adjacencies(self, signals):
        A_mi = mutual_info_adjacency_psd(signals, fs=self.fs)
        A_riem = riemann_log_euclidean(signals, fs=self.fs)
        A_spear = spearman_adjacency_psd(signals, fs=self.fs)
        return [A_mi, A_riem, A_spear], 1  # Riemann = index 1

    def forward(self, signals, feats):

        # Encode features with MAE
        with torch.no_grad():
            H = self.mae.encode(feats)  # (B, C, mae_d_model)

        # Compute fused adjacency
        adj_list, riemann_index = self.compute_adjacencies(signals)
        A = self.adj_fusion(adj_list, riemann_index)

        H = self.graph_model(H, A)   # (B, C, d_hidden)

        # Pool + classify
        H = H.mean(dim=1)  # (B, d_hidden)
        return self.classifier(H)  # (B, num_classes)
