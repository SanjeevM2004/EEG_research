import torch

# ---------------- Hjorth Parameters ----------------
def hjorth_params_multi(x: torch.Tensor) -> torch.Tensor:
    """
    Hjorth parameters: Activity, Mobility, Complexity
    x: (B, T)
    Returns: (B, 3)
    """
    x = x - x.mean(dim=1, keepdim=True)

    # Activity
    var_x = x.var(dim=1, unbiased=False)

    # Mobility
    dx = x.diff(dim=1)
    var_dx = dx.var(dim=1, unbiased=False)
    mobility = torch.sqrt(var_dx / (var_x + 1e-12))

    # Complexity
    ddx = dx.diff(dim=1)
    var_ddx = ddx.var(dim=1, unbiased=False)
    mobility_dx = torch.sqrt(var_ddx / (var_dx + 1e-12))
    complexity = mobility_dx / (mobility + 1e-12)

    return torch.stack([var_x, mobility, complexity], dim=1)  # (B, 3)


# ---------------- Zero Crossing Rate ----------------
def zero_crossings(x: torch.Tensor) -> torch.Tensor:
    """
    Zero-crossing rate.
    x: (B, T)
    Returns: (B,)
    """
    signs = torch.sign(x)
    zc = (signs[:, 1:] * signs[:, :-1] < 0).float().sum(dim=1)
    return zc / (x.shape[1] - 1)


# ---------------- Slope Sign Changes ----------------
def slope_sign_changes(x: torch.Tensor) -> torch.Tensor:
    """
    Slope sign changes.
    x: (B, T)
    Returns: (B,)
    """
    dx = x[:, 1:] - x[:, :-1]
    ssc = ((dx[:, 1:] * dx[:, :-1]) < 0).float().sum(dim=1)
    return ssc / (x.shape[1] - 2)


# ---------------- Waveform Length ----------------
def waveform_length(x: torch.Tensor) -> torch.Tensor:
    """
    Waveform length.
    x: (B, T)
    Returns: (B,)
    """
    return torch.sum(torch.abs(x[:, 1:] - x[:, :-1]), dim=1)


# ---------------- Petrosian Fractal Dimension ----------------
def petrosian_fd(x: torch.Tensor) -> torch.Tensor:
    """
    Petrosian Fractal Dimension (fast).
    x: (B, T)
    Returns: (B,)
    """
    N = x.shape[1]
    diff = x[:, 1:] - x[:, :-1]
    Nzc = ((diff[:, 1:] * diff[:, :-1]) < 0).float().sum(dim=1)
    return torch.log2(torch.tensor(N, device=x.device, dtype=x.dtype)) / \
           (torch.log2(torch.tensor(N, device=x.device, dtype=x.dtype)) +
            torch.log2(N / (N + 0.4 * Nzc + 1e-12)))


# ---------------- Wrapper ----------------
def nonlinear_features(x: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """
    Nonlinear/lightweight features: Hjorth + ZCR + SSC + WL + PFD
    x: (N, C, T) or (C, T)
    Returns: (N, C, d_nonlin)
    """
    if x.ndim == 2:  # single epoch (C, T)
        x = x.unsqueeze(0)  # → (1, C, T)

    N, C, T = x.shape
    if N == 0:
        return torch.empty((0, C, 0), device=device)

    x = x.to(device)
    x_flat = x.view(N * C, T)  # (B, T)

    # Compute features
    hjorth = hjorth_params_multi(x_flat)     # (B, 3)
    zc = zero_crossings(x_flat)[:, None]     # (B, 1)
    ssc = slope_sign_changes(x_flat)[:, None]  # (B, 1)
    wl = waveform_length(x_flat)[:, None]    # (B, 1)
    pfd = petrosian_fd(x_flat)[:, None]      # (B, 1)

    feats = torch.cat([hjorth, zc, ssc, wl, pfd], dim=1)  # (B, d)

    return feats.view(N, C, -1)  # (N, C, d_nonlin)
