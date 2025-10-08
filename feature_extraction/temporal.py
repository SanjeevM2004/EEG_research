import torch

def rms(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Root Mean Square."""
    return torch.sqrt(torch.mean(x ** 2, dim=axis))

def line_length(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Line length (sum of absolute differences)."""
    return torch.sum(torch.abs(x.diff(dim=axis)), dim=axis)

def hjorth_params(x: torch.Tensor, axis: int = -1):
    """
    Hjorth Activity, Mobility, Complexity.
    Works for (N, C, T) or (C, T) if batch dim missing.
    """
    x = x - x.mean(dim=axis, keepdim=True)
    var_x = x.var(dim=axis, unbiased=False)

    dx = x.diff(dim=axis)
    var_dx = dx.var(dim=axis, unbiased=False)
    mobility = torch.sqrt(var_dx / (var_x + 1e-12))

    ddx = dx.diff(dim=axis)
    var_ddx = ddx.var(dim=axis, unbiased=False)
    mobility_dx = torch.sqrt(var_ddx / (var_dx + 1e-12))
    complexity = mobility_dx / (mobility + 1e-12)

    return var_x, mobility, complexity

def skew_torch(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Compute skewness along a dimension (Torch GPU)."""
    mean = x.mean(dim=axis, keepdim=True)
    std = x.std(dim=axis, unbiased=False, keepdim=True)
    z = (x - mean) / (std + 1e-12)
    skew = (z**3).mean(dim=axis)
    return skew

def kurtosis_torch(x: torch.Tensor, axis: int = -1, fisher: bool = True) -> torch.Tensor:
    """Compute kurtosis along a dimension (Torch GPU)."""
    mean = x.mean(dim=axis, keepdim=True)
    std = x.std(dim=axis, unbiased=False, keepdim=True)
    z = (x - mean) / (std + 1e-12)
    kurt = (z**4).mean(dim=axis)
    if fisher:
        kurt = kurt - 3.0  # Fisher definition: normal dist = 0
    return kurt

def temporal_stats(epoch_signals: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Compute temporal stats per channel (GPU-accelerated, batched).

    Parameters
    ----------
    epoch_signals : torch.Tensor, shape (N, C, T) or (C, T)
        EEG signals: N epochs, C channels, T samples.
    device : str
        "cpu" or "cuda".

    Returns
    -------
    feats : torch.Tensor, shape (N, C, d_temp)
        Temporal features per channel:
        [mean, std, skew, kurtosis, RMS, line_length,
         Hjorth_activity, Hjorth_mobility, Hjorth_complexity]
    """
    if epoch_signals.ndim == 2:  # single epoch (C, T)
        epoch_signals = epoch_signals.unsqueeze(0)  # → (1, C, T)

    x = epoch_signals.to(device)  # (N, C, T)

    mean = x.mean(dim=2)  # (N, C)
    std = x.std(dim=2, unbiased=False)  # (N, C)
    sk = skew_torch(x, axis=2)  # (N, C)
    kt = kurtosis_torch(x, axis=2, fisher=True)  # (N, C)
    r = rms(x, axis=2)  # (N, C)
    ll = line_length(x, axis=2)  # (N, C)
    act, mob, comp = hjorth_params(x, axis=2)  # (N, C)

    feats = torch.stack([mean, std, sk, kt, r, ll, act, mob, comp], dim=2)  # (N, C, 9)
    return feats
