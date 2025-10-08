import torch
from typing import Dict, Tuple

# -------- helpers --------
def _next_fast_len(n: int) -> int:
    """Small utility to pick a cuFFT-friendly length (2/3/5-smooth)."""
    if n <= 1:
        return 1
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0 and m > 1:
                m //= p
        if m == 1:
            return n
        n += 1

# -------- Welch PSD (batched) --------
def welch_psd_batched(
    epoch_signals: torch.Tensor,
    fs: float,
    nperseg: int = 256,
    noverlap: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Welch PSD in Torch.
    Args:
        epoch_signals: (N, C, T)
        fs: sampling rate
        nperseg: segment length
        noverlap: overlap (defaults to nperseg//2)
    Returns:
        freqs: (F,)
        psd:   (N, C, F)
    """
    device = epoch_signals.device
    dtype  = epoch_signals.dtype

    # Handle empty batch early
    if epoch_signals.numel() == 0 or epoch_signals.shape[0] == 0:
        return torch.zeros(1, device=device, dtype=dtype), torch.zeros(
            (epoch_signals.shape[0], epoch_signals.shape[1], 1), device=device, dtype=dtype
        )

    N, C, T = epoch_signals.shape
    nperseg = int(min(max(1, nperseg), T))
    if noverlap is None:
        noverlap = nperseg // 2
    step = max(1, nperseg - noverlap)

    # number of windows (at least 1)
    n_windows = max(1, 1 + (T - nperseg) // step)

    # window & normalization
    win = torch.hamming_window(nperseg, device=device, dtype=dtype)
    win_pow_sum = win.square().sum()

    # choose an FFT size that is cuFFT-friendly
    fft_size = _next_fast_len(nperseg)
    freqs = torch.fft.rfftfreq(fft_size, 1.0 / fs).to(device)

    psd_accum = None
    for i in range(n_windows):
        start = i * step
        seg = epoch_signals[:, :, start : start + nperseg]  # (N, C, L<=nperseg)
        L = seg.shape[-1]
        if L < nperseg:
            seg = torch.nn.functional.pad(seg, (0, nperseg - L))

        seg = seg * win.view(1, 1, -1)  # apply window
        fft_vals = torch.fft.rfft(seg, n=fft_size, dim=-1)  # (N, C, F)
        psd_seg = (fft_vals.abs().square()) / (fs * win_pow_sum)

        if psd_accum is None:
            psd_accum = psd_seg
        else:
            psd_accum = psd_accum + psd_seg

    psd = psd_accum / n_windows  # (N, C, F)
    return freqs, psd


# -------- band power (batched) --------
def band_power_from_psd_batched(
    psd: torch.Tensor,
    freqs: torch.Tensor,
    bands: Dict[str, Tuple[float, float]],
) -> Dict[str, torch.Tensor]:
    """
    Integrate PSD over bands.
    Args:
        psd:   (N, C, F)
        freqs: (F,)
    Returns:
        dict {name: (N, C)}
    """
    device, dtype = psd.device, psd.dtype
    if psd.numel() == 0:
        return {name: torch.zeros((psd.shape[0], psd.shape[1]), device=device, dtype=dtype)
                for name in bands.keys()}

    df = torch.diff(freqs, prepend=freqs[0:1])
    out = {}
    for name, (f1, f2) in bands.items():
        idx = (freqs >= f1) & (freqs <= f2)
        out[name] = (psd[:, :, idx] * df[idx]).sum(dim=2)  # (N, C)
    return out


# -------- main: spectral features (batched) --------
def spectral_stats(
    epoch_signals: torch.Tensor,
    fs: float,
    bands: Dict[str, Tuple[float, float]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute spectral features for batched EEG.
    Args:
        epoch_signals: (N, C, T)
    Returns:
        feats: (N, C, d_spec)
            [abs_band_powers || rel_band_powers || ratios(4) ||
             slope,intercept || spectral_entropy || median_f, edge_f]
    """
    x = epoch_signals.to(device)
    N, C, T = x.shape

    # Early out for empty batch
    nb = len(bands)
    d_spec = 2 * nb + 4 + 2 + 1 + 2  # abs + rel + ratios(4) + (slope,intercept) + entropy + (median,edge)
    if N == 0 or T == 0:
        return torch.zeros((N, C, d_spec), device=device, dtype=x.dtype)

    # PSD via Welch
    freqs, psd = welch_psd_batched(x, fs=fs)  # (F,), (N, C, F)
    F = psd.shape[-1]

    # If PSD ended up empty (unlikely after guards), return zeros
    if F == 0:
        return torch.zeros((N, C, d_spec), device=device, dtype=x.dtype)

    # Band powers (abs & relative)
    bp_dict = band_power_from_psd_batched(psd, freqs, bands)
    abs_bp = torch.stack([bp_dict[name] for name in bands.keys()], dim=2)  # (N, C, nb)

    df = torch.diff(freqs, prepend=freqs[0:1])
    total_power = (psd * df).sum(dim=2, keepdim=True) + 1e-24
    rel_bp = abs_bp / total_power  # (N, C, nb)

    # Ratios (4): alpha/theta, beta/alpha, theta/delta, gamma/beta (fallback to 1s if missing)
    def safe_ratio(a, b): return a / (b + 1e-12)

    def get_band(name: str):
        if name in bp_dict:
            return bp_dict[name]
        return torch.ones((N, C), device=device, dtype=x.dtype)

    delta = get_band("delta")
    theta = get_band("theta")
    alpha = get_band("alpha")
    beta  = get_band("beta")
    gamma = get_band("gamma")

    ratios = torch.stack([
        safe_ratio(alpha, theta),
        safe_ratio(beta, alpha),
        safe_ratio(theta, delta),
        safe_ratio(gamma, beta),
    ], dim=2)  # (N, C, 4)

    # 1/f slope & intercept via batched linear LS on log-log PSD within [1, fs/2]
    idx_fit = (freqs >= 1.0) & (freqs <= fs / 2.0)
    x_log = torch.log10(freqs[idx_fit] + 1e-12)                     # (F')
    X = torch.stack([x_log, torch.ones_like(x_log)], dim=1)         # (F', 2)
    X_pinv = torch.linalg.pinv(X)                                   # (2, F')
    y = torch.log10(psd[:, :, idx_fit] + 1e-24)                     # (N, C, F')
    beta = torch.matmul(y, X_pinv.T)                                # (N, C, 2)
    slope, intercept = beta[..., 0], beta[..., 1]                   # (N, C)
    spec_shape = torch.stack([slope, intercept], dim=2)             # (N, C, 2)

    # Spectral entropy on [1, fs/2] (base 2), normalized by log(K)
    P = torch.clamp(psd[:, :, idx_fit], min=1e-24)                  # (N, C, F')
    Pnorm = P / P.sum(dim=2, keepdim=True)
    logP = torch.log(Pnorm) / torch.log(torch.tensor(2.0, device=device, dtype=x.dtype))
    Hs = -(Pnorm * logP).sum(dim=2)                                 # (N, C)
    K = P.shape[2]
    Hs = Hs / (torch.log(torch.tensor(float(K), device=device, dtype=x.dtype)) /
              torch.log(torch.tensor(2.0, device=device, dtype=x.dtype)))
    Hs = Hs.unsqueeze(-1)                                           # (N, C, 1)

    # Median & edge (90%) frequencies using gather (avoid fancy indexing pitfalls)
    psd_norm = psd / psd.sum(dim=2, keepdim=True).clamp_min(1e-24)
    cumsum = torch.cumsum(psd_norm, dim=2)                          # (N, C, F)
    idx50 = (cumsum >= 0.5).to(psd.dtype).argmax(dim=2)             # (N, C)
    idx90 = (cumsum >= 0.9).to(psd.dtype).argmax(dim=2)             # (N, C)

    freqs_exp = freqs.view(1, 1, F).expand(N, C, F)                 # (N, C, F)
    medfreq  = torch.gather(freqs_exp, 2, idx50.unsqueeze(-1))      # (N, C, 1)
    edgefreq = torch.gather(freqs_exp, 2, idx90.unsqueeze(-1))      # (N, C, 1)

    # Concatenate all
    feats = torch.cat([abs_bp, rel_bp, ratios, spec_shape, Hs, medfreq, edgefreq], dim=2)  # (N, C, d_spec)
    return feats
import numpy as np
from scipy.signal import welch
from typing import Tuple

def welch_psd(
    epoch_signal: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch PSD for a single EEG epoch.
    
    Parameters
    ----------
    epoch_signal : np.ndarray, shape (C, T)
        Single EEG epoch (channels × time).
    fs : float
        Sampling frequency.
    nperseg : int
        Segment length for Welch (default=256).
    noverlap : int
        Overlap between segments. Defaults to nperseg//2.

    Returns
    -------
    freqs : np.ndarray, shape (F,)
        Frequency bins.
    psd   : np.ndarray, shape (C, F)
        Power spectral density for each channel.
    """
    C, T = epoch_signal.shape
    psd_list, freqs = [], None

    for c in range(C):
        freqs, Pxx = welch(epoch_signal[c], fs=fs, nperseg=min(nperseg, T), noverlap=noverlap)
        psd_list.append(Pxx)

    psd = np.stack(psd_list, axis=0)  # (C, F)
    return freqs, psd
