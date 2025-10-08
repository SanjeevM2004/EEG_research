import torch

def generate_mask(shape, mask_ratio=0.3, per_channel_same=False, device="cpu"):
    """
    Generate a boolean mask tensor.

    Parameters
    ----------
    shape : tuple
        (B, C, d) = batch, channels, feature dims
    mask_ratio : float
        Fraction of features to mask
    per_channel_same : bool
        If True, same mask applied to all channels per sample
    device : str
        Device for mask tensor

    Returns
    -------
    mask : torch.BoolTensor, shape (B, C, d)
        True = masked, False = keep
    """
    B, C, d = shape
    if per_channel_same:
        mask = (torch.rand(B, d, device=device) < mask_ratio).unsqueeze(1).expand(B, C, d)
    else:
        mask = (torch.rand(B, C, d, device=device) < mask_ratio)
    return mask
