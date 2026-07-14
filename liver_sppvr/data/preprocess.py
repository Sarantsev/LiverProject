from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _load_nifti_dhw(path: str) -> Tuple[np.ndarray, tuple]:
    """Load a NIfTI and return the array in (D,H,W) order + zooms (spacing)."""
    import nibabel as nib
    img = nib.load(str(path))
    arr = img.get_fdata().astype(np.float32)      # (H,W,D) -- native NIfTI order
    arr = np.transpose(arr, (2, 0, 1))            # -> (D,H,W)
    return arr, img.header.get_zooms()


# --------- SegVol-style normalization (ForegroundNorm) ---------
def foreground_norm(arr: np.ndarray) -> np.ndarray:
    """Z-score by foreground statistics -- exact replica of SegVol.ForegroundNorm.

    The pretrained SegVol encoder was trained on exactly this normalization (~zero
    mean, unit variance), NOT on a fixed HU window in [0,1]. Threshold = mean over all
    voxels; statistics (0.05/99.95 percentiles, mean, std) are over voxels above it.
    """
    flat = arr.flatten()
    thred = np.mean(flat)
    fg = flat[flat > thred]
    if fg.size == 0:
        fg = flat
    upper = np.percentile(fg, 99.95)
    lower = np.percentile(fg, 0.05)
    mean = float(np.mean(fg))
    std = float(np.std(fg))
    out = np.clip(arr, lower, upper)
    out = (out - mean) / max(std, 1e-8)
    return out.astype(np.float32)


# --------- zoom-in / CropForeground: fractional bbox + crop by it ---------
FracBox = Tuple[float, float, float, float, float, float]  # (d0,h0,w0,d1,h1,w1) in [0,1]


def _bbox_fraction(idx, shape, margin: float, min_size: float) -> "FracBox":
    """Fractional [0,1] bbox per axis from nonzero indices, expanded by margin/min_size."""
    lows, highs = [], []
    for ax in range(3):
        n = shape[ax]
        lo = idx[ax].min() / n
        hi = (idx[ax].max() + 1) / n
        ext = hi - lo
        lo = max(0.0, lo - margin * ext)
        hi = min(1.0, hi + margin * ext)
        if hi - lo < min_size:            # expand to min_size around the center
            c = 0.5 * (lo + hi)
            lo = max(0.0, c - min_size / 2)
            hi = min(1.0, lo + min_size)
            lo = max(0.0, hi - min_size)
        lows.append(lo); highs.append(hi)
    return tuple(lows + highs)  # type: ignore[return-value]


def bbox_fraction_foreground(
    image_path: str,
    margin: float = 0.0,
    min_size: float = 0.1,
) -> "FracBox | None":
    """Fractional bbox of the body (foreground) by intensity -- for CropForeground before resize.

    Foreground = voxels brighter than the mean (as in SegVol). Crops away air/background,
    giving resolution to the body. None if the volume is empty. One box is applied to all
    phases + mask.
    """
    arr, _ = _load_nifti_dhw(image_path)
    idx = np.nonzero(arr > np.mean(arr))
    if idx[0].size == 0:
        return None
    return _bbox_fraction(idx, arr.shape, margin, min_size)


def bbox_fraction_from_mask(
    mask_path: str,
    margin: float = 0.5,      # add margin*extent on each side (context around the tumor)
    min_size: float = 0.1,    # minimum fraction of the volume per axis (guard against tiny crops)
) -> "FracBox | None":
    """Fractional [0,1] bbox of the tumor per axis (D,H,W). None if the mask is empty.

    Coordinates in volume fractions -> the same box applies to phases of different native shapes.
    """
    arr, _ = _load_nifti_dhw(mask_path)
    idx = np.nonzero(arr > 0)
    if idx[0].size == 0:
        return None
    return _bbox_fraction(idx, arr.shape, margin, min_size)


def _crop_fraction(arr: np.ndarray, frac_box: "FracBox") -> np.ndarray:
    """Crop a sub-volume from (D,H,W) by a fractional [0,1] box. Guarantees a non-empty crop."""
    d0f, h0f, w0f, d1f, h1f, w1f = frac_box
    D, H, W = arr.shape
    def rng(f0, f1, n):
        i0 = max(0, min(n - 1, int(round(f0 * n))))
        i1 = max(i0 + 1, min(n, int(round(f1 * n))))
        return i0, i1
    d0, d1 = rng(d0f, d1f, D); h0, h1 = rng(h0f, h1f, H); w0, w1 = rng(w0f, w1f, W)
    return arr[d0:d1, h0:h1, w0:w1]


def load_ct(
    path: str,
    hu_window: Sequence[float] = (-175, 250),
    spatial_size: Sequence[int] = (32, 256, 256),
    frac_box: "FracBox | None" = None,   # crop by a fractional box (tumor/body) before resize
    normalize: str = "hu",               # 'foreground' (SegVol) | 'hu' ([0,1] window)
) -> torch.Tensor:
    """Return a (1, D, H, W) tensor, normalized and resampled.

    Order as in SegVol: normalize on the FULL volume -> crop by frac_box -> resize.
    normalize='foreground' -- SegVol.ForegroundNorm z-score (the encoder was trained on it);
    normalize='hu' -- the older fixed HU window mapped to [0,1].
    If frac_box is given, crop the region (tumor for zoom-in / body for CropForeground) in
    NATIVE resolution, then resample the crop to spatial_size.
    """
    arr, _ = _load_nifti_dhw(path)
    if normalize == "foreground":
        arr = foreground_norm(arr)                 # z-score over the whole volume (SegVol)
    else:  # hu
        lo, hi = float(hu_window[0]), float(hu_window[1])
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-8)
    if frac_box is not None:
        arr = _crop_fraction(arr, frac_box)        # crop AFTER normalization (norm -> crop -> resize)
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()[None, None]  # (1,1,D,H,W)
    t = F.interpolate(t, size=tuple(spatial_size), mode="trilinear", align_corners=False)
    return t[0]                                    # (1,D,H,W)


def load_mask(
    path: str,
    spatial_size: Sequence[int] = (32, 256, 256),
    frac_box: "FracBox | None" = None,
) -> torch.Tensor:
    """Return a binary mask (1, D, H, W), resampled with nearest-neighbor interpolation."""
    arr, _ = _load_nifti_dhw(path)
    arr = (arr > 0).astype(np.float32)
    if frac_box is not None:
        arr = _crop_fraction(arr, frac_box)
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()[None, None]
    t = F.interpolate(t, size=tuple(spatial_size), mode="nearest")
    return t[0]
