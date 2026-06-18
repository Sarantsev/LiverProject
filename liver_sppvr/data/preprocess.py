from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _load_nifti_dhw(path: str) -> Tuple[np.ndarray, tuple]:
    """Загрузить NIfTI и вернуть массив в порядке (D,H,W) + zooms (spacing)."""
    import nibabel as nib
    img = nib.load(str(path))
    arr = img.get_fdata().astype(np.float32)      # (H,W,D) — нативный порядок NIfTI
    arr = np.transpose(arr, (2, 0, 1))            # -> (D,H,W)
    return arr, img.header.get_zooms()


def load_ct(
    path: str,
    hu_window: Sequence[float] = (-175, 250),
    spatial_size: Sequence[int] = (32, 256, 256),
) -> torch.Tensor:
    """Вернуть тензор (1, D, H, W), нормализованный в [0,1] и ресэмплированный."""
    arr, _ = _load_nifti_dhw(path)
    lo, hi = float(hu_window[0]), float(hu_window[1])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    t = torch.from_numpy(arr).float()[None, None]  # (1,1,D,H,W)
    t = F.interpolate(t, size=tuple(spatial_size), mode="trilinear", align_corners=False)
    return t[0]                                    # (1,D,H,W)


def load_mask(
    path: str,
    spatial_size: Sequence[int] = (32, 256, 256),
) -> torch.Tensor:
    """Вернуть бинарную маску (1, D, H, W), ресэмплированную ближайшей интерполяцией."""
    arr, _ = _load_nifti_dhw(path)
    arr = (arr > 0).astype(np.float32)
    t = torch.from_numpy(arr).float()[None, None]
    t = F.interpolate(t, size=tuple(spatial_size), mode="nearest")
    return t[0]
