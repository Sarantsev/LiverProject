"""Device-agnostic device selection.

The code is developed on a machine without a GPU but trained on a remote GPU box,
so the device is chosen automatically and `.cuda()` is never hard-coded anywhere.
"""
from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Return a torch.device from a config string.

    spec: "auto" -> cuda if available, else cpu; or an explicit "cuda"/"cpu".
    """
    spec = (spec or "auto").lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available on this machine.")
    return torch.device(spec)
