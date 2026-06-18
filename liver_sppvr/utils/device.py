"""Device-agnostic выбор устройства.

Код разрабатывается на машине без GPU, а обучается на удалённом GPU-девайсе,
поэтому устройство выбирается автоматически и нигде не хардкодится `.cuda()`.
"""
from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Вернуть torch.device по строке конфига.

    spec: "auto" -> cuda если доступно, иначе cpu; либо явно "cuda"/"cpu".
    """
    spec = (spec or "auto").lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' запрошен, но CUDA недоступна на этой машине.")
    return torch.device(spec)
