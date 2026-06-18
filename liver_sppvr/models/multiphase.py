"""Fusion мультифазного КТ (новизна).

SegVol нативно принимает одноканальный объём (одна фаза). Мы добавляем fusion
4 фаз (нативная / артериальная / портальная / отсроченная), т.к. динамика
контраста — радиологическая основа различения HCC / ICC / метастазов.

Поддерживаются два режима:

1. "concat_stem" — ранний fusion на уровне вокселей: фазы как каналы пропускаются
   через небольшой 3D-conv stem и сворачиваются в 1 канал, который подаётся в
   неизменённый image_encoder SegVol. Самый дешёвый вариант, SegVol не трогаем.

2. "attention" — fusion на уровне признаков: каждая фаза кодируется отдельно
   (энкодер вызывается снаружи), а здесь эмбеддинги фаз агрегируются обучаемым
   attention-пулингом по оси фаз. Дороже, но сохраняет фазовую информацию в
   признаках -> сильнее для классификации.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseFusion(nn.Module):
    def __init__(
        self,
        mode: str = "attention",
        n_phases: int = 4,
        embed_dim: int = 768,
    ):
        super().__init__()
        if mode not in ("concat_stem", "attention"):
            raise ValueError(f"mode must be 'concat_stem' or 'attention', got {mode!r}")
        self.mode = mode
        self.n_phases = n_phases
        self.embed_dim = embed_dim

        if mode == "concat_stem":
            # (B, n_phases, D, H, W) -> (B, 1, D, H, W)
            self.stem = nn.Sequential(
                nn.Conv3d(n_phases, 8, kernel_size=3, padding=1),
                nn.InstanceNorm3d(8),
                nn.GELU(),
                nn.Conv3d(8, 1, kernel_size=1),
            )
        else:  # attention: скоринг фаз по эмбеддингу
            self.score = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
            )

    # --- режим concat_stem ---
    def fuse_input(self, phases: torch.Tensor) -> torch.Tensor:
        """phases: (B, n_phases, D, H, W) -> (B, 1, D, H, W) для подачи в SegVol."""
        if self.mode != "concat_stem":
            raise RuntimeError("fuse_input доступен только в режиме 'concat_stem'.")
        return self.stem(phases)

    # --- режим attention ---
    def fuse_embeddings(self, phase_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """phase_embeddings: (B, P, C, d, h, w) -> fused (B, C, d, h, w).

        Внимание считается по глобально-усреднённому эмбеддингу каждой фазы;
        возвращаются также веса фаз (B, P) для интерпретируемости.
        """
        if self.mode != "attention":
            raise RuntimeError("fuse_embeddings доступен только в режиме 'attention'.")
        b, p, c, d, h, w = phase_embeddings.shape
        gap = phase_embeddings.flatten(3).mean(-1)        # (B, P, C)
        scores = self.score(gap).squeeze(-1)              # (B, P)
        weights = F.softmax(scores, dim=1)                # (B, P)
        fused = (phase_embeddings * weights[:, :, None, None, None, None]).sum(1)  # (B,C,d,h,w)
        return fused, weights
