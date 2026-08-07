"""Faithful re-implementations of published liver-tumour classification baselines,
adapted to a common protocol so they can be compared apples-to-apples with our model.

Every baseline shares the SAME input as our pipeline produces
    phases: (B, P, 1, D, H, W)   -- P multi-phase CT volumes (missing phases zeroed)
    phase_present: (B, P)        -- 1.0 if the phase is real, 0.0 if it was padded
    extra_feat: (B, F) | None    -- optional clinical/tabular vector (STIC uses it)
and returns classification logits (B, num_classes). No segmentation prompts, no GT
mask -- pure fully-automatic classification from the CT (+ clinical) alone.

Design note: the published repos each expect their own dataset layout, phase set and
class count (SDR-Former: LLD-MMRI 3-CT/8-MR; LCA-Net: MRI 7-class; H-LSTM: CECT
HCC/ICC/normal; STIC: CT HCC/ICC/metastasis). Running them as-is on MCT-LTDiag is
impossible, so we port the *architecture* and retrain it under one identical protocol
(same 5-fold split, augmentation, optimizer, 5-class head). Backbone widths are kept
close to the papers; where a paper leaves a hyper-parameter unspecified we use the
common default and say so in the paper's methods section.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# shared 3D CNN encoder (compact ResNet) -- the imaging backbone reused by the baselines
# ======================================================================================
class BasicBlock3D(nn.Module):
    def __init__(self, inp: int, out: int, stride=1):
        super().__init__()
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.conv1 = nn.Conv3d(inp, out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out)
        self.conv2 = nn.Conv3d(out, out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out)
        self.down = None
        if inp != out or stride != (1, 1, 1):
            self.down = nn.Sequential(nn.Conv3d(inp, out, 1, stride=stride, bias=False),
                                      nn.BatchNorm3d(out))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + idt, inplace=True)


class Encoder3D(nn.Module):
    """A compact 3D ResNet (r3d-18-style). (B,1,D,H,W) -> (B, feat_dim).

    Stem uses in-plane stride only so shallow depth (D=32) is preserved into the
    first stage; later stages downsample all three axes. widths default to a
    ResNet-18-scaled stem to keep VRAM sane for from-scratch 3D training.
    """
    def __init__(self, in_ch: int = 1, widths: Sequence[int] = (32, 64, 128, 256)):
        super().__init__()
        w0 = widths[0]
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, w0, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(w0), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.layer1 = self._stage(widths[0], widths[0], stride=(1, 1, 1))
        self.layer2 = self._stage(widths[0], widths[1], stride=(2, 2, 2))
        self.layer3 = self._stage(widths[1], widths[2], stride=(2, 2, 2))
        self.layer4 = self._stage(widths[2], widths[3], stride=(2, 2, 2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.feat_dim = widths[3]

    @staticmethod
    def _stage(inp, out, stride):
        return nn.Sequential(BasicBlock3D(inp, out, stride=stride), BasicBlock3D(out, out))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.pool(x).flatten(1)                      # (B, feat_dim)


# ======================================================================================
# base class: input reshaping / optional resize shared by all baselines
# ======================================================================================
class _BaselineBase(nn.Module):
    def __init__(self, resize: Optional[Sequence[int]] = None):
        super().__init__()
        self.resize = tuple(resize) if resize else None

    def _prep_phases(self, phases: torch.Tensor) -> torch.Tensor:
        """(B,P,1,D,H,W) -> (B,P,1,D',H',W') after optional trilinear resize (VRAM control)."""
        if phases.dim() == 5:                                # (B,P,D,H,W) -> add channel
            phases = phases.unsqueeze(2)
        if self.resize is not None:
            B, P = phases.shape[:2]
            x = phases.reshape(B * P, 1, *phases.shape[-3:])
            x = F.interpolate(x, size=self.resize, mode="trilinear", align_corners=False)
            phases = x.reshape(B, P, 1, *self.resize)
        return phases


# ======================================================================================
# H-LSTM  (Huang et al., J Cancer Res Clin Oncol 2024) -- ResNet + BiLSTM over phases
# ======================================================================================
class HLSTM(_BaselineBase):
    """3D-ResNet encoder shared across phases -> BiLSTM aggregates the phase sequence
    -> FC. Faithful to ResNet_BiLSTM: a CNN backbone extracts per-phase features and a
    bidirectional LSTM models the (contrast-)phase progression. Missing phases are masked
    out of the sequence via phase_present so variable phase availability is handled.
    """
    def __init__(self, num_classes: int, n_phases: int = 4, widths=(32, 64, 128, 256),
                 lstm_hidden: int = 256, lstm_layers: int = 1, dropout: float = 0.3,
                 resize: Optional[Sequence[int]] = None):
        super().__init__(resize=resize)
        self.encoder = Encoder3D(in_ch=1, widths=widths)
        self.lstm = nn.LSTM(self.encoder.feat_dim, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * lstm_hidden, num_classes))

    def forward(self, phases, phase_present=None, extra_feat=None):
        phases = self._prep_phases(phases)
        B, P = phases.shape[:2]
        feats = self.encoder(phases.reshape(B * P, 1, *phases.shape[-3:]))   # (B*P, F)
        feats = feats.reshape(B, P, -1)                                      # (B, P, F)
        if phase_present is not None:
            feats = feats * phase_present[..., None]        # zero padded phases
        out, _ = self.lstm(feats)                           # (B, P, 2*hidden)
        if phase_present is not None:                       # mean over PRESENT phases only
            w = phase_present[..., None]
            pooled = (out * w).sum(1) / w.sum(1).clamp(min=1e-6)
        else:
            pooled = out.mean(1)
        return self.head(pooled)


# ======================================================================================
# STIC  (Shi et al., J Hematol Oncol 2021) -- imaging CNN + clinical branch, fused
# ======================================================================================
class STIC(_BaselineBase):
    """Two-branch model: a 3D-CNN imaging branch (per-phase encoder, attention-pooled
    across phases) fused with a clinical MLP branch, then a joint classifier. Faithful to
    STIC's Spatio-Temporal + Clinical fusion. The clinical vector arrives as extra_feat
    (build via scripts/build_clinical.py). If no clinical vector is given it degrades to
    the imaging branch alone (report that ablation separately).
    """
    def __init__(self, num_classes: int, n_phases: int = 4, clinical_dim: int = 0,
                 widths=(32, 64, 128, 256), clin_hidden: int = 64, fuse_hidden: int = 256,
                 dropout: float = 0.3, resize: Optional[Sequence[int]] = None):
        super().__init__(resize=resize)
        self.encoder = Encoder3D(in_ch=1, widths=widths)
        F_img = self.encoder.feat_dim
        self.phase_attn = nn.Linear(F_img, 1)               # attention pooling over phases
        self.clinical_dim = clinical_dim
        if clinical_dim > 0:
            self.clin = nn.Sequential(nn.Linear(clinical_dim, clin_hidden), nn.ReLU(inplace=True),
                                      nn.Dropout(dropout), nn.Linear(clin_hidden, clin_hidden),
                                      nn.ReLU(inplace=True))
            fused_in = F_img + clin_hidden
        else:
            self.clin = None
            fused_in = F_img
        self.head = nn.Sequential(nn.Linear(fused_in, fuse_hidden), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(fuse_hidden, num_classes))

    def forward(self, phases, phase_present=None, extra_feat=None):
        phases = self._prep_phases(phases)
        B, P = phases.shape[:2]
        feats = self.encoder(phases.reshape(B * P, 1, *phases.shape[-3:])).reshape(B, P, -1)
        attn = self.phase_attn(feats).squeeze(-1)           # (B, P)
        if phase_present is not None:
            attn = attn.masked_fill(phase_present < 0.5, float("-inf"))
        attn = torch.softmax(attn, dim=1).unsqueeze(-1)     # (B, P, 1)
        img_feat = (feats * attn).sum(1)                    # (B, F_img)
        if self.clin is not None and extra_feat is not None:
            clin_feat = self.clin(extra_feat.float())
            fused = torch.cat([img_feat, clin_feat], dim=1)
        else:
            fused = img_feat
        return self.head(fused)


# ======================================================================================
# registry
# ======================================================================================
def build_baseline(name: str, num_classes: int, n_phases: int, *, clinical_dim: int = 0,
                   resize=None, **kw) -> nn.Module:
    name = name.lower()
    if name == "hlstm":
        return HLSTM(num_classes, n_phases=n_phases, resize=resize, **kw)
    if name == "stic":
        return STIC(num_classes, n_phases=n_phases, clinical_dim=clinical_dim, resize=resize, **kw)
    raise ValueError(f"unknown baseline '{name}'. available: hlstm, stic "
                     "(sdrformer, lcanet coming next)")


BASELINES = ("hlstm", "stic")
