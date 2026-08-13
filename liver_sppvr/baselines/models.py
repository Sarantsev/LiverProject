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

    def forward_map(self, x):
        """(B,1,D,H,W) -> (B, feat_dim, d, h, w) feature map (before global pooling)."""
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return x

    def forward(self, x):
        return self.pool(self.forward_map(x)).flatten(1)    # (B, feat_dim)


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
# shared transformer utilities (used by SDR-Former / RA-CMFormer)
# ======================================================================================
class _TokenTransformer(nn.Module):
    """A small pre-norm Transformer encoder over a token sequence (B, N, C)."""
    def __init__(self, dim: int, depth: int = 2, heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)

    def forward(self, x, key_padding_mask=None):            # mask True == ignore token
        return self.enc(x, src_key_padding_mask=key_padding_mask)


class _LowResFormer(nn.Module):
    """Low-resolution branch of SDR-Former: downsample -> conv patch-embed -> 3D
    Transformer -> CLS vector. Mirrors DR-Former's transformer path over coarse images."""
    def __init__(self, out_dim: int, low_size=(16, 64, 64), patch=(4, 16, 16),
                 dim: int = 192, depth: int = 2, heads: int = 4):
        super().__init__()
        self.low_size = tuple(low_size)
        self.embed = nn.Conv3d(1, dim, kernel_size=patch, stride=patch)
        gd = tuple(l // p for l, p in zip(low_size, patch))
        self.n_tok = gd[0] * gd[1] * gd[2]
        self.pos = nn.Parameter(torch.zeros(1, self.n_tok, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.tf = _TokenTransformer(dim, depth=depth, heads=heads)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x):                                  # (N,1,D,H,W) -> (N,out_dim)
        x = F.interpolate(x, size=self.low_size, mode="trilinear", align_corners=False)
        x = self.embed(x).flatten(2).transpose(1, 2) + self.pos      # (N, n_tok, dim)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = self.tf(torch.cat([cls, x], dim=1))
        return self.proj(x[:, 0])


# ======================================================================================
# SDR-Former  (Lou et al., Neural Networks 2025) -- Siamese Dual-Resolution Transformer
# ======================================================================================
class SDRFormer(_BaselineBase):
    """Weight-shared (Siamese) per-phase encoder whose DR-Former fuses a high-resolution
    3D-CNN path with a low-resolution 3D-Transformer path; an Adaptive Phase Selection
    Module (APSM) lets phases communicate (a cross-phase Transformer) and adaptively
    weights each phase before the head. Image-only. Padded phases are masked throughout.
    """
    def __init__(self, num_classes: int, n_phases: int = 4, widths=(32, 64, 128, 256),
                 apsm_heads: int = 4, dropout: float = 0.3,
                 resize: Optional[Sequence[int]] = None):
        super().__init__(resize=resize)
        self.cnn = Encoder3D(in_ch=1, widths=widths)        # high-res Siamese CNN path
        C = self.cnn.feat_dim
        self.lowformer = _LowResFormer(out_dim=C)           # low-res Siamese Transformer path
        self.fuse = nn.Sequential(nn.Linear(2 * C, C), nn.ReLU(inplace=True))
        self.apsm_tf = _TokenTransformer(C, depth=1, heads=apsm_heads)   # phase communication
        self.apsm_gate = nn.Linear(C, 1)                    # adaptive phase weighting
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(C, num_classes))

    def forward(self, phases, phase_present=None, extra_feat=None):
        phases = self._prep_phases(phases)
        B, P = phases.shape[:2]
        x = phases.reshape(B * P, 1, *phases.shape[-3:])
        hi = self.cnn(x)                                    # (B*P, C)
        lo = self.lowformer(x)                              # (B*P, C)
        feat = self.fuse(torch.cat([hi, lo], dim=1)).reshape(B, P, -1)   # (B, P, C)
        kpm = (phase_present < 0.5) if phase_present is not None else None
        comm = self.apsm_tf(feat, key_padding_mask=kpm)     # cross-phase communication
        gate = self.apsm_gate(comm).squeeze(-1)             # (B, P)
        if phase_present is not None:
            gate = gate.masked_fill(phase_present < 0.5, float("-inf"))
        w = torch.softmax(gate, dim=1).unsqueeze(-1)        # (B, P, 1)
        pooled = (comm * w).sum(1)                          # (B, C)
        return self.head(pooled)


# ======================================================================================
# LCA-Net / LCA-DB  (Wang et al., Information Fusion 2024) -- dual-branch cross-attention
# ======================================================================================
class LCANet(_BaselineBase):
    """LCA-DB: a global branch (IA-Net) and a patch branch (PA-Net) exchange information
    through cross-attention, with an attention-similarity auxiliary loss that pulls the
    two branches' attention distributions together. The aux term is stashed in
    ``self.last_aux`` so the trainer can add it. Image-only; pooled over present phases.
    """
    def __init__(self, num_classes: int, n_phases: int = 4, widths=(32, 64, 128, 256),
                 heads: int = 4, dropout: float = 0.3, aux_weight: float = 0.1,
                 resize: Optional[Sequence[int]] = None):
        super().__init__(resize=resize)
        self.ia = Encoder3D(in_ch=1, widths=widths)         # global (image) branch
        self.pa = Encoder3D(in_ch=1, widths=widths)         # patch branch
        C = self.ia.feat_dim
        self.ca_i2p = nn.MultiheadAttention(C, heads, dropout=dropout, batch_first=True)
        self.ca_p2i = nn.MultiheadAttention(C, heads, dropout=dropout, batch_first=True)
        self.aux_weight = aux_weight
        self.last_aux = None
        self.head = nn.Sequential(nn.Linear(2 * C, C), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(C, num_classes))

    def forward(self, phases, phase_present=None, extra_feat=None):
        phases = self._prep_phases(phases)
        B, P = phases.shape[:2]
        x = phases.reshape(B * P, 1, *phases.shape[-3:])
        g = self.ia(x).unsqueeze(1)                         # (B*P, 1, C) global token
        tok = self.pa.forward_map(x).flatten(2).transpose(1, 2)          # (B*P, N, C) patches
        gi, a_i2p = self.ca_i2p(g, tok, tok, need_weights=True, average_attn_weights=True)
        pi, a_p2i = self.ca_p2i(tok.mean(1, keepdim=True), tok, tok,
                                need_weights=True, average_attn_weights=True)
        self.last_aux = self.aux_weight * F.mse_loss(a_i2p, a_p2i)       # attention similarity
        fused = torch.cat([gi.squeeze(1), pi.squeeze(1)], dim=1).reshape(B, P, -1)   # (B,P,2C)
        if phase_present is not None:
            w = phase_present[..., None]
            pooled = (fused * w).sum(1) / w.sum(1).clamp(min=1e-6)
        else:
            pooled = fused.mean(1)
        return self.head(pooled)


# ======================================================================================
# RA-CMFormer  (Su et al., MCT-LTDiag benchmark 2025) -- UniFormer image branch
# ======================================================================================
class RACMFormer(_BaselineBase):
    """Image branch of RA-CMFormer: a UniFormer-style per-phase encoder (conv stages then
    a global self-attention stage over the feature-map tokens) followed by a cross-phase
    Transformer whose CLS token summarises the phases. With ``clinical_dim>0`` and an
    ``extra_feat`` radiomics vector it becomes the full cross-modal model (Table 2a);
    ``clinical_dim=0`` is the pure image branch (Table 1). Padded phases are masked.
    """
    def __init__(self, num_classes: int, n_phases: int = 4, clinical_dim: int = 0,
                 widths=(32, 64, 128, 256), tf_depth: int = 2, heads: int = 4,
                 fuse_hidden: int = 256, dropout: float = 0.3,
                 resize: Optional[Sequence[int]] = None):
        super().__init__(resize=resize)
        self.cnn = Encoder3D(in_ch=1, widths=widths)
        C = self.cnn.feat_dim
        self.spatial_tf = _TokenTransformer(C, depth=tf_depth, heads=heads)  # global stage
        self.phase_tf = _TokenTransformer(C, depth=1, heads=heads)           # cross-phase fusion
        self.phase_cls = nn.Parameter(torch.zeros(1, 1, C))
        self.clinical_dim = clinical_dim
        if clinical_dim > 0:
            self.rad = nn.Sequential(nn.Linear(clinical_dim, fuse_hidden), nn.ReLU(inplace=True),
                                     nn.Dropout(dropout))
            fused_in = C + fuse_hidden
        else:
            self.rad = None
            fused_in = C
        self.head = nn.Sequential(nn.Linear(fused_in, fuse_hidden), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(fuse_hidden, num_classes))

    def forward(self, phases, phase_present=None, extra_feat=None):
        phases = self._prep_phases(phases)
        B, P = phases.shape[:2]
        x = phases.reshape(B * P, 1, *phases.shape[-3:])
        tok = self.cnn.forward_map(x).flatten(2).transpose(1, 2)         # (B*P, N, C)
        tok = self.spatial_tf(tok)                          # global self-attention stage
        feat = tok.mean(1).reshape(B, P, -1)                # (B, P, C) per-phase
        cls = self.phase_cls.expand(B, -1, -1)              # (B, 1, C)
        seq = torch.cat([cls, feat], dim=1)                 # (B, 1+P, C)
        kpm = None
        if phase_present is not None:
            cls_ok = torch.ones(B, 1, device=feat.device, dtype=phase_present.dtype)
            kpm = torch.cat([cls_ok, phase_present], dim=1) < 0.5        # CLS always kept
        img_feat = self.phase_tf(seq, key_padding_mask=kpm)[:, 0]       # (B, C)
        if self.rad is not None and extra_feat is not None:
            fused = torch.cat([img_feat, self.rad(extra_feat.float())], dim=1)
        else:
            fused = img_feat
        return self.head(fused)


# ======================================================================================
# registry
# ======================================================================================
def build_baseline(name: str, num_classes: int, n_phases: int, *, clinical_dim: int = 0,
                   resize=None, **kw) -> nn.Module:
    name = name.lower().replace("-", "").replace("_", "")
    if name == "hlstm":
        return HLSTM(num_classes, n_phases=n_phases, resize=resize, **kw)
    if name == "stic":
        return STIC(num_classes, n_phases=n_phases, clinical_dim=clinical_dim, resize=resize, **kw)
    if name in ("sdrformer", "sdr"):
        return SDRFormer(num_classes, n_phases=n_phases, resize=resize, **kw)
    if name in ("lcanet", "lca", "lcadb"):
        return LCANet(num_classes, n_phases=n_phases, resize=resize, **kw)
    if name in ("racmformer", "racm", "uniformer"):
        return RACMFormer(num_classes, n_phases=n_phases, clinical_dim=clinical_dim,
                          resize=resize, **kw)
    raise ValueError(f"unknown baseline '{name}'. available: "
                     "hlstm, stic, sdrformer, lcanet, racmformer")


BASELINES = ("hlstm", "stic", "sdrformer", "lcanet", "racmformer")
