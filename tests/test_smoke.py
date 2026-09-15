"""CPU smoke tests: model pieces, loss, splits -- synthetic tensors, no weights, no data.

    python -m pytest tests/test_smoke.py -q
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from liver_sppvr.data.augment import perturb_mask
from liver_sppvr.models import (Backbone, LiverTumorClassifier, PhaseFusion,
                                TumorClassificationHead)
from liver_sppvr.train.losses import ClsLoss
from liver_sppvr.train.splits import dump_splits, make_kfold_splits, stratified_patient_split
from liver_sppvr.utils.device import resolve_device

B, C, N_CLASSES, N_PHASES = 2, 32, 5, 4
D, H, W = 16, 64, 64            # small volume for speed
d, h, w = 4, 8, 8               # feature grid


class _StubBackbone(Backbone):
    """Tiny conv encoder with the Backbone contract: (B,1,D,H,W) -> (B,C,d,h,w)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(1, C, 3, padding=1), nn.GELU(),
                                 nn.Conv3d(C, C, 3, padding=1))
        self.embed_dim = C
        self.preprocess_spec = {"spatial_size": (D, H, W), "normalize": "hu", "hu_window": (-175, 250)}

    @property
    def encoder(self):
        return self.net

    def forward(self, x):
        return F.adaptive_avg_pool3d(self.net(x), (d, h, w))


def _model(pool="masked", extra=0):
    return LiverTumorClassifier(_StubBackbone(), num_classes=N_CLASSES, n_phases=N_PHASES,
                                pool=pool, extra_feat_dim=extra, hidden_dim=16)


def test_device_resolves():
    assert resolve_device("auto").type in ("cuda", "cpu")


def test_cls_head_masked_and_empty_fallback():
    head = TumorClassificationHead(embed_dim=C, num_classes=N_CLASSES, pool="masked", hidden_dim=16)
    emb = torch.randn(B, C, d, h, w)
    assert head(emb, mask=(torch.rand(B, 1, D, H, W) > 0.7).float()).shape == (B, N_CLASSES)
    assert torch.isfinite(head(emb, mask=torch.zeros(B, 1, D, H, W))).all()   # empty -> GAP


def test_phase_fusion_ignores_absent_phase():
    for mode in ("attention", "cross_attention"):
        fusion = PhaseFusion(mode=mode, n_phases=N_PHASES, embed_dim=C, n_heads=4)
        emb = torch.randn(B, N_PHASES, C, d, h, w)
        present = torch.ones(B, N_PHASES); present[0, 3] = 0.0
        fused, wts = fusion.fuse_embeddings(emb, phase_present=present)
        assert fused.shape == (B, C, d, h, w) and wts.shape == (B, N_PHASES)
        assert torch.isfinite(fused).all()
        if mode == "attention":
            assert wts[0, 3].item() == 0.0 and torch.allclose(wts.sum(1), torch.ones(B), atol=1e-5)


def test_classifier_gap_gt_pred_and_proj():
    phases = torch.randn(B, N_PHASES, 1, D, H, W)
    present = torch.ones(B, N_PHASES); present[1, 0] = 0.0
    gt = (torch.rand(B, 1, D, H, W) > 0.7).float()
    m = _model(pool="gap")
    assert not m.uses_mask and m(phases, present).shape == (B, N_CLASSES)
    m = _model(pool="masked", extra=5)
    assert m.uses_mask
    logits, proj = m(phases, present, extra_feat=torch.randn(B, 5), mask=gt, return_proj=True)
    assert logits.shape == (B, N_CLASSES) and proj.shape[0] == B
    assert torch.allclose(proj.norm(dim=1), torch.ones(B), atol=1e-4)
    noisy = perturb_mask(gt[0], p=1.0)                       # predicted-like mask still works
    assert torch.isfinite(m(phases, present, extra_feat=torch.randn(B, 5),
                            mask=torch.stack([noisy, gt[1]]))).all()
    assert m.last_phase_weights.shape == (B, N_PHASES)


def test_backbone_finetune_modes():
    bb = _StubBackbone()
    bb.set_finetune("frozen"); assert not any(p.requires_grad for p in bb.encoder.parameters())
    bb.set_finetune("full"); assert all(p.requires_grad for p in bb.encoder.parameters())
    bb.set_finetune("partial", {"partial_targets": ["2"]})   # only net.2 (last conv)
    assert bb.net[2].weight.requires_grad and not bb.net[0].weight.requires_grad


def test_cls_loss_with_supcon_queue():
    loss_fn = ClsLoss(class_weight=torch.ones(N_CLASSES), supcon_weight=0.2, proj_dim=8, queue_size=16)
    labels = torch.tensor([0, 0])
    for _ in range(3):                                         # queue fills, positives appear
        out = loss_fn(torch.randn(B, N_CLASSES), labels, F.normalize(torch.randn(B, 8), dim=1))
    assert torch.isfinite(out["loss"]) and out["con_loss"] >= 0
    assert int((loss_fn.q_labels >= 0).sum()) == 6


def test_perturb_mask_keeps_shape_and_binary():
    m = torch.zeros(1, D, H, W); m[:, 4:10, 20:40, 20:40] = 1
    for _ in range(20):
        out = perturb_mask(m.clone(), p=1.0)
        assert out.shape == m.shape and set(out.unique().tolist()) <= {0.0, 1.0}


def test_splits_deterministic_and_dumpable():
    labels = {f"p{i:03d}": i % N_CLASSES for i in range(53)}
    a = make_kfold_splits(labels, k=5, seed=2023)
    b = make_kfold_splits(labels, k=5, seed=2023)
    assert a == b
    vals = [p for _, va in a for p in va]
    assert sorted(vals) == sorted(labels) and len(set(vals)) == len(labels)   # each patient once
    for tr, va in a:
        assert not set(tr) & set(va)
    tr, va = stratified_patient_split(labels, val_frac=0.2, seed=2023)
    assert len(tr) + len(va) == len(labels)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.json"); dump_splits(a, path)
        assert len(json.load(open(path))) == 5


def test_build_classifier_config_wiring(monkeypatch):
    """build_classifier reads the config blocks correctly (backbone monkeypatched to the stub)."""
    import liver_sppvr.train.build as build
    monkeypatch.setattr(build, "build_backbone", lambda name, cfg: _StubBackbone())
    cfg = {"classifier": {"backbone": "segvol", "finetune": "frozen", "mask_source": "pred",
                          "num_classes": N_CLASSES, "hidden_dim": 16, "dropout": 0.1},
           "multiphase": {"phases": ["a", "b", "c", "d"], "fusion": "attention"},
           "segvol": {"lora": {"rank": 4}, "partial_targets": ["2"]}}
    m = build.build_classifier(cfg, "cpu", mask_source="none", clinical_dim=5)
    assert not m.uses_mask and m.head.extra_feat_dim == 5
    m = build.build_classifier(cfg, "cpu", finetune="partial")          # mask_source from cfg -> pred
    assert m.uses_mask and m.backbone.net[2].weight.requires_grad
    assert not m.backbone.net[0].weight.requires_grad


if __name__ == "__main__":
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            fn(); print(f"OK  {k}")
    print("\nAll smoke tests passed.")
