"""CPU dry-run: проверяем формы и forward классификационной головы и fusion фаз.

Запуск без GPU и без данных (синтетические тензоры):
    ../segvol_env/bin/python -m pytest tests/test_smoke.py -q
или просто:
    ../segvol_env/bin/python tests/test_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from liver_sppvr.models import TumorClassificationHead, PhaseFusion, SegVolMultiTask
from liver_sppvr.utils.device import resolve_device


class _StubEncoder(torch.nn.Module):
    """Заглушка image_encoder SegVol: (B,1,D,H,W) -> (B, N_tokens, C)."""
    def __init__(self, feat=(8, 16, 16), c=768):
        super().__init__()
        self.n = feat[0] * feat[1] * feat[2]
        self.c = c
        self.proj = torch.nn.Linear(1, c)

    def forward(self, x):
        b = x.shape[0]
        tok = torch.randn(b, self.n, 1, device=x.device)
        return self.proj(tok), None

# Геометрия SegVol: feat = roi/patch = (32,256,256)/(4,16,16) = (8,16,16), C=768
B, C, d, h, w = 2, 768, 8, 16, 16
D, H, W = 32, 256, 256
N_CLASSES, N_PHASES = 5, 4


def test_device_resolves():
    dev = resolve_device("auto")
    assert dev.type in ("cuda", "cpu")


def test_cls_head_masked():
    head = TumorClassificationHead(embed_dim=C, num_classes=N_CLASSES, pool="masked")
    emb = torch.randn(B, C, d, h, w)
    mask = (torch.rand(B, 1, D, H, W) > 0.7).float()       # маска в полном разрешении
    logits = head(emb, mask=mask)
    assert logits.shape == (B, N_CLASSES)


def test_cls_head_empty_mask_fallback():
    head = TumorClassificationHead(embed_dim=C, num_classes=N_CLASSES, pool="masked")
    emb = torch.randn(B, C, d, h, w)
    mask = torch.zeros(B, 1, D, H, W)                       # пустая маска -> fallback на GAP
    logits = head(emb, mask=mask)
    assert torch.isfinite(logits).all()


def test_cls_head_hybrid_radiomics():
    extra = 32                                             # размер радиомического вектора
    head = TumorClassificationHead(embed_dim=C, num_classes=N_CLASSES,
                                   pool="gap", extra_feat_dim=extra)
    emb = torch.randn(B, C, d, h, w)
    feat = torch.randn(B, extra)
    logits = head(emb, extra_feat=feat)
    assert logits.shape == (B, N_CLASSES)


def test_phase_fusion_concat_stem():
    fusion = PhaseFusion(mode="concat_stem", n_phases=N_PHASES)
    phases = torch.randn(B, N_PHASES, 8, 64, 64)           # уменьшенный объём для скорости
    fused = fusion.fuse_input(phases)
    assert fused.shape == (B, 1, 8, 64, 64)


def test_phase_fusion_attention():
    fusion = PhaseFusion(mode="attention", n_phases=N_PHASES, embed_dim=C)
    emb = torch.randn(B, N_PHASES, C, d, h, w)
    fused, weights = fusion.fuse_embeddings(emb)
    assert fused.shape == (B, C, d, h, w)
    assert weights.shape == (B, N_PHASES)
    assert torch.allclose(weights.sum(1), torch.ones(B), atol=1e-5)


def test_multitask_encode_and_classify():
    model = SegVolMultiTask(
        image_encoder=_StubEncoder(feat=(d, h, w), c=C),
        prompt_encoder=None, mask_decoder=None, text_encoder=None,
        roi_size=(D, H, W), patch_size=(4, 16, 16), embed_dim=C,
        num_classes=N_CLASSES, cls_pool="masked", fusion_mode="attention", n_phases=N_PHASES,
    )
    image = torch.randn(B, 1, D, H, W)
    emb = model.encode(image)
    assert emb.shape == (B, C, d, h, w)
    mask = (torch.rand(B, 1, D, H, W) > 0.7).float()
    out = model(image=image, cls_mask=mask, return_seg=False, return_cls=True)
    assert out["cls_logits"].shape == (B, N_CLASSES)


def test_multitask_multiphase_attention():
    model = SegVolMultiTask(
        image_encoder=_StubEncoder(feat=(d, h, w), c=C),
        prompt_encoder=None, mask_decoder=None, text_encoder=None,
        roi_size=(D, H, W), patch_size=(4, 16, 16), embed_dim=C,
        num_classes=N_CLASSES, cls_pool="masked", fusion_mode="attention", n_phases=N_PHASES,
    )
    phases = torch.randn(B, N_PHASES, 1, D, H, W)
    mask = (torch.rand(B, 1, D, H, W) > 0.7).float()
    out = model(phases=phases, cls_mask=mask, return_seg=False, return_cls=True)
    assert out["cls_logits"].shape == (B, N_CLASSES)
    assert out["phase_weights"].shape == (B, N_PHASES)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print("\nВсе smoke-тесты прошли.")
