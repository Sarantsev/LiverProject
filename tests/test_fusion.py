"""CPU tests for multi-phase fusion (no datasets, no pretrained weights).

    python -m pytest tests/test_fusion.py -q

The first test is the reason the "diff" mode exists: it shows that the previous default
("attention") is mathematically blind to contrast DYNAMICS, which is the radiological
basis for every subtype in MCT-LTDiag.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from liver_sppvr.models.multiphase import MODES, PhaseFusion

C, D, H, W = 8, 2, 3, 3
P = 4


def _washout_pair():
    """Two 4-phase inputs with the SAME per-phase content but opposite dynamics.

    a: enhancement rises then falls (washout, as in HCC)
    b: enhancement rises monotonically (progressive, as in ICC)

    The multiset of phase embeddings is identical, so any permutation-invariant convex
    combination of the phases maps both to the same point.
    """
    torch.manual_seed(0)
    lo, mid, hi = (torch.randn(1, C, D, H, W) for _ in range(3))
    a = torch.stack([lo, hi, mid, lo], dim=1)       # low -> high -> mid -> low
    b = torch.stack([lo, lo, mid, hi], dim=1)       # low -> low  -> mid -> high
    return a, b


def test_attention_fusion_is_blind_to_contrast_dynamics():
    """The old default collapses washout and progressive enhancement to one point."""
    a, b = _washout_pair()
    f = PhaseFusion(mode="attention", n_phases=P, embed_dim=C).eval()
    with torch.no_grad():
        ya, _ = f.fuse_embeddings(a)
        yb, _ = f.fuse_embeddings(b)
    # identical phase CONTENT, opposite ORDER -> a convex combination cannot tell them apart.
    # The scores are computed per phase from its own embedding, so equal phases get equal
    # weights and the weighted sums coincide.
    assert torch.allclose(ya, yb, atol=1e-5), "attention unexpectedly separated the two"


def test_diff_fusion_separates_contrast_dynamics():
    """The new default distinguishes them, because differences are explicit streams."""
    a, b = _washout_pair()
    f = PhaseFusion(mode="diff", n_phases=P, embed_dim=C).eval()
    with torch.no_grad():
        ya, _ = f.fuse_embeddings(a)
        yb, _ = f.fuse_embeddings(b)
    rel = (ya - yb).norm() / ya.norm().clamp_min(1e-6)
    assert rel > 0.1, f"diff fusion should separate the dynamics, got rel={rel:.4f}"


def test_diff_streams_are_phases_plus_differences():
    f = PhaseFusion(mode="diff", n_phases=P, embed_dim=C)
    assert f.n_streams == 2 * P - 1 == 7
    assert f.out_dim == C                      # projects back to the backbone width


@pytest.mark.parametrize("mode", [m for m in MODES if m != "concat_stem"])
def test_shapes_and_weight_contract(mode):
    x = torch.randn(2, P, C, D, H, W)
    f = PhaseFusion(mode=mode, n_phases=P, embed_dim=C).eval()
    with torch.no_grad():
        y, wts = f.fuse_embeddings(x)
    assert y.shape == (2, C, D, H, W)
    assert wts.shape == (2, P)
    assert torch.allclose(wts.sum(1), torch.ones(2), atol=1e-4)


def test_diff_ignores_absent_phases():
    """A zero-filled (missing) phase must not leak in as a spurious difference."""
    x = torch.randn(1, P, C, D, H, W)
    x[:, 3] = 0.0                                          # delayed phase missing
    present = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    f = PhaseFusion(mode="diff", n_phases=P, embed_dim=C).eval()
    with torch.no_grad():
        f.fuse_embeddings(x, phase_present=present)
    sw = f.last_stream_weights[0]
    # streams: 0..3 phases, 4 = a-nc, 5 = p-a, 6 = d-p  -> 3 and 6 must be exactly zero
    assert sw[3].item() == 0.0 and sw[6].item() == 0.0
    assert sw[:3].sum().item() > 0.0


def test_diff_parameter_count_stays_near_budget():
    """The bottleneck keeps the added capacity ~1.6M at the real embed_dim of 768."""
    f = PhaseFusion(mode="diff", n_phases=4, embed_dim=768, width=256)
    n = sum(p.numel() for p in f.parameters())
    assert 1.2e6 < n < 2.0e6, f"diff fusion has {n/1e6:.2f}M params"


# --------------------------------------------------------------------- AMP dtypes
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_masked_pool_accepts_half_embeddings(dtype):
    """Regression: under AMP the fused map is fp16 because `diff` ends in a Conv3d.

    The mask is float32, so `embedding * mask` promotes to float32, while the empty-mask
    GAP fallback stayed fp16 -- index_put_ then raised
    "Index put requires the source and destination dtypes match".
    """
    from liver_sppvr.models.cls_head import TumorClassificationHead
    head = TumorClassificationHead(embed_dim=C, num_classes=3, hidden_dim=16, pool="masked")
    emb = torch.randn(2, C, D, H, W, dtype=dtype)
    mask = torch.zeros(2, 1, D * 4, H * 4, W * 4)          # fully empty -> GAP fallback
    mask[0, :, :2] = 1.0                                   # one real, one empty
    with torch.no_grad():
        pooled = head._masked_pool(emb, mask)
    assert pooled.shape == (2, C) and pooled.dtype == torch.float32
    assert torch.isfinite(pooled).all()
