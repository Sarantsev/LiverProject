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


def test_merlin_encoder_random_init_forward_and_partial():
    """Vendored I3D-ResNet-152 builds from the torchvision skeleton, runs on (B,1,D,H,W),
    and exposes layer4/contrastive_head for finetune=partial. (No weights downloaded.)"""
    import torchvision
    from liver_sppvr.models.merlin import I3ResNetEncoder, MerlinBackbone
    enc = I3ResNetEncoder(torchvision.models.resnet152(weights=None), features="contrastive")
    bb = MerlinBackbone(enc, spatial_size=(16, 64, 64))
    assert bb.embed_dim == 512 and bb.preprocess_spec["spacing"] == (3.0, 1.5, 1.5)
    bb.eval()
    with torch.no_grad():
        fmap = bb(torch.rand(1, 1, 16, 64, 64))
    assert fmap.shape == (1, 512, 1, 2, 2)                       # D/16, H/32, W/32
    bb.set_finetune("partial", {"partial_targets": ["layer4", "contrastive_head"]})
    assert enc.layer4[0].conv1.weight.requires_grad and enc.contrastive_head.weight.requires_grad
    assert not enc.layer3[0].conv1.weight.requires_grad and not enc.conv1.weight.requires_grad
    # state-dict key layout matches Merlin's image tower (loader strips 'encode_image.i3_resnet.')
    keys = set(enc.state_dict())
    assert {"conv1.weight", "bn1.running_mean", "layer4.2.conv3.weight", "contrastive_head.bias"} <= keys
    assert not any(k.startswith("classifier") for k in keys)


def test_merlin_freezes_batchnorm_running_stats():
    """freeze_bn must keep every BatchNorm in eval mode and leave its pretrained running
    stats untouched by a train-mode forward. Without this, batch-1-per-GPU training
    overwrites them with noise and the classifier collapses onto one class."""
    import torchvision
    from liver_sppvr.models.merlin import I3ResNetEncoder, MerlinBackbone
    enc = I3ResNetEncoder(torchvision.models.resnet152(weights=None))
    bb = MerlinBackbone(enc, spatial_size=(16, 64, 64))
    bns = [m for m in bb.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert len(bns) > 100, len(bns)                       # ResNet-152 has many

    bb.train()                                            # the whole point: train() must not un-freeze
    assert bb.training and all(not m.training for m in bns)
    before = [m.running_mean.clone() for m in bns]
    x = torch.rand(1, 1, 16, 64, 64)
    x.requires_grad_(True)                                # exercise the checkpointed path too
    bb(x).sum().backward()
    assert all(torch.equal(m.running_mean, b) for m, b in zip(bns, before))

    # and the opt-out still behaves like a normal ResNet
    bb2 = MerlinBackbone(I3ResNetEncoder(torchvision.models.resnet152(weights=None)),
                         spatial_size=(16, 64, 64), freeze_bn=False).train()
    assert all(m.training for m in bb2.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm))


def test_finetune_mode_follows_the_backbone(monkeypatch):
    """The adaptation mode must come from the backbone, not from a global classifier default:
    DoRA wraps nn.Linear and is impossible on a convolutional encoder."""
    import liver_sppvr.train.build as build
    from liver_sppvr.models.backbones import Backbone, SegVolViTBackbone

    assert SegVolViTBackbone.default_finetune == "dora"          # ViT -> adapters
    from liver_sppvr.models.merlin import MerlinBackbone
    assert MerlinBackbone.default_finetune == "partial"          # conv -> partial unfreeze
    assert _StubBackbone.default_finetune == Backbone.default_finetune

    # a conv encoder asked for DoRA fails with a message that names the alternative
    bb = _StubBackbone()
    assert not any(isinstance(m, nn.Linear) for m in bb.encoder.modules())
    try:
        bb.set_finetune("dora", {"targets": ["qkv"]})
    except ValueError as e:
        assert "partial" in str(e) and "convolutional" in str(e), str(e)
    else:
        raise AssertionError("DoRA on a convolutional encoder must raise")

    # build_classifier takes the mode from the backbone's config block, not classifier.*
    monkeypatch.setattr(build, "build_backbone", lambda name, cfg: _StubBackbone())
    cfg = {"classifier": {"backbone": "stub", "mask_source": "none", "num_classes": N_CLASSES,
                          "hidden_dim": 16, "dropout": 0.1},
           "multiphase": {"phases": ["a", "b", "c", "d"], "fusion": "attention"},
           "stub": {"finetune": "partial", "partial_targets": ["2"]}}
    m = build.build_classifier(cfg, "cpu")
    assert m.backbone.net[2].weight.requires_grad and not m.backbone.net[0].weight.requires_grad


def test_gated_head_forces_both_modalities_to_learn():
    """One end-to-end model, not an ensemble: a gated fusion plus per-modality auxiliary
    heads plus modality dropout, so the clinical branch keeps receiving gradient after the
    much larger image branch has driven the main loss to zero."""
    emb = torch.randn(B, C, d, h, w)
    clin = torch.randn(B, 4)

    head = TumorClassificationHead(embed_dim=C, num_classes=N_CLASSES, hidden_dim=16,
                                   pool="gap", extra_feat_dim=4, head="gated", fuse_dim=32,
                                   img_drop=0.3, clin_drop=0.15)
    head.train()
    logits, ex = head(emb, extra_feat=clin, return_extras=True)
    assert logits.shape == (B, N_CLASSES)
    assert ex["aux_img"].shape == (B, N_CLASSES) and ex["aux_clin"].shape == (B, N_CLASSES)
    assert ex["gate"].shape == (B,) and (0 <= ex["gate"]).all() and (ex["gate"] <= 1).all()

    # the auxiliary path delivers gradient to the clinical encoder
    (logits.sum() + ex["aux_clin"].sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.clin.parameters())

    # modality dropout never zeroes both branches at once
    hard = TumorClassificationHead(embed_dim=C, num_classes=N_CLASSES, hidden_dim=16,
                                   pool="gap", extra_feat_dim=4, head="gated", fuse_dim=32,
                                   img_drop=0.95, clin_drop=0.95).train()
    for _ in range(100):
        i, c = hard._modality_dropout(torch.ones(8, C), torch.ones(8, 64))
        assert not ((i.abs().sum(1) == 0) & (c.abs().sum(1) == 0)).any()

    # eval is deterministic (dropout off) and still responds to the clinical vector
    head.eval()
    with torch.no_grad():
        a, b = head(emb, extra_feat=clin), head(emb, extra_feat=clin)
        cc = head(emb, extra_feat=clin * -3)
    assert torch.allclose(a, b) and not torch.allclose(a, cc, atol=1e-4)

    # image-only and the legacy concat head both still build and run
    for kind, dim in (("gated", 0), ("concat", 4)):
        hh = TumorClassificationHead(embed_dim=C, num_classes=N_CLASSES, hidden_dim=16,
                                     pool="gap", extra_feat_dim=dim, head=kind, fuse_dim=32)
        assert hh(emb, extra_feat=(clin if dim else None)).shape == (B, N_CLASSES)


def test_aux_loss_term_is_added():
    """ClsLoss must add the per-modality terms; without them the clinical branch is starved."""
    from liver_sppvr.train.losses import ClsLoss
    y = torch.tensor([0, 1])
    main = torch.randn(2, N_CLASSES)
    aux = [torch.randn(2, N_CLASSES), torch.randn(2, N_CLASSES)]
    off = ClsLoss(aux_weight=0.0)(main, y, None, aux)
    on = ClsLoss(aux_weight=0.5)(main, y, None, aux)
    assert float(off["aux_loss"]) == 0.0 and float(on["aux_loss"]) > 0
    assert float(on["loss"]) > float(off["loss"])
    assert float(ClsLoss(aux_weight=0.5)(main, y, None, [None, None])["aux_loss"]) == 0.0


def test_trainer_does_not_override_the_backbone_finetune_mode():
    """The CLI must pass finetune=None through when the user did not ask for a mode, so the
    backbone resolves it. A default resolved in the trainer silently sent 'dora' to a
    convolutional encoder and crashed the run."""
    import ast, pathlib as _p
    src = _p.Path("scripts/train_cls.py").read_text()
    tree = ast.parse(src)
    # no statement may assign a fallback to args.finetune
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "finetune":
                    raise AssertionError("train_cls.py must not default args.finetune; "
                                         "build_classifier resolves it per backbone")
    # and the CLI default stays None
    assert '--finetune", default=None' in src.replace("'", '"')


def test_class_merge_keeps_the_folds_comparable():
    """The 4-class task (CRLM+BCLM -> Metastasis, as in the published work) must reuse the
    5-class folds: stratifying on the merged labels would give different folds and make the
    two formulations incomparable on the same patients."""
    import pandas as pd
    from liver_sppvr.train.splits import (labels_from_manifest, make_kfold_splits,
                                          merged_class_names, parse_class_merge)
    orig = ["HCC", "ICC", "CRLM", "BCLM", "HH"]
    man = pd.DataFrame([{"patient_id": f"p{i:03d}", "tumor_type": orig[i % 5],
                         "phase": "portal", "image_path": "x", "mask_path": "m"}
                        for i in range(50)])

    # spec parsing, both the config form and the CLI form
    assert parse_class_merge({"Metastasis": ["CRLM", "BCLM"]}) == \
           parse_class_merge(["Metastasis=CRLM,BCLM"]) == \
           {"CRLM": "Metastasis", "BCLM": "Metastasis"}
    assert parse_class_merge(None) == {} and parse_class_merge({}) == {}

    m = parse_class_merge(["Metastasis=CRLM,BCLM"])
    assert merged_class_names(orig, m) == ["HCC", "ICC", "Metastasis", "HH"]

    lo = labels_from_manifest(man, orig)
    lm = labels_from_manifest(man, orig, m)
    assert len(set(lo.values())) == 5 and len(set(lm.values())) == 4
    assert lm["p002"] == lm["p003"]          # CRLM and BCLM collapse together
    assert lm["p000"] != lm["p002"]          # HCC stays separate
    assert set(lo) == set(lm)                # nobody is dropped

    # folds built from the ORIGINAL labels are what both formulations must share
    a = make_kfold_splits(lo, 5, 2023)
    assert a == make_kfold_splits(lo, 5, 2023)
    assert a != make_kfold_splits(lm, 5, 2023), "merged labels would change the folds"


def test_clinical_exclude_drops_a_column():
    """--clinical-exclude clin_chemo must remove it, and must fail loudly on a typo."""
    import tempfile, os
    from liver_sppvr.data.clinical import feat_dim, load_clinical
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.csv")
        with open(path, "w") as f:
            f.write("patient_id,clin_sex,clin_age,clin_chemo,tumor_type\n")
            for i in range(6):
                f.write(f"p{i},{i%2},{50+i},{i%2},HCC\n")
        ids = [f"p{i}" for i in range(4)]
        assert feat_dim(load_clinical(path, ids)) == 3
        assert feat_dim(load_clinical(path, ids, exclude=("clin_chemo",))) == 2
        try:
            load_clinical(path, ids, exclude=("clin_nope",))
        except ValueError as e:
            assert "clin_nope" in str(e)
        else:
            raise AssertionError("excluding a missing column must raise")


if __name__ == "__main__":
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            fn(); print(f"OK  {k}")
    print("\nAll smoke tests passed.")
