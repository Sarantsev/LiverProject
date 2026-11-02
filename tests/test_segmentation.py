"""Segmentation export, native-space predictions and dual-encoder gradients."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("timm")

from liver_sppvr.data.segmentation import SliceDataset, native_dice, predict_volume, prepare_cache
from liver_sppvr.models.seg_dual import DualSegmenter, segmentation_loss


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cohort(tmp_path):
    rows = []
    phases = ["non_contrast", "arterial", "portal", "delayed"]
    for i in range(6):
        mask = tmp_path / f"m{i}.nii.gz"
        arr = np.zeros((12, 16, 2), dtype=np.float32)
        arr[3:9, 4:12, :] = 1
        nib.save(nib.Nifti1Image(arr, np.diag([1., 2., 3., 1.])), mask)
        for phase in phases:
            img = tmp_path / f"p{i}_{phase}.nii.gz"
            nib.save(nib.Nifti1Image(arr * 100, np.diag([1., 2., 3., 1.])), img)
            rows.append(dict(patient_id=f"MCT:p{i}", tumor_type="HCC" if i < 3 else "ICC",
                             dataset="MCT", phase=phase, image_path=str(img), mask_path=str(mask)))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    cfg = {"project": {"seed": 2023}, "classifier": {"class_names": ["HCC", "ICC"]},
           "data": {"manifest": str(manifest)}}
    return cfg, phases


def export_args(tmp, phases):
    return argparse.Namespace(manifest=None, phases=phases, phase="portal", kfold=3,
                              dump_splits=str(tmp / "splits.json"), out=str(tmp / "raw"),
                              preprocessed_dir=str(tmp / "pre"), copy=False)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_channels_folds_and_strict_collect(tmp_path, cohort, channels):
    cfg, phases = cohort
    selected = ["portal"] if channels == 1 else phases[:channels]
    args = export_args(tmp_path, selected)
    mod = script("export_nnunet")
    mod.export(args, cfg)
    raw = Path(args.out)
    assert len(list((raw / "imagesTr").glob("*.nii.gz"))) == 6 * channels
    splits = json.loads((tmp_path / "pre/splits_final.json").read_text())
    vals = [c for s in splits for c in s["val"]]
    assert len(vals) == len(set(vals)) == 6
    for i, split in enumerate(splits):
        assert not set(split["train"]) & set(split["val"])
        assert len(list((raw / "oof" / f"fold{i}").glob("*.nii.gz"))) == len(split["val"]) * channels
    args.pred_dir, args.work_dir = str(tmp_path / "pred"), str(tmp_path / "work")
    with pytest.raises(FileNotFoundError):
        mod.collect(args, cfg)
    for i, split in enumerate(splits):
        dest = Path(args.pred_dir) / f"fold{i}"
        dest.mkdir(parents=True)
        for cid in split["val"]:
            nib.save(nib.load(raw / "labelsTr" / f"{cid}.nii.gz"), dest / f"{cid}.nii.gz")
    mod.collect(args, cfg)
    summary = json.loads((Path(args.work_dir) / "summary.json").read_text())
    assert summary["mean_std"]["dice"]["mean"] == 1.0


def test_export_rejects_missing_and_misaligned_phase(tmp_path, cohort):
    cfg, phases = cohort
    df = pd.read_csv(cfg["data"]["manifest"])
    image = df.iloc[0]["image_path"]
    nib.save(nib.Nifti1Image(np.zeros((12, 16, 2)), np.eye(4)), image)
    with pytest.raises(ValueError, match="geometry"):
        script("export_nnunet").export(export_args(tmp_path, phases[:3]), cfg)
    df = df.iloc[1:]
    df.to_csv(cfg["data"]["manifest"], index=False)
    with pytest.raises(ValueError, match="expected exactly one"):
        script("export_nnunet").export(export_args(tmp_path, phases[:3]), cfg)


def test_native_prediction_cache_and_split_validation(tmp_path, cohort):
    cfg, phases = cohort
    args = export_args(tmp_path, phases[:3])
    script("export_nnunet").export(args, cfg)
    records = prepare_cache(args.out, tmp_path / "cache")
    splits = script("train_seg_dual").read_splits(tmp_path / "pre/splits_final.json", records)
    cid = splits[0]["val"][0]
    ds = SliceDataset(records, [cid])
    x, y = ds[0]
    assert x.shape == (3, 256, 256) and y.shape == (1, 256, 256)
    model = torch.nn.Conv2d(3, 1, 1)
    with torch.no_grad():
        model.weight.fill_(10)
        model.bias.fill_(-10)
    pred, ref = predict_volume(model, records[cid], "cpu", batch_size=1)
    assert pred.shape == ref.shape and pred.dtype == np.uint8
    assert native_dice(pred, ref) > 0.9
    splits[0]["train"].append(cid)
    (tmp_path / "bad.json").write_text(json.dumps(splits))
    with pytest.raises(ValueError, match="overlapping"):
        script("train_seg_dual").read_splits(tmp_path / "bad.json", records)


def test_dual_encoder_backward():
    torch.set_num_threads(2)
    # Reduced depth exercises every scale and attention path without downloading weights.
    model = DualSegmenter(pretrained=False, depths=(1, 1, 1, 1), cnn_name="convnext_atto")
    x = torch.randn(1, 3, 256, 256)
    target = torch.zeros(1, 1, 256, 256)
    target[:, :, 90:140, 90:140] = 1
    logits = model(x)
    assert logits.shape == target.shape
    loss = segmentation_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.stem.weight.grad is not None
    assert model.align[0].weight.grad.abs().sum() > 0
    assert next(model.cnn.parameters()).grad is not None


def test_training_checkpoint_oof_collection(tmp_path, cohort, monkeypatch):
    torch.set_num_threads(2)
    cfg, phases = cohort
    args = export_args(tmp_path, phases[:3])
    script("export_nnunet").export(args, cfg)
    records = prepare_cache(args.out, tmp_path / "cache")
    trainer = script("train_seg_dual")
    splits = trainer.read_splits(tmp_path / "pre/splits_final.json", records)
    # Exercise checkpoint selection and OOF routing cheaply, independently of model size.
    monkeypatch.setattr(trainer, "DualSegmenter", lambda **kw: torch.nn.Conv2d(3, 1, 1))
    args.work_dir = str(tmp_path / "work")
    args.dataset = args.out
    args.fold, args.no_pretrained = None, True
    args.workers, args.batch_size, args.epochs, args.patience = 0, 2, 1, 1
    args.lr, args.weight_decay, args.pos_weight, args.threshold = 1e-4, 1e-4, 500, 0.5
    trainer.train(args, splits, records, torch.device("cpu"), 42)
    trainer.predict(args, splits, records, torch.device("cpu"))
    trainer.collect(args, splits, records)
    with (Path(args.work_dir) / "oof_masks/pred_masks.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6 and len({r["patient_id"] for r in rows}) == 6
    for row in rows:
        assert row["patient_id"].replace(":", "_") in splits[int(row["fold"])]["val"]
    comparison = script("compare_segmentation").compare([args.work_dir, args.work_dir])
    assert len(comparison) == 2 and comparison[0]["patients"] == 6
    rows[0]["fold"] = (int(rows[0]["fold"]) + 1) % 3
    other = tmp_path / "other/oof_masks"
    other.mkdir(parents=True)
    with (other / "pred_masks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="assignments differ"):
        script("compare_segmentation").compare([args.work_dir, other.parent])
