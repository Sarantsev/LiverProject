"""Train/predict the experimental dual encoder on exported nnU-Net patient folds."""
import argparse
import csv
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from liver_sppvr.data.segmentation import SliceDataset, native_dice, predict_volume, prepare_cache
from liver_sppvr.models.seg_dual import DualSegmenter, segmentation_loss


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def read_splits(path, records):
    with open(path) as f:
        splits = json.load(f)
    if len(splits) < 2:
        raise ValueError("At least two patient folds are required")
    seen = []
    for split in splits:
        tr, va = split["train"], split["val"]
        if not tr or not va or set(tr) & set(va):
            raise ValueError("Empty or overlapping train/validation sets")
        if len(set(tr)) != len(tr) or len(set(va)) != len(va):
            raise ValueError("Duplicate cases in splits")
        if set(tr) | set(va) != set(records):
            raise ValueError("Split cohort differs from exported dataset")
        seen.extend(va)
    if len(seen) != len(set(seen)) or set(seen) != set(records):
        raise ValueError("Every patient must be validated exactly once")
    return splits


def train(args, splits, records, device, seed):
    for fold, split in enumerate(splits):
        if args.fold is not None and fold != args.fold:
            continue
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)
        random.seed(seed + fold)
        fd = Path(args.work_dir) / f"fold{fold}"
        fd.mkdir(parents=True, exist_ok=True)
        if (fd / "best.pth").exists():
            raise FileExistsError(f"{fd}/best.pth exists; use a new work directory")
        model = DualSegmenter(pretrained=not args.no_pretrained).to(device)
        ds = SliceDataset(records, split["train"], augment=True)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed + fold),
                            pin_memory=device.type == "cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=1e-8)
        best, stale = -1.0, 0
        history = []
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = segmentation_loss(model(x), y, args.pos_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                opt.step()
                total += loss.item() * len(x)
            # Native-space patient means, not a mean over mostly empty 2D slices.
            scores = [native_dice(*predict_volume(model, records[cid], device, args.batch_size,
                                                  args.threshold)) for cid in split["val"]]
            score = float(np.mean(scores))
            scheduler.step(1 - score)
            history.append({"epoch": epoch + 1, "loss": total / len(ds), "native_dice": score})
            print(f"fold={fold} epoch={epoch + 1} loss={total / len(ds):.5f} native_dice={score:.5f}", flush=True)
            (fd / "history.json").write_text(json.dumps(history, indent=2))
            if score > best:
                best, stale = score, 0
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                            "native_dice": best, "args": vars(args), "split": split,
                            "phases": ["non_contrast", "arterial", "portal"]}, fd / "best.pth")
            else:
                stale += 1
                if stale >= args.patience:
                    break
        del model, opt, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()


def predict(args, splits, records, device):
    for fold, split in enumerate(splits):
        if args.fold is not None and fold != args.fold:
            continue
        path = Path(args.work_dir) / f"fold{fold}" / "best.pth"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["split"] != split:
            raise ValueError(f"{path}: checkpoint split differs from requested OOF split")
        model = DualSegmenter(pretrained=False).to(device)
        model.load_state_dict(ck["model"])
        dest = Path(args.work_dir) / "oof" / f"fold{fold}"
        dest.mkdir(parents=True, exist_ok=True)
        for cid in split["val"]:
            pred, ref = predict_volume(model, records[cid], device, args.batch_size, ck["args"]["threshold"])
            header = ref.header.copy()
            header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(pred, ref.affine, header), dest / f"{cid}.nii.gz")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def collect(args, splits, records):
    with (Path(args.dataset) / "pid_map.csv").open() as f:
        patients = {r["case_id"]: r["patient_id"] for r in csv.DictReader(f)}
    rows, folds = [], []
    for fold, split in enumerate(splits):
        values = []
        for cid in split["val"]:
            path = (Path(args.work_dir) / "oof" / f"fold{fold}" / f"{cid}.nii.gz").resolve()
            img, ref = nib.load(path), nib.load(records[cid]["mask"])
            if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
                raise ValueError(f"{cid}: prediction geometry differs")
            pred = np.asarray(img.dataobj) > 0
            dice = native_dice(pred, ref)
            values.append(dice)
            rows.append({"patient_id": patients[cid], "pred_mask_path": str(path), "fold": fold,
                         "dice": dice, "empty_pred": int(not pred.any())})
        folds.append({"fold": fold, "dice": float(np.mean(values)), "n": len(values)})
    work = Path(args.work_dir)
    dest = work / "oof_masks"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "pred_masks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scores = [r["dice"] for r in folds]
    summary = {"model": "convnext-small-cswin-2d", "kfold": len(splits), "per_fold": folds,
               "mean_std": {"dice": {"mean": float(np.mean(scores)), "std": float(np.std(scores))}},
               "per_patient": {"mean": float(np.mean([r["dice"] for r in rows])), "n": len(rows)}}
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    with (work / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *[f"fold{i}" for i in range(len(scores))], "mean", "std"])
        writer.writerow(["dice", *scores, np.mean(scores), np.std(scores)])
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("command", choices=["cache", "train", "train-all", "predict-oof", "collect"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=500)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.workers < 0:
        ap.error("Epochs/patience/batch size must be positive; workers nonnegative")
    if not 0 < args.threshold < 1:
        ap.error("Threshold must be between 0 and 1")
    records = prepare_cache(args.dataset, Path(args.work_dir) / "cache")
    splits = read_splits(args.splits, records)
    if args.fold is not None and not 0 <= args.fold < len(splits):
        ap.error("Fold index outside available splits")
    device = torch.device(args.device)
    if args.command in ("train", "train-all"):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        train(args, splits, records, device, cfg["project"]["seed"])
    elif args.command == "predict-oof":
        predict(args, splits, records, device)
    elif args.command == "collect":
        collect(args, splits, records)


if __name__ == "__main__":
    main()
