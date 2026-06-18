"""CLI обучения multi-task (seg + cls). Single-device, config-driven.

Запуск на GPU-девайсе:
    python -m liver_sppvr.train.multitask --config configs/default.yaml

Проверка цикла на CPU без весов/данных:
    python -m liver_sppvr.train.multitask --config configs/default.yaml --dry-run
"""
from __future__ import annotations

import argparse
import math
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.device import resolve_device
from .build import load_config, set_seed, build_dryrun_model, build_segvol_multitask
from .engine import train_one_epoch, evaluate
from .losses import MultiTaskLoss


# --------- утилиты ---------
def stratified_patient_split(labels_by_patient: dict, val_frac: float = 0.2, seed: int = 2023):
    """labels_by_patient: {patient_id: label} -> (train_ids, val_ids), стратификация по классу."""
    import random as _r
    rng = _r.Random(seed)
    by_cls = {}
    for pid, y in labels_by_patient.items():
        by_cls.setdefault(y, []).append(pid)
    train, val = [], []
    for y, pids in by_cls.items():
        pids = pids[:]; rng.shuffle(pids)
        k = max(1, int(round(len(pids) * val_frac))) if len(pids) > 1 else 0
        val += pids[:k]; train += pids[k:]
    return train, val


def class_weights(labels, num_classes: int) -> torch.Tensor:
    cnt = Counter(labels)
    w = torch.tensor([1.0 / max(cnt.get(c, 0), 1) for c in range(num_classes)])
    return w / w.sum() * num_classes


def make_scheduler(optimizer, warmup: int, total: int):
    def fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        prog = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# --------- синтетический датасет для dry-run ---------
class _SyntheticDataset(Dataset):
    def __init__(self, n=8, n_phases=4, num_classes=5, spatial=(4, 16, 16)):
        self.n, self.p, self.k, self.s = n, n_phases, num_classes, spatial
    def __len__(self): return self.n
    def __getitem__(self, i):
        d, h, w = self.s
        return dict(
            phases=torch.rand(self.p, 1, d, h, w),
            mask=(torch.rand(1, d, h, w) > 0.7).float(),
            label=torch.tensor(i % self.k, dtype=torch.long),
            phase_present=torch.ones(self.p),
            patient_id=f"syn{i}",
        )


def _collate(batch):
    from ..data import collate_multiphase
    return collate_multiphase(batch)


# --------- основной запуск ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="синтетика + заглушка на CPU")
    ap.add_argument("--device", default=None, help="переопределить device из конфига")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(args.device or cfg.get("device", "auto"))
    num_classes = cfg["classifier"]["num_classes"]
    tcfg = cfg["train"]
    epochs = args.epochs or (2 if args.dry_run else tcfg["num_epochs"])
    print(f"device={device} | dry_run={args.dry_run} | epochs={epochs}")

    # данные
    if args.dry_run:
        train_ds = _SyntheticDataset(n=8, n_phases=len(cfg["multiphase"]["phases"]),
                                     num_classes=num_classes)
        val_ds = _SyntheticDataset(n=4, n_phases=len(cfg["multiphase"]["phases"]),
                                   num_classes=num_classes)
        model = build_dryrun_model(cfg).to(device)
        train_labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
    else:
        from ..data import load_manifest, MultiPhaseLiverDataset
        man = load_manifest(cfg["data"]["manifest"])
        full = MultiPhaseLiverDataset(
            man, class_names=cfg["classifier"]["class_names"],
            phases=cfg["multiphase"]["phases"],
            spatial_size=cfg["segvol"]["spatial_size"],
            hu_window=cfg["multiphase"]["hu_window"])
        labels_by_patient = {p["patient_id"]: p["label"] for p in full._patients}
        tr_ids, va_ids = stratified_patient_split(labels_by_patient, seed=cfg["project"]["seed"])
        train_ds = MultiPhaseLiverDataset(man, class_names=cfg["classifier"]["class_names"],
                                          phases=cfg["multiphase"]["phases"],
                                          spatial_size=cfg["segvol"]["spatial_size"],
                                          hu_window=cfg["multiphase"]["hu_window"],
                                          patient_ids=tr_ids)
        val_ds = MultiPhaseLiverDataset(man, class_names=cfg["classifier"]["class_names"],
                                        phases=cfg["multiphase"]["phases"],
                                        spatial_size=cfg["segvol"]["spatial_size"],
                                        hu_window=cfg["multiphase"]["hu_window"],
                                        patient_ids=va_ids)
        model = build_segvol_multitask(cfg, device)
        train_labels = [labels_by_patient[p] for p in tr_ids]

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                              collate_fn=_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                            collate_fn=_collate, num_workers=0)

    # лосс / оптимизатор
    cw = class_weights(train_labels, num_classes).to(device)
    loss_fn = MultiTaskLoss(seg_weight=tcfg["loss_weights"]["seg"],
                            cls_weight=tcfg["loss_weights"]["cls"],
                            class_weight=cw).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    scheduler = make_scheduler(optimizer, tcfg["warmup_epoch"], epochs)

    work_dir = tcfg["work_dir"]
    os.makedirs(work_dir, exist_ok=True)
    best = -1.0
    for epoch in range(epochs):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        ev = evaluate(model, val_loader, device, num_classes=num_classes)
        print(f"[epoch {epoch+1}/{epochs}] train_loss={tr['loss']:.4f} "
              f"seg={tr['seg_loss']:.4f} cls={tr['cls_loss']:.4f} | "
              f"val_dice={ev.get('dice', float('nan')):.4f} "
              f"acc={ev.get('accuracy', float('nan')):.4f} "
              f"macroF1={ev.get('macro_f1', float('nan')):.4f}")
        score = ev.get("macro_f1", ev.get("accuracy", 0.0))
        if score is not None and score > best:
            best = score
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg},
                       os.path.join(work_dir, "best.pth"))
    print(f"Готово. Лучший score={best:.4f}. Чекпойнт: {os.path.join(work_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
