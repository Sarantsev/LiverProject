"""Patient-level splits shared by EVERY pipeline (baselines, ours, nnU-Net export).

The paper's comparisons are apples-to-apples only if every model sees the same folds:
always build `labels_by_patient` with `labels_from_manifest` (same patient order as
MultiPhaseLiverDataset), call `make_kfold_splits` with the config seed, and `dump_splits`
in each pipeline so the JSON files can be diffed byte-for-byte.
"""
from __future__ import annotations

import json
import random


def labels_from_manifest(manifest, class_names) -> dict:
    """{patient_id: label_idx} in the order MultiPhaseLiverDataset enumerates patients
    (pandas groupby -> sorted patient_id). Patients whose tumor_type is not in
    class_names are skipped, exactly as the dataset does."""
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    out = {}
    for pid, grp in manifest.groupby("patient_id"):
        t = grp["tumor_type"].iloc[0]
        if t in class_to_idx:
            out[pid] = class_to_idx[t]
    return out


def stratified_patient_split(labels_by_patient: dict, val_frac: float = 0.2, seed: int = 2023):
    """{patient_id: label} -> (train_ids, val_ids), stratified by class."""
    rng = random.Random(seed)
    by_cls: dict = {}
    for pid, y in labels_by_patient.items():
        by_cls.setdefault(y, []).append(pid)
    train, val = [], []
    for y, pids in by_cls.items():
        pids = pids[:]; rng.shuffle(pids)
        k = max(1, int(round(len(pids) * val_frac))) if len(pids) > 1 else 0
        val += pids[:k]; train += pids[k:]
    return train, val


def make_kfold_splits(labels_by_patient: dict, k: int = 5, seed: int = 2023):
    """Stratified k-fold on patients -> list of (train_ids, val_ids). Each patient is
    validated exactly once; class proportions are preserved per fold (round-robin)."""
    rng = random.Random(seed)
    by_cls: dict = {}
    for pid, y in labels_by_patient.items():
        by_cls.setdefault(y, []).append(pid)
    folds_val = [[] for _ in range(k)]
    for y, pids in by_cls.items():
        pids = pids[:]; rng.shuffle(pids)
        for i, pid in enumerate(pids):
            folds_val[i % k].append(pid)
    all_ids = list(labels_by_patient.keys())
    return [([p for p in all_ids if p not in set(va)], va) for va in folds_val]


def dump_splits(splits, path: str) -> None:
    """Write the folds as JSON with sorted ids -> two pipelines can be `diff`-ed."""
    payload = [{"fold": i, "train": sorted(tr), "val": sorted(va)}
               for i, (tr, va) in enumerate(splits)]
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
