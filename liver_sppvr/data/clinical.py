"""Per-patient clinical (tabular) features for early fusion into the classifier.

CSV format (built by scripts/build_clinical.py):
    patient_id, clin_sex, clin_age, clin_cirrhosis, clin_hepatitis, clin_chemo, [tumor_type]
Only objective variables belong here; radiological descriptors of the tumour are the
radiologist's diagnosis and would leak the label.
"""
from __future__ import annotations

import os


def load_clinical(csv_path, train_ids, exclude=()):
    """-> {patient_id: np.float32 vector} or None when csv_path is empty/missing.

    Features are z-scored with TRAIN-fold statistics only (no leakage); NaN -> 0.
    Call once per fold with that fold's train ids.

    exclude: column names to drop, e.g. ("clin_chemo",) for the ablation that removes the
    feature disclosing the primary tumour.
    """
    if not csv_path or not os.path.exists(csv_path):
        return None
    import pandas as pd
    df = pd.read_csv(csv_path).set_index("patient_id")
    drop = {"tumor_type", *exclude}
    feat_cols = [c for c in df.columns if c not in drop]
    missing = set(exclude) - set(df.columns)
    if missing:
        raise ValueError(f"cannot exclude {sorted(missing)}: not in {csv_path}")
    train_rows = df.loc[df.index.intersection(list(train_ids)), feat_cols]
    mu = train_rows.mean()
    sd = train_rows.std().replace(0, 1.0)
    norm = ((df[feat_cols] - mu) / sd).fillna(0.0)
    return {pid: norm.loc[pid].to_numpy(dtype="float32") for pid in norm.index}


def feat_dim(clinical) -> int:
    return len(next(iter(clinical.values()))) if clinical else 0
