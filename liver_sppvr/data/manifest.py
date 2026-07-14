from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence

MANIFEST_COLUMNS = [
    "patient_id",   # unique patient id (the split is done on it -- no leakage)
    "dataset",      # source dataset name
    "tumor_type",   # class label (HCC/ICC/CRLM/BCLM/HH/...)
    "phase",        # CT phase (non_contrast/arterial/portal/delayed)
    "image_path",   # path to the NIfTI image of this phase
    "mask_path",    # path to the tumor mask NIfTI (shared per patient)
]


def scan_by_dir(
    root: str,
    dataset: str,
    phases: Sequence[str] = ("non_contrast", "arterial", "portal", "delayed"),
    mask_name: str = "mask.nii.gz",
    image_ext: str = ".nii.gz",
) -> List[dict]:
    """Scanner for the standard layout root/<tumor_type>/<patient_id>/<phase>.nii.gz."""
    records: List[dict] = []
    for tumor_type in sorted(os.listdir(root)):
        type_dir = os.path.join(root, tumor_type)
        if not os.path.isdir(type_dir):
            continue
        for patient_id in sorted(os.listdir(type_dir)):
            pdir = os.path.join(type_dir, patient_id)
            if not os.path.isdir(pdir):
                continue
            mask_path = os.path.join(pdir, mask_name)
            for phase in phases:
                img_path = os.path.join(pdir, f"{phase}{image_ext}")
                if os.path.exists(img_path):
                    records.append(dict(
                        patient_id=f"{dataset}:{patient_id}",
                        dataset=dataset,
                        tumor_type=tumor_type,
                        phase=phase,
                        image_path=img_path,
                        mask_path=mask_path,
                    ))
    return records


def build_manifest(
    out_csv: str,
    root: Optional[str] = None,
    dataset: str = "dataset",
    record_fn: Optional[Callable[[str], List[dict]]] = None,
    **scan_kwargs,
) -> "pandas.DataFrame":
    """Build the manifest and write it to CSV.

    record_fn: custom layout adapter; if None, scan_by_dir is used.
    """
    import pandas as pd
    if record_fn is not None:
        records = record_fn(root)
    else:
        if root is None:
            raise ValueError("root or record_fn is required.")
        records = scan_by_dir(root, dataset=dataset, **scan_kwargs)
    df = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def load_manifest(csv_path: str) -> "pandas.DataFrame":
    import pandas as pd
    df = pd.read_csv(csv_path)
    validate_manifest(df)
    return df


def validate_manifest(df) -> None:
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if df.empty:
        raise ValueError("Manifest is empty.")
