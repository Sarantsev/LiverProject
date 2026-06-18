from __future__ import annotations

import glob
import os
from typing import Callable, List, Optional, Sequence

MANIFEST_COLUMNS = [
    "patient_id",   # уникальный id пациента (сплит делается по нему — без утечки)
    "dataset",      # имя датасета-источника
    "tumor_type",   # метка класса (HCC/ICC/CRLM/BCLM/HH/...)
    "phase",        # фаза КТ (non_contrast/arterial/portal/delayed)
    "image_path",   # путь к NIfTI изображения данной фазы
    "mask_path",    # путь к NIfTI маски опухоли (общая для пациента)
]


def scan_by_dir(
    root: str,
    dataset: str,
    phases: Sequence[str] = ("non_contrast", "arterial", "portal", "delayed"),
    mask_name: str = "mask.nii.gz",
    image_ext: str = ".nii.gz",
) -> List[dict]:
    """Сканер типовой раскладки root/<tumor_type>/<patient_id>/<phase>.nii.gz."""
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
    """Собрать манифест и записать в CSV.

    record_fn: кастомный адаптер раскладки; если None — используется scan_by_dir.
    """
    import pandas as pd
    if record_fn is not None:
        records = record_fn(root)
    else:
        if root is None:
            raise ValueError("Нужен root или record_fn.")
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
        raise ValueError(f"В манифесте нет колонок: {missing}")
    if df.empty:
        raise ValueError("Манифест пуст.")
