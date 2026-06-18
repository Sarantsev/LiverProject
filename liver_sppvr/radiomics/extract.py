from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    from radiomics import featureextractor  # type: ignore
    RADIOMICS_AVAILABLE = True
except Exception:
    featureextractor = None
    RADIOMICS_AVAILABLE = False


def _require():
    if not RADIOMICS_AVAILABLE:
        raise ImportError(
            "PyRadiomics не установлен. Установите: pip install pyradiomics "
            "(см. requirements.txt). На рабочем GPU-девайсе он должен быть в окружении.")


def _get_extractor(params: Optional[str]):
    _require()
    if params:
        return featureextractor.RadiomicsFeatureExtractor(params)
    ext = featureextractor.RadiomicsFeatureExtractor()
    ext.enableAllFeatures()
    return ext


def _filter_features(result: dict) -> dict:
    """Убрать диагностические поля (diagnostics_*), оставить только числовые признаки."""
    out = {}
    for k, v in result.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def extract_features(image_path: str, mask_path: str, params: Optional[str] = None) -> dict:
    """Извлечь признаки из NIfTI изображения и маски. Возвращает {feature_name: value}."""
    extractor = _get_extractor(params)
    result = extractor.execute(str(image_path), str(mask_path))
    return _filter_features(result)


def extract_from_arrays(
    image: np.ndarray,
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    params: Optional[str] = None,
) -> dict:
    """Извлечь признаки из numpy-массивов (image, mask) с заданным spacing."""
    import SimpleITK as sitk
    extractor = _get_extractor(params)
    img = sitk.GetImageFromArray(image.astype(np.float32))
    msk = sitk.GetImageFromArray((mask > 0).astype(np.uint8))
    img.SetSpacing(tuple(float(s) for s in spacing))
    msk.SetSpacing(tuple(float(s) for s in spacing))
    result = extractor.execute(img, msk, label=1)
    return _filter_features(result)


def batch_extract(
    manifest,
    params: Optional[str] = None,
    phase: str = "portal",
):
    """Пройти по манифесту и собрать признаки по одной фазе на пациента.

    Возвращает pandas.DataFrame: индекс patient_id, колонки — радиомические признаки
    + tumor_type/label. Удобно как вход для бустинга или для слияния с deep-фичами.
    """
    import pandas as pd
    rows, index = [], []
    sub = manifest[manifest["phase"] == phase]
    for pid, grp in sub.groupby("patient_id"):
        r = grp.iloc[0]
        feats = extract_features(r["image_path"], r["mask_path"], params=params)
        feats["tumor_type"] = r["tumor_type"]
        rows.append(feats); index.append(pid)
    df = pd.DataFrame(rows, index=index)
    df.index.name = "patient_id"
    return df
