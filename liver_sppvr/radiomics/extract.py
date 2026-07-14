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
            "PyRadiomics is not installed. Install it: pip install pyradiomics "
            "(see requirements.txt). It should be present in the GPU box environment.")


def _get_extractor(params: Optional[str]):
    _require()
    if params:
        return featureextractor.RadiomicsFeatureExtractor(params)
    ext = featureextractor.RadiomicsFeatureExtractor()
    ext.enableAllFeatures()
    return ext


def _filter_features(result: dict) -> dict:
    """Drop diagnostic fields (diagnostics_*), keep only numeric features."""
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
    """Extract features from a NIfTI image and mask. Returns {feature_name: value}."""
    extractor = _get_extractor(params)
    result = extractor.execute(str(image_path), str(mask_path))
    return _filter_features(result)


def extract_from_arrays(
    image: np.ndarray,
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    params: Optional[str] = None,
) -> dict:
    """Extract features from numpy arrays (image, mask) with the given spacing."""
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
    """Walk the manifest and collect features from a single phase per patient.

    Returns a pandas.DataFrame indexed by patient_id, with radiomics feature columns
    + tumor_type/label. Convenient as input for boosting or for fusion with deep features.
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
