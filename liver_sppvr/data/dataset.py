from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .preprocess import load_ct, load_mask

ROI_MODES = ("none", "body", "pred")


def load_pred_masks(csv_path: str) -> dict:
    """pred_masks.csv (patient_id, pred_mask_path) -> {patient_id: path}. These are the
    OUT-OF-FOLD predictions of the segmentation model: every patient's mask comes from a
    model that never trained on that patient."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return dict(zip(df["patient_id"].astype(str), df["pred_mask_path"].astype(str)))


class MultiPhaseLiverDataset(Dataset):
    """One item = one patient: all phases resampled to `spatial_size` + tumour mask + label.

    mask_override: {patient_id: path} -- use these masks (e.g. nnU-Net out-of-fold
        predictions) INSTEAD of the ground truth. Strict: a listed dataset patient
        without an override raises, so GT can never leak in silently.
    roi: 'none' | 'body' (crop the body by intensity, SegVol CropForeground) |
         'pred' (crop around the OVERRIDE mask -> the tumour fills the window; needs
         mask_override; never derived from GT).
    mask_noise: train-time probability of perturbing the mask (dilate/erode/shift/drop)
        so a classifier trained on masks stays robust to segmentation errors.
    clinical: {patient_id: np.float32 vector} -> item['clinical'].
    """
    def __init__(
        self,
        manifest,
        class_names: Sequence[str],
        phases: Sequence[str] = ("non_contrast", "arterial", "portal", "delayed"),
        spatial_size: Sequence[int] = (32, 256, 256),
        hu_window: Sequence[float] = (-175, 250),
        normalize: str = "foreground",
        spacing: Optional[Sequence[float]] = None,     # (D,H,W) mm: fixed-spacing window mode (Merlin)
        patient_ids: Optional[Sequence[str]] = None,
        augment: bool = False,
        roi: str = "body",
        roi_margin: float = 0.5,
        mask_override: Optional[dict] = None,
        mask_noise: float = 0.0,
        clinical: Optional[dict] = None,
    ):
        if roi not in ROI_MODES:
            raise ValueError(f"roi must be one of {ROI_MODES}, got {roi!r}")
        if roi == "pred" and not mask_override:
            raise ValueError("roi='pred' needs mask_override (predicted masks); "
                             "a GT-derived ROI is a leak and is not supported")
        self.class_names = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.phases = list(phases)
        self.spatial_size = tuple(spatial_size)
        self.hu_window = tuple(hu_window)
        self.normalize = normalize
        self.spacing = tuple(spacing) if spacing else None
        self.augment = augment
        self.roi, self.roi_margin = roi, roi_margin
        self.mask_override = mask_override
        self.mask_noise = mask_noise
        self.clinical = clinical
        self.clinical_dim = len(next(iter(clinical.values()))) if clinical else 0

        df = manifest
        if patient_ids is not None:
            df = df[df["patient_id"].isin(set(patient_ids))]
        self._patients: List[dict] = []
        for pid, grp in df.groupby("patient_id"):
            tumor_type = grp["tumor_type"].iloc[0]
            if tumor_type not in self.class_to_idx:
                continue
            if mask_override is not None and pid not in mask_override:
                raise KeyError(f"mask_override has no predicted mask for patient {pid!r}")
            self._patients.append(dict(
                patient_id=pid, tumor_type=tumor_type, label=self.class_to_idx[tumor_type],
                mask_path=(mask_override[pid] if mask_override is not None else grp["mask_path"].iloc[0]),
                phase_to_path=dict(zip(grp["phase"], grp["image_path"])),
            ))

    def __len__(self) -> int:
        return len(self._patients)

    def _frac_box(self, rec: dict):
        """One fractional box per patient, applied identically to every phase + mask."""
        if self.roi == "pred":
            from .preprocess import bbox_fraction_from_mask
            return bbox_fraction_from_mask(rec["mask_path"], margin=self.roi_margin)
        if self.roi == "body":
            from .preprocess import bbox_fraction_foreground
            ref = rec["phase_to_path"].get("portal") or next(iter(rec["phase_to_path"].values()), None)
            return bbox_fraction_foreground(ref) if ref is not None else None
        return None

    def __getitem__(self, idx: int) -> dict:
        rec = self._patients[idx]
        d, h, w = self.spatial_size
        frac_box = self._frac_box(rec)

        tensors, present = [], []
        for phase in self.phases:
            path = rec["phase_to_path"].get(phase)
            if path is None:
                tensors.append(torch.zeros(1, d, h, w)); present.append(0.0)
            else:
                tensors.append(load_ct(path, self.hu_window, self.spatial_size, frac_box=frac_box,
                                       normalize=self.normalize, spacing=self.spacing))
                present.append(1.0)
        phases = torch.stack(tensors, dim=0)                        # (P,1,D,H,W)
        mask = load_mask(rec["mask_path"], self.spatial_size, frac_box=frac_box, spacing=self.spacing)

        if self.augment:
            from .augment import augment_multiphase, perturb_mask
            phases, mask = augment_multiphase(phases, mask, clip01=(self.normalize == "hu"))
            if self.mask_noise > 0:
                mask = perturb_mask(mask, p=self.mask_noise)

        item = dict(phases=phases, mask=mask,
                    label=torch.tensor(rec["label"], dtype=torch.long),
                    phase_present=torch.tensor(present), patient_id=rec["patient_id"])
        if self.clinical is not None:
            vec = self.clinical.get(rec["patient_id"])
            item["clinical"] = (torch.from_numpy(vec).float() if vec is not None
                                else torch.zeros(self.clinical_dim))
        return item


def collate_multiphase(batch: List[dict]) -> dict:
    out = dict(
        phases=torch.stack([b["phases"] for b in batch], dim=0),        # (B,P,1,D,H,W)
        mask=torch.stack([b["mask"] for b in batch], dim=0),            # (B,1,D,H,W)
        label=torch.stack([b["label"] for b in batch], dim=0),
        phase_present=torch.stack([b["phase_present"] for b in batch], dim=0),
        patient_id=[b["patient_id"] for b in batch],
    )
    if "clinical" in batch[0]:
        out["clinical"] = torch.stack([b["clinical"] for b in batch], dim=0)   # (B,F)
    return out
