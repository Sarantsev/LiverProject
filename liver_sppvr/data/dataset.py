from __future__ import annotations

import random
from typing import List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .preprocess import load_ct, load_mask


class MultiPhaseLiverDataset(Dataset):
    def __init__(
        self,
        manifest,                      # pandas.DataFrame (see manifest.load_manifest)
        class_names: Sequence[str],
        phases: Sequence[str] = ("non_contrast", "arterial", "portal", "delayed"),
        spatial_size: Sequence[int] = (32, 256, 256),
        hu_window: Sequence[float] = (-175, 250),
        patient_ids: Optional[Sequence[str]] = None,   # for the train/val/test split
        augment: bool = False,                         # 3D augmentations (train only)
        zoom: float = 0.0,                             # probability of a zoom-in crop [0..1] (0 = off)
        zoom_margin: float = 0.5,                      # context margin around the tumor (fraction of extent)
        normalize: str = "hu",                         # 'foreground' (SegVol) | 'hu'
        crop_foreground: bool = False,                 # crop the body before resize (SegVol-style)
    ):
        self.class_names = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.phases = list(phases)
        self.spatial_size = tuple(spatial_size)
        self.hu_window = tuple(hu_window)
        self.augment = augment
        self.zoom = zoom
        self.zoom_margin = zoom_margin
        self.normalize = normalize
        self.crop_foreground = crop_foreground

        df = manifest
        if patient_ids is not None:
            df = df[df["patient_id"].isin(set(patient_ids))]
        # group by patient
        self._patients: List[dict] = []
        for pid, grp in df.groupby("patient_id"):
            tumor_type = grp["tumor_type"].iloc[0]
            if tumor_type not in self.class_to_idx:
                continue
            phase_to_path = dict(zip(grp["phase"], grp["image_path"]))
            self._patients.append(dict(
                patient_id=pid,
                tumor_type=tumor_type,
                label=self.class_to_idx[tumor_type],
                mask_path=grp["mask_path"].iloc[0],
                phase_to_path=phase_to_path,
            ))

    def __len__(self) -> int:
        return len(self._patients)

    def __getitem__(self, idx: int) -> dict:
        rec = self._patients[idx]
        d, h, w = self.spatial_size

        # One fractional box per sample, applied synchronously to all phases + mask
        # (fractional coords -> works across phases of different native shape). Priority:
        # tumor zoom-in (with probability self.zoom) -> otherwise CropForeground by body.
        frac_box = None
        if self.zoom > 0 and random.random() < self.zoom:
            from .preprocess import bbox_fraction_from_mask
            frac_box = bbox_fraction_from_mask(rec["mask_path"], margin=self.zoom_margin)
        elif self.crop_foreground:
            from .preprocess import bbox_fraction_foreground
            # body is computed from the reference phase (portal -- the mask lives on it),
            # then the same box is applied to all phases
            ref = rec["phase_to_path"].get("portal") or next(iter(rec["phase_to_path"].values()), None)
            if ref is not None:
                frac_box = bbox_fraction_foreground(ref)

        phase_tensors, phase_present = [], []
        for phase in self.phases:
            path = rec["phase_to_path"].get(phase)
            if path is None:
                phase_tensors.append(torch.zeros(1, d, h, w))
                phase_present.append(0.0)
            else:
                phase_tensors.append(load_ct(path, self.hu_window, self.spatial_size,
                                             frac_box=frac_box, normalize=self.normalize))
                phase_present.append(1.0)
        phases = torch.stack(phase_tensors, dim=0)            # (P,1,D,H,W)

        mask = load_mask(rec["mask_path"], self.spatial_size, frac_box=frac_box)  # (1,D,H,W)
        if self.augment:
            from .augment import augment_multiphase
            phases, mask = augment_multiphase(phases, mask, clip01=(self.normalize == "hu"))
        return dict(
            phases=phases,
            mask=mask,
            label=torch.tensor(rec["label"], dtype=torch.long),
            phase_present=torch.tensor(phase_present),
            patient_id=rec["patient_id"],
        )


def collate_multiphase(batch: List[dict]) -> dict:
    return dict(
        phases=torch.stack([b["phases"] for b in batch], dim=0),       # (B,P,1,D,H,W)
        mask=torch.stack([b["mask"] for b in batch], dim=0),           # (B,1,D,H,W)
        label=torch.stack([b["label"] for b in batch], dim=0),         # (B,)
        phase_present=torch.stack([b["phase_present"] for b in batch], dim=0),
        patient_id=[b["patient_id"] for b in batch],
    )
