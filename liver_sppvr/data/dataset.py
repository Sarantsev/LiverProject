from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .preprocess import load_ct, load_mask


class MultiPhaseLiverDataset(Dataset):
    def __init__(
        self,
        manifest,                      # pandas.DataFrame (см. manifest.load_manifest)
        class_names: Sequence[str],
        phases: Sequence[str] = ("non_contrast", "arterial", "portal", "delayed"),
        spatial_size: Sequence[int] = (32, 256, 256),
        hu_window: Sequence[float] = (-175, 250),
        patient_ids: Optional[Sequence[str]] = None,   # для train/val/test сплита
    ):
        self.class_names = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.phases = list(phases)
        self.spatial_size = tuple(spatial_size)
        self.hu_window = tuple(hu_window)

        df = manifest
        if patient_ids is not None:
            df = df[df["patient_id"].isin(set(patient_ids))]
        # группировка по пациенту
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
        phase_tensors, phase_present = [], []
        for phase in self.phases:
            path = rec["phase_to_path"].get(phase)
            if path is None:
                phase_tensors.append(torch.zeros(1, d, h, w))
                phase_present.append(0.0)
            else:
                phase_tensors.append(load_ct(path, self.hu_window, self.spatial_size))
                phase_present.append(1.0)
        phases = torch.stack(phase_tensors, dim=0)            # (P,1,D,H,W)

        mask = load_mask(rec["mask_path"], self.spatial_size) # (1,D,H,W)
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
