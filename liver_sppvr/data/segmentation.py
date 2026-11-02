"""Memory-mapped axial slices cached on disk, preserving native NIfTI geometry."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def prepare_cache(dataset, cache):
    dataset, cache = Path(dataset), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    with (dataset / "experiment.json").open() as f:
        info = json.load(f)
    if info["phases"] != ["non_contrast", "arterial", "portal"]:
        raise ValueError("Dual encoder expects NC, arterial, portal in this order")
    records = {}
    for label in sorted((dataset / "labelsTr").glob("*.nii.gz")):
        cid = label.name[:-7]
        images = [dataset / "imagesTr" / f"{cid}_{i:04d}.nii.gz" for i in range(3)]
        sources = [*images, label]
        signature = [{"path": str(p.resolve()), "size": p.stat().st_size,
                      "mtime": p.stat().st_mtime_ns} for p in sources]
        stamp = cache / f"{cid}.json"
        xfile, yfile = cache / f"{cid}_x.npy", cache / f"{cid}_y.npy"
        ref = nib.load(label)
        if len(ref.shape) != 3 or nib.aff2axcodes(ref.affine)[2] not in ("S", "I"):
            raise ValueError(f"{cid}: expected axial NIfTI with slices along axis 2")
        if not (stamp.exists() and xfile.exists() and yfile.exists() and
                json.loads(stamp.read_text()) == signature):
            x = []
            for path in images:
                img = nib.load(path)
                if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
                    raise ValueError(f"{cid}: unregistered phase geometry")
                a = np.asarray(img.dataobj, dtype=np.float32)
                a = (np.clip(a, -15, 135) + 15) / 150
                x.append(a.transpose(2, 0, 1))
            xt = torch.from_numpy(np.stack(x, axis=1))
            yt = torch.from_numpy((np.asarray(ref.dataobj) > 0).astype(np.float32).transpose(2, 0, 1).copy())[:, None]
            np.save(xfile, F.interpolate(xt, (256, 256), mode="bilinear", align_corners=False).numpy())
            np.save(yfile, F.interpolate(yt, (256, 256), mode="nearest").numpy().astype(np.uint8))
            stamp.write_text(json.dumps(signature))
        records[cid] = {"x": str(xfile), "y": str(yfile), "mask": str(label), "depth": ref.shape[2]}
    if not records:
        raise ValueError("Empty exported dataset")
    return records


class SliceDataset(Dataset):
    def __init__(self, records, ids, augment=False):
        self.records, self.augment = records, augment
        self.index = [(cid, z) for cid in ids for z in range(records[cid]["depth"])]
        self._cid, self._x, self._y = None, None, None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cid, z = self.index[idx]
        if cid != self._cid:
            self._x = np.load(self.records[cid]["x"], mmap_mode="r")
            self._y = np.load(self.records[cid]["y"], mmap_mode="r")
            self._cid = cid
        x, y = torch.from_numpy(self._x[z].copy()), torch.from_numpy(self._y[z].copy()).float()
        if self.augment:
            for axis in (-1, -2):
                if torch.rand(()) < 0.5:
                    x, y = x.flip(axis), y.flip(axis)
        return x, y


@torch.no_grad()
def predict_volume(model, record, device, batch_size=8, threshold=0.5):
    model.eval()
    x = np.load(record["x"], mmap_mode="r")
    ref = nib.load(record["mask"])
    pred = np.empty(ref.shape, dtype=np.uint8)
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start:start + batch_size].copy()).to(device)
        prob = model(batch).sigmoid()
        prob = F.interpolate(prob, ref.shape[:2], mode="bilinear", align_corners=False)
        pred[:, :, start:start + len(batch)] = (prob[:, 0] >= threshold).cpu().numpy().transpose(1, 2, 0)
    return pred, ref


def native_dice(pred, ref):
    gt = np.asarray(ref.dataobj) > 0
    denom = int(pred.sum()) + int(gt.sum())
    return float(2 * np.logical_and(pred, gt).sum() / denom) if denom else 1.0
