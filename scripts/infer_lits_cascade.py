"""Honest fully-automatic tumor segmentation on LiTS via SegVol-style zoom-out-zoom-in.

The difference from the GT-zoom used elsewhere: the zoom-in ROI is derived from the
model's own COARSE PREDICTION, NOT from the ground-truth mask. No GT is used to place
the crop, so this can run on data without labels (fully automatic, no leakage).

Pipeline per volume (replica of SegVol inference):
  1. zoom-out (coarse): resize the whole volume to spatial_size, segment with a TEXT
     prompt -> coarse mask over the whole volume.
  2. ROI: bounding box around the coarse PREDICTION's foreground (+ margin).
  3. zoom-in (fine): crop the native volume to that ROI, resize, segment again -> fine mask.
  4. back-fill: place the fine mask back into the coarse grid -> final full-volume mask.

Reported (per-volume mean +/- std):
  * dice_coarse   -- single coarse pass (no cascade) vs full-volume GT
  * dice_fine_roi -- fine mask vs GT inside the auto ROI (delineation given auto-localization)
  * dice_full     -- back-filled final vs full-volume GT (HONEST end-to-end fully-auto number)

NB: the coarse pass segments from TEXT, so the checkpoint must be trained with a text
prompt (train_lits.py --prompt text). A box-trained checkpoint gives a weak coarse mask.
The metric is at spatial_size resolution (coarse depth), so it is a lower bound vs
native-resolution sliding-window methods -- it is not a challenge submission.

Example:
    CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
      python scripts/infer_lits_cascade.py --ckpt cv_lits_text/best.pth \
        --manifest data/lits_manifest.csv --margin 0.5
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from liver_sppvr.data import load_manifest
from liver_sppvr.data.preprocess import _bbox_fraction, load_ct, load_mask
from liver_sppvr.train.build import build_segvol_multitask, load_config, set_seed
from liver_sppvr.utils.device import resolve_device


def _dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    p = (pred > 0.5).float()
    t = (gt > 0.5).float()
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    return float((2 * inter + 1e-6) / (denom + 1e-6))


@torch.no_grad()
def _seg(model, img: torch.Tensor, device, text: str) -> torch.Tensor:
    """img: (1,D,H,W) -> probability map (D,H,W) on CPU."""
    out = model(image=img[None].to(device), return_seg=True, return_cls=False, seg_text=text)
    return torch.sigmoid(out["seg_logits"].float())[0, 0].cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="text-prompt-trained checkpoint (best.pth)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default=None, help="fallback if the ckpt has no embedded config")
    ap.add_argument("--text", default="liver tumor", help="text prompt for both passes")
    ap.add_argument("--margin", type=float, default=0.5, help="ROI margin around coarse foreground")
    ap.add_argument("--min-size", type=float, default=0.1, help="min ROI fraction per axis")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N volumes (0 = all)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config") or (load_config(args.config) if args.config else None)
    if cfg is None:
        raise SystemExit("no embedded config in ckpt and --config not given")
    cfg.setdefault("classifier", {})["extra_feat_dim"] = 0
    set_seed(cfg["project"]["seed"])
    device = resolve_device(args.device or cfg.get("device", "auto"))

    pcfg = cfg.get("preprocess", {})
    spatial = tuple(cfg["segvol"]["spatial_size"])
    hu = cfg["multiphase"]["hu_window"]
    norm = pcfg.get("normalize", "hu")

    model = build_segvol_multitask(cfg, device)
    missing, unexpected = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    print(f"loaded {args.ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    man = load_manifest(args.manifest)
    portal = man[man["phase"] == "portal"] if "portal" in set(man["phase"]) else man
    rows = portal.drop_duplicates("patient_id").to_dict("records")
    if args.limit:
        rows = rows[:args.limit]
    print(f"volumes: {len(rows)} | spatial={spatial} normalize={norm} margin={args.margin} "
          f"text='{args.text}'")

    d_coarse, d_fine_roi, d_full, n_empty = [], [], [], 0
    for i, r in enumerate(rows):
        img_path, mask_path = r["image_path"], r["mask_path"]

        # 1. coarse (whole volume)
        img_c = load_ct(img_path, hu, spatial, frac_box=None, normalize=norm)   # (1,D,H,W)
        prob_c = _seg(model, img_c, device, args.text)                          # (D,H,W)
        mask_c = (prob_c > 0.5)
        gt_full = load_mask(mask_path, spatial, frac_box=None)[0]               # (D,H,W)
        d_coarse.append(_dice(prob_c, gt_full))

        # 2. ROI from the COARSE PREDICTION (not GT)
        idx = torch.nonzero(mask_c, as_tuple=True)
        if idx[0].numel() == 0:
            n_empty += 1
            d_fine_roi.append(0.0)
            d_full.append(d_coarse[-1])            # nothing to refine -> coarse is final
            continue
        idx_np = [a.numpy() for a in idx]
        frac = _bbox_fraction(idx_np, spatial, args.margin, args.min_size)

        # 3. zoom-in (fine) on the ROI cropped from the NATIVE volume
        img_f = load_ct(img_path, hu, spatial, frac_box=frac, normalize=norm)
        prob_f = _seg(model, img_f, device, args.text)                          # (D,H,W) in ROI space
        gt_roi = load_mask(mask_path, spatial, frac_box=frac)[0]
        d_fine_roi.append(_dice(prob_f, gt_roi))

        # 4. back-fill fine into a copy of the coarse grid, score full volume
        D, H, W = spatial
        d0, h0, w0, d1, h1, w1 = frac
        z = [max(0, min(D - 1, int(round(d0 * D)))), max(0, min(H - 1, int(round(h0 * H)))),
             max(0, min(W - 1, int(round(w0 * W))))]
        z1 = [max(z[0] + 1, min(D, int(round(d1 * D)))), max(z[1] + 1, min(H, int(round(h1 * H)))),
              max(z[2] + 1, min(W, int(round(w1 * W))))]
        roi_shape = (z1[0] - z[0], z1[1] - z[1], z1[2] - z[2])
        fine_roi = F.interpolate((prob_f > 0.5).float()[None, None], size=roi_shape,
                                 mode="nearest")[0, 0]
        final = mask_c.float().clone()
        final[z[0]:z1[0], z[1]:z1[1], z[2]:z1[2]] = fine_roi
        d_full.append(_dice(final, gt_full))

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(rows)}] running means: coarse={np.mean(d_coarse):.4f} "
                  f"fine_roi={np.mean(d_fine_roi):.4f} full={np.mean(d_full):.4f}")

    def ms(x):
        a = np.array(x, dtype=float)
        return a.mean(), a.std()
    print("\n===== LiTS fully-auto cascade (per-volume) =====")
    for name, x in (("dice_coarse (single pass)", d_coarse),
                    ("dice_fine_roi (auto-ROI delineation)", d_fine_roi),
                    ("dice_full  (HONEST end-to-end)", d_full)):
        mu, sd = ms(x)
        print(f"  {name:<38} {mu:.4f} +/- {sd:.4f}")
    print(f"  coarse produced NO foreground on {n_empty}/{len(rows)} volumes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
