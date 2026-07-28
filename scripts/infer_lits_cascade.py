"""Honest fully-automatic tumor segmentation on LiTS via SegVol-style zoom-out-zoom-in
with a NATIVE-RESOLUTION sliding-window fine stage.

The zoom-in ROI is derived from the model's own COARSE PREDICTION, not from the
ground-truth mask -- no GT is used to place the crop (fully automatic, no leakage).

Pipeline per volume (replica of SegVol inference):
  1. zoom-out (coarse): resize the whole volume to spatial_size, segment with a TEXT
     prompt -> coarse mask over the whole volume (upsampled to native for scoring).
  2. ROI: bounding box around the coarse PREDICTION's foreground (+ margin).
  3. zoom-in (fine): crop the NATIVE-resolution volume to that ROI and run a SLIDING
     WINDOW (window = spatial_size, gaussian blending) over it -- this restores
     resolution instead of squashing the ROI into one (32,256,256) resize.
  4. back-fill: place the fine mask back into the native coarse grid -> final mask.

All three metrics are scored at NATIVE resolution vs the native GT:
  * dice_coarse   -- single coarse pass upsampled to native (no cascade)
  * dice_fine_roi -- fine sliding-window mask vs native GT inside the auto ROI
  * dice_full     -- back-filled final vs full-volume native GT (HONEST fully-auto number)

NB: the coarse pass segments from TEXT, so the checkpoint must be trained with a text
prompt (train_lits.py --prompt text). A box-trained checkpoint gives a weak coarse mask.

Example:
    CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
      python scripts/infer_lits_cascade.py --ckpt cv_lits_text/best.pth \
        --manifest data/lits_manifest.csv --margin 0.3 --overlap 0.5
"""
from __future__ import annotations

import argparse
import gc
import sys

import numpy as np
import torch
import torch.nn.functional as F

from liver_sppvr.data import load_manifest
from liver_sppvr.data.preprocess import (_bbox_fraction, _load_nifti_dhw,
                                         foreground_norm)
from liver_sppvr.train.build import build_segvol_multitask, load_config, set_seed
from liver_sppvr.utils.device import resolve_device


def _dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    p = (pred > 0.5).float()
    t = (gt > 0.5).float()
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    return float((2 * inter + 1e-6) / (denom + 1e-6))


def _normalize(arr: np.ndarray, normalize: str, hu) -> np.ndarray:
    if normalize == "foreground":
        return foreground_norm(arr)
    lo, hi = float(hu[0]), float(hu[1])
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)


def _frac_bounds(frac, shape):
    """Fractional box -> integer [lo,hi) index bounds per axis for `shape` (D,H,W)."""
    d0, h0, w0, d1, h1, w1 = frac
    out = []
    for f0, f1, n in ((d0, d1, shape[0]), (h0, h1, shape[1]), (w0, w1, shape[2])):
        i0 = max(0, min(n - 1, int(round(f0 * n))))
        i1 = max(i0 + 1, min(n, int(round(f1 * n))))
        out.append((i0, i1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="text-prompt-trained checkpoint (best.pth)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default=None, help="fallback if the ckpt has no embedded config")
    ap.add_argument("--text", default="liver tumor", help="text prompt for both passes")
    ap.add_argument("--margin", type=float, default=0.3, help="ROI margin around coarse foreground")
    ap.add_argument("--min-size", type=float, default=0.1, help="min ROI fraction per axis")
    ap.add_argument("--overlap", type=float, default=0.5, help="sliding-window overlap (fine stage)")
    ap.add_argument("--sw-batch", type=int, default=2, help="sliding-window batch size")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N volumes (0 = all)")
    args = ap.parse_args()

    from monai.inferers import sliding_window_inference

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
    amp = getattr(device, "type", str(device)) == "cuda"

    def predictor(win: torch.Tensor) -> torch.Tensor:
        """win: (B,1,*spatial) -> logits (B,1,*spatial). Encode + text-prompted segment."""
        with torch.cuda.amp.autocast(enabled=amp):
            emb = model.encode(win)
            logits = model.segment(emb, win.shape[-3:], text=[args.text] * win.shape[0])
        return logits.float()

    man = load_manifest(args.manifest)
    portal = man[man["phase"] == "portal"] if "portal" in set(man["phase"]) else man
    rows = portal.drop_duplicates("patient_id").to_dict("records")
    if args.limit:
        rows = rows[:args.limit]
    print(f"volumes: {len(rows)} | spatial={spatial} normalize={norm} margin={args.margin} "
          f"overlap={args.overlap} text='{args.text}'")

    d_coarse, d_fine_roi, d_full, n_empty = [], [], [], 0
    for i, r in enumerate(rows):
        img_arr, _ = _load_nifti_dhw(r["image_path"])          # native (D,H,W)
        gt_arr, _ = _load_nifti_dhw(r["mask_path"])
        native = img_arr.shape
        img_arr = _normalize(img_arr, norm, hu)
        gt_full = torch.from_numpy((gt_arr > 0).astype(np.float32))    # native GT (D,H,W)

        # 1. coarse: whole volume resized to spatial
        img_c = F.interpolate(torch.from_numpy(img_arr)[None, None], size=spatial,
                              mode="trilinear", align_corners=False)
        with torch.no_grad():
            prob_c = torch.sigmoid(predictor(img_c.to(device)))[0, 0].cpu()   # (spatial)
        mask_c_native = F.interpolate((prob_c > 0.5).float()[None, None], size=native,
                                      mode="nearest")[0, 0]                    # native
        d_coarse.append(_dice(mask_c_native, gt_full))

        # 2. ROI from the COARSE PREDICTION (not GT), in fractional coords
        idx = torch.nonzero(prob_c > 0.5, as_tuple=True)
        if idx[0].numel() == 0:
            n_empty += 1
            d_fine_roi.append(0.0)
            d_full.append(d_coarse[-1])          # nothing to refine -> coarse is final
            del img_arr, img_c, prob_c, mask_c_native, gt_full; gc.collect()
            continue
        frac = _bbox_fraction([a.numpy() for a in idx], spatial, args.margin, args.min_size)

        # 3. zoom-in: crop the NATIVE volume to the ROI, sliding-window at native resolution
        (d0, d1), (h0, h1), (w0, w1) = _frac_bounds(frac, native)
        roi = torch.from_numpy(img_arr[d0:d1, h0:h1, w0:w1])[None, None].to(device)
        with torch.no_grad():
            logits_f = sliding_window_inference(
                roi, roi_size=spatial, sw_batch_size=args.sw_batch, predictor=predictor,
                overlap=args.overlap, mode="gaussian")
        fine = (torch.sigmoid(logits_f)[0, 0] > 0.5).float().cpu()            # native ROI
        gt_roi = gt_full[d0:d1, h0:h1, w0:w1]
        d_fine_roi.append(_dice(fine, gt_roi))

        # 4. back-fill fine into the native coarse grid, score full volume
        final = mask_c_native.clone()
        final[d0:d1, h0:h1, w0:w1] = fine
        d_full.append(_dice(final, gt_full))

        del img_arr, img_c, prob_c, mask_c_native, roi, logits_f, fine, final, gt_full
        gc.collect()
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(rows)}] means: coarse={np.mean(d_coarse):.4f} "
                  f"fine_roi={np.mean(d_fine_roi):.4f} full={np.mean(d_full):.4f}")

    def ms(x):
        a = np.array(x, dtype=float)
        return a.mean(), a.std()
    print("\n===== LiTS fully-auto cascade, native-res sliding-window (per-volume) =====")
    for name, x in (("dice_coarse (single pass)", d_coarse),
                    ("dice_fine_roi (auto-ROI, sliding window)", d_fine_roi),
                    ("dice_full  (HONEST end-to-end fully-auto)", d_full)):
        mu, sd = ms(x)
        print(f"  {name:<42} {mu:.4f} +/- {sd:.4f}")
    print(f"  coarse produced NO foreground on {n_empty}/{len(rows)} volumes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
