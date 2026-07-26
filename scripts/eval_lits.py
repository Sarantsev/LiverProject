"""Zero-shot / fine-tuned TUMOR-segmentation eval on LiTS (segmentation only).

Loads a trained checkpoint (work_dir/best.pth) and reports per-volume tumor Dice on a
LiTS manifest, for one or both prompt modes:
  box   -- box prompt derived from the GT tumor mask (semi-automatic; matches our 0.63-0.65)
  text  -- text prompt "liver tumor" (fully automatic; no localization hint)

The classifier head is bypassed (return_cls=False) -- LiTS has no tumor-type label, so
only segmentation is meaningful. Radiomics is disabled (extra_feat_dim=0) so a checkpoint
trained with radiomics fusion still loads (strict=False) and the seg path is unaffected.

Example (on the GPU box):
    CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
      python scripts/eval_lits.py --config configs/default.yaml \
        --ckpt work_dir/best.pth --manifest data/lits_manifest.csv \
        --prompt both --batch-size 2 --num-workers 6
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from liver_sppvr.data import MultiPhaseLiverDataset, collate_multiphase, load_manifest
from liver_sppvr.train.build import build_segvol_multitask, load_config, set_seed
from liver_sppvr.train.engine import _batch_to_device, _seg_prompt_kwargs
from liver_sppvr.utils.device import resolve_device


def _per_sample_dice(seg_logits, mask, thr: float = 0.5):
    """Return a (B,) numpy array of per-volume Dice (same formula as engine._dice_score)."""
    prob = torch.sigmoid(seg_logits.float())
    pred = (prob > thr).float()
    t = (mask > 0.5).float()
    inter = (pred * t).flatten(1).sum(1)
    denom = pred.flatten(1).sum(1) + t.flatten(1).sum(1)
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    return dice.detach().cpu().numpy()


@torch.no_grad()
def run_mode(model, loader, device, mode: str):
    model.eval()
    dices = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        seg_kw = _seg_prompt_kwargs(batch, mode, "liver tumor", bbox_shift=0)
        out = model(phases=batch["phases"], return_seg=True, return_cls=False, **seg_kw)
        if out.get("seg_logits") is not None:
            dices.extend(_per_sample_dice(out["seg_logits"], batch["mask"]).tolist())
    return np.array(dices, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="fallback config; by default the config saved INSIDE the checkpoint is used "
                         "(guarantees the architecture matches the weights)")
    ap.add_argument("--ckpt", required=True, help="path to best.pth (dict with key 'model'/'config')")
    ap.add_argument("--manifest", required=True, help="LiTS manifest CSV (from build_manifest.py)")
    ap.add_argument("--prompt", choices=["box", "text", "both"], default="both")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--zoom-eval", action="store_true",
                    help="zoom-in crop around the tumor on eval (match your best-run val setup)")
    args = ap.parse_args()

    # prefer the config saved inside the checkpoint -> the model is built exactly as trained
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config")
    if cfg is not None:
        print(f"using config embedded in {args.ckpt}")
    elif args.config:
        cfg = load_config(args.config)
        print(f"checkpoint has no embedded config -> using {args.config}")
    else:
        raise SystemExit("checkpoint has no embedded config and --config was not given")

    set_seed(cfg["project"]["seed"])
    device = resolve_device(args.device or cfg.get("device", "auto"))

    # NB: keep the embedded classifier dims (incl. radiomics extra_feat_dim) so the cls-head
    # weights load without a size mismatch. The classifier is never executed here -- we call
    # the model with return_cls=False -- so no radiomics input is needed at eval time.

    pcfg = cfg.get("preprocess", {})
    tcfg = cfg.get("train", {})
    zoom_eval = args.zoom_eval or tcfg.get("zoom_eval", False)
    man = load_manifest(args.manifest)
    ds = MultiPhaseLiverDataset(
        man,
        class_names=cfg["classifier"]["class_names"],
        phases=cfg["multiphase"]["phases"],
        spatial_size=cfg["segvol"]["spatial_size"],
        hu_window=cfg["multiphase"]["hu_window"],
        normalize=pcfg.get("normalize", "hu"),
        crop_foreground=pcfg.get("crop_foreground", False),
        zoom=(1.0 if zoom_eval else 0.0),
        zoom_margin=tcfg.get("zoom_margin", 0.5),
    )
    print(f"LiTS volumes: {len(ds)} | zoom_eval={zoom_eval} | "
          f"normalize={pcfg.get('normalize', 'hu')} crop_fg={pcfg.get('crop_foreground', False)}")
    if len(ds) == 0:
        raise SystemExit("empty dataset -- check the manifest / dummy tumor_type is in class_names")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_multiphase)

    model = build_segvol_multitask(cfg, device)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.ckpt}: missing={len(missing)} unexpected={len(unexpected)} "
          f"(cls-head mismatch is expected & harmless)")

    modes = ["box", "text"] if args.prompt == "both" else [args.prompt]
    print("\n===== LiTS tumor Dice =====")
    for mode in modes:
        d = run_mode(model, loader, device, mode)
        print(f"  prompt={mode:<4}  Dice = {d.mean():.4f} ± {d.std():.4f}  "
              f"(median {np.median(d):.4f}, n={len(d)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
