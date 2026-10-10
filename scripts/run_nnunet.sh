#!/usr/bin/env bash
# Stage A runbook: tumour segmentation with nnU-Net v2 on OUR patient folds (GPU box).
#
# nnU-Net v2 needs torch>=2 and cannot live in segvol_env (torch 1.13 for SegVol), so it
# gets its own venv (./nnunet_env). Everything that touches our code (export / collect)
# runs in segvol_env.
#
#   bash scripts/run_nnunet.sh setup          # one-off: create ./nnunet_env with nnunetv2
#   bash scripts/run_nnunet.sh export         # manifest -> nnUNet_raw + splits_final.json (our folds)
#   bash scripts/run_nnunet.sh plan           # nnUNetv2_plan_and_preprocess (+ integrity check)
#   bash scripts/run_nnunet.sh train 0 0      # fold 0 on GPU 0   (train-all: folds 0,2,4 -> GPU0; 1,3 -> GPU1)
#   bash scripts/run_nnunet.sh predict-oof    # each fold predicts ITS validation cases -> work/segA/oof/fold{i}
#   bash scripts/run_nnunet.sh collect        # -> work/segA/oof_masks/pred_masks.csv + native Dice summary
#
# Env overrides: TRAINER (default nnUNetTrainer_250epochs), DATASET_ID (501), NNENV, SEGVOL_PY.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export nnUNet_raw="${nnUNet_raw:-$ROOT/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$ROOT/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$ROOT/nnUNet_results}"
DATASET_ID="${DATASET_ID:-501}"
DATASET="Dataset${DATASET_ID}_MCTLiver"
TRAINER="${TRAINER:-nnUNetTrainer_250epochs}"
CONFIG="3d_fullres"
NNENV="${NNENV:-$ROOT/nnunet_env}"
SEGVOL_PY="${SEGVOL_PY:-$ROOT/../segvol_env/bin/python}"
WORK="$ROOT/work/segA"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$WORK"

cmd="${1:-help}"
case "$cmd" in
  setup)
    # Prefer python3.10: it is the version the rest of the project is proven on, and some
    # nnU-Net dependencies still lack wheels for very new interpreters.
    PYBIN="${PYBIN:-}"
    if [ -z "$PYBIN" ]; then
      for c in python3.11 python3.10 python3; do
        command -v "$c" >/dev/null 2>&1 && { PYBIN="$c"; break; }
      done
    fi
    echo "interpreter: $PYBIN ($("$PYBIN" --version 2>&1))"
    [ -d "$NNENV" ] && echo "note: $NNENV exists; reusing it (rm -rf it to start clean)"
    "$PYBIN" -m venv "$NNENV" 2>/dev/null || true
    # The GPU box has a flaky link to PyPI -> retry hard, and never abort on the pip bump.
    PIPFLAGS="--retries 10 --timeout 120"
    # shellcheck disable=SC2086
    "$NNENV/bin/pip" install $PIPFLAGS -U pip wheel || echo "pip bump failed -- continuing"
    # shellcheck disable=SC2086
    "$NNENV/bin/pip" install $PIPFLAGS torch torchvision --index-url https://download.pytorch.org/whl/cu121
    # shellcheck disable=SC2086
    "$NNENV/bin/pip" install $PIPFLAGS nnunetv2
    "$NNENV/bin/python" -c "import torch, nnunetv2; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
    # Which shortened-schedule trainers this nnU-Net build actually ships (the default
    # nnUNetTrainer runs 1000 epochs per fold, which is far too slow here).
    "$NNENV/bin/python" - <<'PY'
import os, nnunetv2.training.nnUNetTrainer as t
names = set()
for root, _, files in os.walk(os.path.dirname(t.__file__)):
    for f in files:
        if f.startswith("nnUNetTrainer") and f.endswith(".py"):
            names.add(f[:-3])
epochs = sorted(n for n in names if "poch" in n)
print("trainers with a shortened schedule:", epochs or "(none found -- use TRAINER=nnUNetTrainer)")
PY
    ;;
  export)
    cd "$ROOT" && PYTHONPATH=. "$SEGVOL_PY" scripts/export_nnunet.py --config configs/default.yaml \
      --out "$nnUNet_raw/$DATASET" --preprocessed-dir "$nnUNet_preprocessed/$DATASET" \
      --dump-splits "$ROOT/work/splits_seg.json"
    ;;
  plan)
    "$NNENV/bin/nnUNetv2_plan_and_preprocess" -d "$DATASET_ID" --verify_dataset_integrity -c "$CONFIG"
    test -f "$nnUNet_preprocessed/$DATASET/splits_final.json" || { echo "splits_final.json missing -- run export first"; exit 1; }
    ;;
  train)
    fold="${2:?fold index}"; gpu="${3:-0}"
    CUDA_VISIBLE_DEVICES="$gpu" "$NNENV/bin/nnUNetv2_train" "$DATASET_ID" "$CONFIG" "$fold" -tr "$TRAINER" --npz
    ;;
  train-all)
    ( for f in 0 2 4; do CUDA_VISIBLE_DEVICES=0 "$NNENV/bin/nnUNetv2_train" "$DATASET_ID" "$CONFIG" "$f" -tr "$TRAINER" --npz; done ) \
      > "$WORK/train_gpu0.log" 2>&1 &
    ( for f in 1 3;   do CUDA_VISIBLE_DEVICES=1 "$NNENV/bin/nnUNetv2_train" "$DATASET_ID" "$CONFIG" "$f" -tr "$TRAINER" --npz; done ) \
      > "$WORK/train_gpu1.log" 2>&1 &
    echo "training folds 0,2,4 on GPU0 and 1,3 on GPU1 in background; logs: $WORK/train_gpu*.log"
    ;;
  predict-oof)
    for f in 0 1 2 3 4; do
      gpu=$(( f % 2 ))
      CUDA_VISIBLE_DEVICES="$gpu" "$NNENV/bin/nnUNetv2_predict" -i "$nnUNet_raw/$DATASET/oof/fold$f" \
        -o "$WORK/oof/fold$f" -d "$DATASET_ID" -c "$CONFIG" -f "$f" -tr "$TRAINER"
    done
    ;;
  collect)
    cd "$ROOT" && PYTHONPATH=. "$SEGVOL_PY" scripts/export_nnunet.py --config configs/default.yaml --collect \
      --out "$nnUNet_raw/$DATASET" --preprocessed-dir "$nnUNet_preprocessed/$DATASET" \
      --pred-dir "$WORK/oof" --work-dir "$WORK"
    ;;
  *)
    sed -n '2,15p' "${BASH_SOURCE[0]}"
    ;;
esac
