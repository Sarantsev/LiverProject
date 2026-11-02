#!/usr/bin/env bash
# bash scripts/run_segmentation.sh portal|triphasic|fourphase|dual COMMAND [ARGS...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="${1:?experiment: portal|triphasic|fourphase|dual}"
shift
export TASK_CONFIG="${TASK_CONFIG:-$ROOT/configs/default.yaml}"
export WORK="${WORK:-$ROOT/work/seg_${experiment}}"
export SPLITS_OUT="$WORK/splits.json"
export KFOLD="${KFOLD:-5}"
case "$experiment" in
  portal) export DATASET_ID=501 INPUT_PHASES="portal" ;;
  triphasic) export DATASET_ID=502 INPUT_PHASES="non_contrast arterial portal" ;;
  fourphase) export DATASET_ID=503 INPUT_PHASES="non_contrast arterial portal delayed" ;;
  dual) export DATASET_ID=504 INPUT_PHASES="non_contrast arterial portal" ;;
  *) echo "Unknown experiment: $experiment" >&2; exit 2 ;;
esac
export DATASET="Dataset${DATASET_ID}_MCTLiver"
if [ "$experiment" = dual ] && [ "${1:-}" = setup ]; then
  PY="${SEG_PY:-$ROOT/nnunet_env/bin/python}"
  exec "$PY" -m pip install -r "$ROOT/requirements-seg.txt"
fi
if [ "$experiment" != dual ] || [ "${1:-}" = export ]; then
  exec bash "$ROOT/scripts/run_nnunet.sh" "$@"
fi
PY="${SEG_PY:-$ROOT/nnunet_env/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" "$ROOT/scripts/train_seg_dual.py" --config "$TASK_CONFIG" \
  --dataset "${nnUNet_raw:-$ROOT/nnUNet_raw}/$DATASET" \
  --splits "${nnUNet_preprocessed:-$ROOT/nnUNet_preprocessed}/$DATASET/splits_final.json" \
  --work-dir "$WORK" "$@"
