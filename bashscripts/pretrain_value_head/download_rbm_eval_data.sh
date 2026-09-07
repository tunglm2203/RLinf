#!/bin/bash
# Download the processed robometer datasets for the 3 standard eval groups (18 datasets) from the
# robometer HF dataset repo. Selective mirror of robometer/scripts/download_processed_datasets.sh
# (which pulls the ENTIRE repo, ~all 34 datasets) — here we fetch only the 18 we evaluate on.
#
# The dataset list is sourced from vr_value_head/rbm_eval_groups.py, which in turn reads robometer's
# dataset_category.py (rbm-1m-id / rbm-1m-ood / libero_pi0 eval splits) — so it stays in sync.
#
#   ROBOMETER_PROCESSED_DATASETS_PATH=/mnt/workspace2/all_dataset/robometer_dataset/processed_datasets \
#     bash bashscripts/pretrain_value_head/download_rbm_eval_data.sh
set -euo pipefail

VENV="${VENV:-/opt/venv/egoverse}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG="$REPO/vr_value_head"
DEST="${ROBOMETER_PROCESSED_DATASETS_PATH:-${RBM_PROCESSED_DATASETS_PATH:-/mnt/workspace2/all_dataset/robometer_dataset/processed_datasets}}"
HF_REPO="${HF_REPO:-robometer/processed_datasets}"

[ -x "$VENV/bin/python" ] || { echo "ERROR: venv missing: $VENV" >&2; exit 1; }
ulimit -n 65535 || true                                   # HF snapshot opens many files (robometer does this too)

# the 18 dataset names for the 3 eval groups
mapfile -t DATASETS < <("$VENV/bin/python" -c "import sys; sys.path.insert(0,'$PKG'); from rbm_eval_groups import ALL_DATASETS; print('\n'.join(ALL_DATASETS))")
echo "=== downloading ${#DATASETS[@]} robometer eval datasets ==="
echo "  repo=$HF_REPO -> $DEST"

INCLUDES=()
for d in "${DATASETS[@]}"; do INCLUDES+=(--include "${d}/*"); done
"$VENV/bin/huggingface-cli" download "$HF_REPO" --repo-type dataset --local-dir "$DEST" "${INCLUDES[@]}"

echo "=== done. present datasets: ==="
for d in "${DATASETS[@]}"; do
  n=$(ls "$DEST/$d/frames"/*.npz 2>/dev/null | wc -l)
  printf "  %6s eps  %s\n" "$n" "$d"
done
