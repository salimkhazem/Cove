#!/usr/bin/env bash
# Pilot (K2 gate): random identity split (exact A1), 3 encoders, 4 datasets.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
GPU=$1; shift
export CUDA_VISIBLE_DEVICES=$GPU
for spec in "$@"; do
  IFS=: read -r ds enc ms <<< "$spec"
  python scripts/run_cells.py --dataset "$ds" --encoder "$enc" --ms ${ms//,/ } --reps 100 --split random --tag ${TAG:-pilot2}
done
