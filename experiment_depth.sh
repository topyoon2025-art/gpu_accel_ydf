#!/usr/bin/env bash
# capture_depth_times.sh
# ------------------------------------------------------------------
# Usage:  ./capture_depth_times.sh TREE_DEPTH  [csv1 csv2 ...]
# Runs all split types (Random, Equal Width, Exact) with GPU=0 and 1,
# extracts "Cumulative time spent at depth ..." lines, and saves them
# into depth_times_all.csv (dataset, split_type, gpu_usage, depth, ms).

set -euo pipefail

BIN=./bazel-bin/examples/train_oblique_forest
CSV_DIR=/home/ubuntu/projects/dataset

DEFAULT_DATASETS=(
  "$CSV_DIR/1048576x100.csv"
)

# ------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 TREE_DEPTH [csv1 csv2 ...]" >&2
  exit 1
fi
TREE_DEPTH="$1"
shift

DATASETS=("$@")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=("${DEFAULT_DATASETS[@]}")

# ------------------------------------------------------------------
SPLIT_TYPES=("Exact" "Random" "Equal Width")
GPU_USAGES=(0 1)

COMMON_ARGS=(
  --input_mode=csv
  --max_num_projections=100
  --num_trees=1
  --label_col=target
  --num_threads=1
  --tree_depth="$TREE_DEPTH"
)

OUTFILE="/home/ubuntu/projects/results/depth_times_all.csv"
echo "dataset,split_type,gpu_usage,depth,time_ms" > "$OUTFILE"

for GPU in "${GPU_USAGES[@]}"; do
  for SPLIT_TYPE in "${SPLIT_TYPES[@]}"; do
    echo
    echo "###############################################################"
    echo "Split-type: $SPLIT_TYPE   |   GPU_usage: $GPU   |   depth: $TREE_DEPTH"
    echo "###############################################################"

    # Build flag list that does *not* change per-dataset:
    RUNTIME_ARGS=("${COMMON_ARGS[@]}"
                  "--numerical_split_type=${SPLIT_TYPE}"
                  "--GPU_usage=${GPU}")

    for csv in "${DATASETS[@]}"; do
      echo
      echo "Running $(basename "$csv") ..."
      log=$(mktemp)

      # Run learner
      "$BIN" --train_csv="$csv" "${RUNTIME_ARGS[@]}" 2>&1 | tee "$log"

      # Extract timing lines
      grep -oP 'Cumulative time spent at depth\s+\K[0-9]+:\s+[0-9]+' "$log" |
      while read -r entry; do
        depth=${entry%%:*}
        time_ms=$(echo "${entry#*:}" | awk '{print $1}')
        echo "${csv##*/},\"$SPLIT_TYPE\",$GPU,$depth,$time_ms" >> "$OUTFILE"
      done

      rm -f "$log"
    done
  done
done

echo
echo "Finished. Results stored in $OUTFILE"