#!/usr/bin/env bash

set -euo pipefail

BIN=bazel-bin/examples/train_oblique_forest
CSV_DIR=/home/ubuntu/projects/dataset

CSV_FILES=(
#   "$CSV_DIR/32x100.csv"
#   "$CSV_DIR/64x100.csv"
#   "$CSV_DIR/128x100.csv"
#   "$CSV_DIR/256x100.csv"
#   "$CSV_DIR/512x100.csv"
#   "$CSV_DIR/1024x100.csv"
#   "$CSV_DIR/2048x100.csv"
#   "$CSV_DIR/4096x100.csv"
#   "$CSV_DIR/8192x100.csv"
#   "$CSV_DIR/16384x100.csv"
#   "$CSV_DIR/32768x100.csv"
#   "$CSV_DIR/65536x100.csv"
#   "$CSV_DIR/131072x100.csv"
#   "$CSV_DIR/262144x100.csv"
#   "$CSV_DIR/524288x100.csv"
  "$CSV_DIR/1048576x100.csv"
)

SPLIT_TYPES=("Random" "Equal Width")

OUTCSV="/home/ubuntu/projects/results/benchmark_all.csv"

echo "dataset,split_type,metric,value" > "$OUTCSV"

run_test() {
    local dataset="$1"
    local split_type="$2"

    echo "Running ${split_type} on ${dataset}..."

    start_time=$(date +%s.%N)

    output="$(${BIN} \
        --input_mode csv \
        --max_num_projections 100 \
        --num_trees 1 \
        --label_col target \
        --num_threads 1 \
        --tree_depth 2 \
        --train_csv ${dataset} \
        --numerical_split_type "${split_type}" \
        --GPU_usage 1 2>&1)" 

    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)

    echo "$output" | grep -E "Time taken:" | while read -r line; do
        metric=$(echo "$line" | cut -d':' -f1 | xargs)
        value=$(echo "$line" | awk -F':' '{print $2}' | xargs)
        echo "${dataset},${split_type},${metric},${value}" >> "$OUTCSV"
    done

    echo "${dataset},${split_type},Wall Time,${elapsed}s" >> "$OUTCSV"

    echo "Done ${split_type} on ${dataset}"
}

for DATA in "${CSV_FILES[@]}"; do
    for split in "${SPLIT_TYPES[@]}"; do
        run_test "$DATA" "$split"
    done
done

echo "Benchmark complete. CSV saved to ${OUTCSV}"