#!/usr/bin/env bash

set -euo pipefail

# -----------------------------
# Number of runs from CLI arg
# -----------------------------
if [[ $# -ge 1 ]]; then
    RUNS="$1"
else
    RUNS=5   # default
fi

echo "Using RUNS = $RUNS"

BIN="bazel-bin/examples/train_oblique_forest"
CSV_DIR="/home/ubuntu/projects/dataset"

CSV_FILES=(
  "$CSV_DIR/32x100.csv"
  "$CSV_DIR/64x100.csv"
  "$CSV_DIR/128x100.csv"
  "$CSV_DIR/256x100.csv"
  "$CSV_DIR/512x100.csv"
  "$CSV_DIR/1024x100.csv"
  "$CSV_DIR/2048x100.csv"
  "$CSV_DIR/4096x100.csv"
  "$CSV_DIR/8192x100.csv"
  "$CSV_DIR/16384x100.csv"
  "$CSV_DIR/32768x100.csv"
  "$CSV_DIR/65536x100.csv"
  "$CSV_DIR/131072x100.csv"
  "$CSV_DIR/262144x100.csv"
  "$CSV_DIR/524288x100.csv"
  "$CSV_DIR/1048576x100.csv"
)

SPLIT_TYPES=("Exact" "Random" "Equal Width")

OUTCSV="/home/ubuntu/projects/results/benchmark_all_total.csv"

echo "dataset,split_type,metric,avg_value,runs" > "$OUTCSV"

run_test() {
    local dataset="$1"
    local split_type="$2"

    echo "Running ${split_type} on ${dataset} for ${RUNS} runs..."

    local total=0
    local count=0

    for ((i=1; i<=RUNS; i++)); do
        echo "  Run $i..."

        output="$(${BIN} \
            --input_mode csv \
            --max_num_projections 100 \
            --num_trees 1 \
            --label_col target \
            --num_threads 1 \
            --tree_depth 2 \
            --train_csv ${dataset} \
            --GPU_usage 1 \
            --histogram_num_bins 255 \
            --numerical_split_type "${split_type}" 2>&1)"

        # Show the timing line we are about to parse (one-time sanity check; you can comment this out later)
        echo "$output" | grep "GPU Total Time taken" || true

        # Extract numeric field from the line containing "GPU Total Time taken"
        # This picks the last token that looks like a number (digits + optional dot)
        value=$(echo "$output" | awk '
            /GPU Total Time taken/ {
                for (i=1; i<=NF; i++) {
                    if ($i ~ /^[0-9]+(\.[0-9]+)?$/) {
                        v = $i
                    }
                }
            }
            END { if (v != "") print v }
        ' | xargs)

        # Validate numeric
        if ! [[ "$value" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "Warning: Could not extract numeric GPU time on run $i"
            echo "Extracted: '$value'"
            echo "Full output:"
            echo "$output"
            continue
        fi

        total=$(echo "$total + $value" | bc -l)
        count=$((count + 1))
    done

    if [[ $count -eq 0 ]]; then
        echo "ERROR: No valid runs for ${dataset} with ${split_type}"
        return
    fi

    avg=$(echo "scale=6; $total / $count" | bc -l)

    echo "${dataset},${split_type},GPU Total Time taken,${avg},${count}" >> "$OUTCSV"

    echo "Done ${split_type} on ${dataset} (avg=${avg}, runs=${count})"
}

for split in "${SPLIT_TYPES[@]}"; do
    for DATA in "${CSV_FILES[@]}"; do
        run_test "$DATA" "$split"
    done
done

echo "Benchmark complete. CSV saved to ${OUTCSV}"