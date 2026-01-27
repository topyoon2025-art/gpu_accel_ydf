#!/usr/bin/env bash
set -euo pipefail

# Usage: ./benchmark_dynamic_splits.sh <num_runs>
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <num_runs>"
    exit 1
fi

RUNS="$1"

BINARY="./bazel-bin/examples/train_oblique_forest"
DATASET="/home/ubuntu/projects/dataset/100000x4096.csv"

# Extract dataset filename without path
DATASET_NAME=$(basename "$DATASET" .csv)

# Output CSV file
OUTDIR="/home/ubuntu/projects/results"
mkdir -p "$OUTDIR"
CSV_FILE="$OUTDIR/results_dynamic_${DATASET_NAME}.csv"

# Write CSV header
echo "dataset,numerical_split_type,run_gpu_accel,num_runs,valid_runs,avg_ms" > "$CSV_FILE"

# All configurations requested
SPLITS=(
    "Dynamic Equal Width Histogram"
    "Dynamic Random Histogram"
    "Dynamic Equal Width Histogram"
    "Dynamic Random Histogram"
    "Equal Width"
    "Random"
)

GPU_ACCEL=(
    true
    true
    false
    false
    false
    false
)

# Extract numeric ms value from:
#   Tree 0 training time: 9006.85 ms
extract_time() {
    grep -oiE 'training[[:space:]]*time[^0-9]*[:=][[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]*ms' \
    | grep -oE '[0-9]+(\.[0-9]+)?'
}

for idx in "${!SPLITS[@]}"; do
    split="${SPLITS[$idx]}"
    accel="${GPU_ACCEL[$idx]}"

    echo "=== numerical_split_type=\"$split\" | run_gpu_accel=$accel ==="

    total=0
    count=0

    for ((i=1; i<=RUNS; i++)); do
        echo "Run $i/$RUNS"

        output="$(
            $BINARY \
                --input_mode csv \
                --label_col target \
                --train_csv "$DATASET" \
                --numerical_split_type "$split" \
                --run_gpu_accel="$accel"
        )"

        t=$(printf "%s" "$output" | extract_time)

        if [[ -z "$t" ]]; then
            echo "  Warning: could not extract training time"
            continue
        fi

        echo "  time = $t ms"

        total=$(awk -v a="$total" -v b="$t" 'BEGIN{print a+b}')
        count=$((count + 1))
    done

    if [[ $count -gt 0 ]]; then
        avg=$(awk -v sum="$total" -v n="$count" 'BEGIN{print sum/n}')
        echo "Average Tree 0 training time = $avg ms"
    else
        echo "No valid runs for this configuration"
        avg=""
    fi

    # Append row to CSV
    echo "${DATASET_NAME},\"${split}\",${accel},${RUNS},${count},${avg}" >> "$CSV_FILE"

    echo
done