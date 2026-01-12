#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_bench.sh <num_runs>
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <num_runs>"
    exit 1
fi

RUNS="$1"

BINARY="./bazel-bin/examples/train_oblique_forest"
DATASET="/home/ubuntu/projects/dataset/1048576x100.csv"
OUTDIR="/home/ubuntu/projects/results"

mkdir -p "$OUTDIR"

CSV_FILE="$OUTDIR/results.csv"

# Write CSV header
echo "gpu_usage,split_type,num_runs,valid_runs,avg_ms" > "$CSV_FILE"

GPU_USAGES=(0 1)
SPLITS=("Equal Width" "Random" "Exact")

# Robust extractor: only extract the number AFTER the colon/equals
extract_time() {
    grep -oiE 'training[[:space:]]*time[^0-9]*[:=][[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]*ms' \
    | grep -oE '[0-9]+(\.[0-9]+)?'
}

for gpu in "${GPU_USAGES[@]}"; do
    for split in "${SPLITS[@]}"; do

        echo "=== GPU_usage=$gpu | split='$split' ==="

        total=0
        count=0

        for ((i=1; i<=RUNS; i++)); do
            echo "Run $i/$RUNS"

            # Capture output only (no log file)
            output="$(
                stdbuf -oL -eL \
                $BINARY \
                    --input_mode csv \
                    --max_num_projections 100 \
                    --num_trees 1 \
                    --label_col target \
                    --numerical_split_type "$split" \
                    --num_threads 1 \
                    --tree_depth 2 \
                    --train_csv "$DATASET" \
                    --GPU_usage "$gpu" \
                2>&1
            )"

            t=$(printf "%s" "$output" | extract_time)

            if [[ -z "$t" ]]; then
                echo "Warning: could not extract training time"
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

        # Append to CSV
        echo "$gpu,$split,$RUNS,$count,$avg" >> "$CSV_FILE"

        echo
    done
done