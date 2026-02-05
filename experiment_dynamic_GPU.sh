#!/usr/bin/env bash
set -euo pipefail

# Usage: ./benchmark_dynamic_splits.sh <num_runs>
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <num_runs>"
    exit 1
fi

RUNS="$1"

BINARY="./bazel-bin/examples/train_oblique_forest"

# List of datasets to benchmark
DATASETS=(
    "/home/ubuntu/projects/dataset/100000x4096.csv"
)

DTG=$(date -u +"%Y%m%dT%H%M%SZ")
OUTDIR="/home/ubuntu/projects/results"

# ===== CREATE SINGLE OUTPUT FILE =====
CSV_FILE="$OUTDIR/results_benchmark_dynamic_threads_${DTG}.csv"
echo "dataset,numerical_split_type,run_gpu_accel,num_threads,num_runs,valid_runs,avg_tree_ms,avg_total_s_final" > "$CSV_FILE"

echo "==============================="
echo "Output file: $CSV_FILE"
echo "==============================="

# Configurations
SPLITS=(
    "Dynamic Random Histogram"
    # "Dynamic Random Histogram"
    # "Exact"
    # "Random"
)

GPU_ACCEL=(
    true
    # false
    # false
    # false
)

NUM_THREADS=(16 8 4 2 1)

# Extract function (same as before)
extract_time() {
    awk '
        /Train[[:space:]]+tree[[:space:]]+[0-9]+\/[0-9]+/ {
            if (match($0, /tree[[:space:]]+([0-9]+)\/([0-9]+)/)) {
                x_part = substr($0, RSTART, RLENGTH)
                split(x_part, nums, /[[:space:]\/]+/)
                current_tree = nums[2]
                total_trees = nums[3]
                
                if (current_tree == total_trees) {
                    if (match($0, /total:[0-9a-z\.]+/)) {
                        total_str = substr($0, RSTART + 6, RLENGTH - 6)
                        
                        if (match(total_str, /([0-9]+)h([0-9]+)m([0-9\.]+)s/)) {
                            split(total_str, parts, /[hms]/)
                            hours = parts[1]
                            minutes = parts[2]
                            seconds = parts[3]
                            final_total_s = hours * 3600 + minutes * 60 + seconds
                        }
                        else if (match(total_str, /([0-9]+)m([0-9\.]+)s/)) {
                            split(total_str, parts, /[ms]/)
                            minutes = parts[1]
                            seconds = parts[2]
                            final_total_s = minutes * 60 + seconds
                        }
                        else if (match(total_str, /([0-9\.]+)s/)) {
                            gsub(/s/, "", total_str)
                            final_total_s = total_str
                        }
                    }
                }
            }
        }

        /[Tt]ree[[:space:]]+[0-9]+[[:space:]]+training[[:space:]]+time[[:space:]]*:/ {
            if (match($0, /[0-9]+(\.[0-9]+)?[[:space:]]*ms/)) {
                val = substr($0, RSTART, RLENGTH)
                gsub(/[[:space:]]*ms/, "", val)
                tree_sum += val
                tree_count++
            }
        }

        END {
            if (tree_count > 0)
                avg_tree_ms = tree_sum / tree_count
            else
                avg_tree_ms = ""
            print avg_tree_ms, final_total_s
        }
    '
}

###############################################
# MAIN LOOP: iterate over datasets
###############################################
for DATASET in "${DATASETS[@]}"; do

    DATASET_NAME=$(basename "$DATASET" .csv)

    echo
    echo "==============================="
    echo "Benchmarking dataset: $DATASET"
    echo "==============================="

    for idx in "${!SPLITS[@]}"; do
        split="${SPLITS[$idx]}"
        accel="${GPU_ACCEL[$idx]}"

        echo "=== numerical_split_type=\"$split\" | run_gpu_accel=$accel ==="

        for num_threads in "${NUM_THREADS[@]}"; do
            echo "  --- num_threads=$num_threads ---"

            total_tree_ms=0
            total_total_s=0
            count=0

            for ((i=1; i<=RUNS; i++)); do
                echo "  Run $i/$RUNS"

                output="$(
                    $BINARY \
                        --input_mode csv \
                        --label_col target \
                        --train_csv "$DATASET" \
                        --numerical_split_type "$split" \
                        --run_gpu_accel="$accel" \
                        --tree_depth -1 \
                        --num_threads "$num_threads" \
                        --num_trees 1000 2>&1 | tee >(grep --line-buffered "Train tree" >&2)
                )"

                read avg_tree_ms final_total_s <<< "$(printf "%s" "$output" | extract_time)"

                if [[ -z "$avg_tree_ms" ]]; then
                    echo "    Warning: could not extract training time"
                    continue
                fi

                echo "    avg_tree_ms    = $avg_tree_ms ms"
                echo "    final_total_s  = $final_total_s s"

                total_tree_ms=$(awk -v a="$total_tree_ms" -v b="$avg_tree_ms" 'BEGIN{printf "%.6f", a+b}')
                total_total_s=$(awk -v a="$total_total_s" -v b="$final_total_s" 'BEGIN{printf "%.6f", a+b}')

                count=$((count + 1))
            done

            if [[ $count -gt 0 ]]; then
                avg_tree_ms_final=$(awk -v s="$total_tree_ms" -v n="$count" 'BEGIN{printf "%.6f", s/n}')
                avg_total_s_final=$(awk -v s="$total_total_s" -v n="$count" 'BEGIN{printf "%.6f", s/n}')

                echo "  Averages over $count runs (${num_threads} threads):"
                echo "    avg_tree_ms    = $avg_tree_ms_final ms"
                echo "    avg_total_s    = $avg_total_s_final s"
            else
                echo "  No valid runs for this configuration"
                avg_tree_ms_final=""
                avg_total_s_final=""
            fi

            # ===== APPEND TO SINGLE FILE (NOT CREATE NEW) =====
            echo "${DATASET_NAME},\"${split}\",${accel},${num_threads},${RUNS},${count},${avg_tree_ms_final},${avg_total_s_final}" >> "$CSV_FILE"

            echo
        done
    done
done

echo
echo "==============================="
echo "All benchmarks complete."
echo "Results saved to: $CSV_FILE"
echo "==============================="