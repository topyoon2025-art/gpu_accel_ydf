#!/usr/bin/env bash
set -euo pipefail

# ============================
# 0. Argument
# ============================
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <num_runs>"
    exit 1
fi
RUNS="$1"

# ============================
# 1. Paths / files
# ============================
BINARY="./bazel-bin/examples/train_oblique_forest"
DATASETS=(
        # "/home/ubuntu/projects/dataset/524288x100.csv"
        # "/home/ubuntu/projects/dataset/100000x4096.csv"
        "/home/ubuntu/projects/dataset/1000000x4096.csv"
        "/home/ubuntu/projects/dataset/HIGGS1.csv"
        "/home/ubuntu/projects/dataset/SUSY1.csv"
        )

DTG=$(date -u +"%Y%m%dT%H%M%SZ")
OUTDIR="/home/ubuntu/projects/results"
mkdir -p "$OUTDIR"

CSV_FILE="$OUTDIR/results_benchmark_dynamic_datasets${DTG}.csv"
echo "dataset,numerical_split_type,run_gpu_accel,num_threads,num_runs,valid_runs,avg_total_s" \
  > "$CSV_FILE"

echo "==============================="
echo "Output file: $CSV_FILE"
echo "==============================="

# ============================
# 2. Grid
# ============================

SPLITS=(
       
        "Dynamic Random Histogram"  #Vectoriced with the compilation flag
        "Random" #Vectorized with the compilation flag
        )

GPU_ACCEL=(
           # true 
            false
            false
           # false
            )

NUM_THREADS=(16)

echo "Thread counts to test: ${NUM_THREADS[*]}"

# ============================
# 3. Extractor
# ============================
extract_final_time() {
  awk '
    /random_forest\.cc[[:space:]]+Training[[:space:]]+block[[:space:]]+took:/ {
        if (match($0, /took:[[:space:]]*([0-9hms\.]+)[[:space:]]*s?/, a)) {
            ts=a[1]
            if      (ts~/[0-9]+h[0-9]+m[0-9\.]+s/) {split(ts,p,/[hms]/); secs=p[1]*3600+p[2]*60+p[3]}
            else if (ts~/[0-9]+m[0-9\.]+s/)       {split(ts,p,/[ms]/);  secs=p[1]*60+p[2]}
            else                                  {gsub(/s/,"",ts);     secs=ts}
            print secs
        }
    }'
}

# ============================
# 4. Benchmark loop
# ============================
for DATASET in "${DATASETS[@]}"; do
  DATASET_NAME=$(basename "$DATASET" .csv)

  echo
  echo "==============================="
  echo "Benchmarking dataset: $DATASET"
  echo "==============================="

  for idx in "${!SPLITS[@]}"; do
    split="${SPLITS[$idx]}"
    accel="${GPU_ACCEL[$idx]}"

    echo "=== split=\"$split\" | gpu=$accel ==="

    for num_threads in "${NUM_THREADS[@]}"; do
      echo "  --- threads=$num_threads ---"
      export OMP_NUM_THREADS=$num_threads

      declare -i count=0
      total_s=0

      for ((i=1; i<=RUNS; i++)); do
        echo "  Run $i/$RUNS"

        # ----------------------------
        # Run binary safely
        # ----------------------------
        set +e
        set +o pipefail
        out="$(
          "$BINARY" \
            --input_mode csv \
            --label_col target \
            --train_csv "$DATASET" \
            --numerical_split_type "$split" \
            --run_gpu_accel="$accel" \
            --tree_depth -1 \
            --num_threads "$num_threads" \
            --num_trees 128 \
            2>&1 | tee >(grep --line-buffered "Train tree" >&2)
        )"
        status_bin=$?
        set -e
        set -o pipefail
        # We intentionally ignore $status_bin

        # ----------------------------
        # Extract timing safely
        # ----------------------------
        set +e
        final_s=$(printf "%s" "$out" | extract_final_time)
        status_extract=$?
        set -e

        if [[ $status_extract -ne 0 || -z ${final_s:-} ]]; then
          echo "    Warning: timing not found – skipping run"
          continue
        fi

        echo "    final_total_s = $final_s s"

        # ----------------------------
        # Accumulate
        # ----------------------------
        total_s=$(awk -v a="$total_s" -v b="$final_s" 'BEGIN{printf "%.6f", a+b}')
        count+=1
      done

      # ----------------------------
      # Compute averages
      # ----------------------------
      if ((count > 0)); then
        avg_s=$(awk -v s="$total_s" -v n="$count" 'BEGIN{printf "%.6f", s/n}')
        echo "  Average over $count runs: $avg_s s"
      else
        avg_s=""
      fi

      # ----------------------------
      # Write CSV row
      # ----------------------------
      echo "${DATASET_NAME},\"${split}\",${accel},${num_threads},${RUNS},${count},${avg_s}" \
        >> "$CSV_FILE"

      echo
    done
  done
done

echo
echo "==============================="
echo "All benchmarks complete."
echo "Results saved to: $CSV_FILE"
echo "==============================="
