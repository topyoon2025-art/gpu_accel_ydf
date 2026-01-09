#!/usr/bin/env bash
# ---------------------------------------------------------------------
# experiment_avg.sh  –  run the oblique‐forest trainer on many
#                            datasets / split types / GPU modes
#
# Usage:  ./experiment_avg.sh <runs>
# ---------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------
# number of repetitions
# ---------------------------------------------------------------------
RUNS="${1:?Usage: $0 <runs>}"

# ---------------------------------------------------------------------
# binaries & data
# ---------------------------------------------------------------------
BIN=./bazel-bin/examples/train_oblique_forest
CSV_DIR=/home/ubuntu/projects/dataset

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
GPU_MODES=(0 1)

COMMON_ARGS=(
  --input_mode           csv
  --max_num_projections  100
  --num_trees            1
  --label_col            target
  --num_threads          1
  --tree_depth           2
  --histogram_num_bins   255
  --computation_method   0
)

# ---------------------------------------------------------------------
# output CSV
# ---------------------------------------------------------------------
OUTFILE=/home/ubuntu/projects/results/results_all_splits.csv
echo "gpu_usage,split_type,dataset,rows,"\
"avg_train_ms,avg_prep_ms,avg_model_ms,avg_accuracy,successful_runs" \
  > "$OUTFILE"

###############################################################################
# main loops
###############################################################################
for GPU in "${GPU_MODES[@]}"; do
  echo
  echo "#################################################################"
  echo "###  GPU_usage = $GPU"
  echo "#################################################################"

  for SPLIT in "${SPLIT_TYPES[@]}"; do
    echo
    echo "==================================================="
    echo "Split type: $SPLIT"
    echo "---------------------------------------------------"

    for CSV in "${CSV_FILES[@]}"; do
      echo
      echo "Dataset: $(basename "$CSV")"

      # accumulation variables
      sum_train=0
      sum_prep=0
      sum_model=0
      sum_acc=0
      ok_runs=0

      set +e   # allow individual runs to fail parsing without aborting all
      for ((r=1; r<=RUNS; ++r)); do
        printf "  Run %2d/%d ...\n" "$r" "$RUNS"

        tmp_log=$(mktemp)

        "$BIN"  --train_csv "$CSV" \
                --numerical_split_type "$SPLIT" \
                --GPU_usage "$GPU" \
                "${COMMON_ARGS[@]}" 2>&1 | tee "$tmp_log"

        # -------------------------------------------------------------------
        # Parse all required numbers (latest occurrence on stdout)
        # -------------------------------------------------------------------
        train=$(grep -oP 'Training time\s*[:=]\s*\K[0-9.]+' "$tmp_log" | tail -1)

        cpu_load=$(grep -oP 'CPU Dataset Load Time taken\s*[:=]\s*\K[0-9.]+' "$tmp_log" | tail -1)
        gpu_prep=$(grep -oP 'GPU Data Prep Time taken\s*[:=]\s*\K[0-9.]+' "$tmp_log" | tail -1)
        cuda_warm=$(grep -oP 'CUDA Warmup Time taken\s*[:=]\s*\K[0-9.]+'     "$tmp_log" | tail -1)

        acc=$(grep -oP 'accuracy\s*[:=]\s*\K[0-9.]+' "$tmp_log" | tail -1)

        rm -f "$tmp_log"

        # default missing fields to 0
        cpu_load=${cpu_load:-0}
        gpu_prep=${gpu_prep:-0}
        cuda_warm=${cuda_warm:-0}

        # total prep = CPU load + GPU prep + CUDA warm-up
        prep=$(awk -v a="$cpu_load" -v b="$gpu_prep" -v c="$cuda_warm" \
                 'BEGIN{printf "%.6f", a+b+c}')

        # validate
        if [[ ! $train =~ ^[0-9.]+$ || ! $prep =~ ^[0-9.]+$ || ! $acc =~ ^[0-9.]+$ ]]; then
          echo "    ↳ could not parse output, ignoring this run"
          continue
        fi

        # model time = training − (cpu_load + gpu_prep + warmup)
        model=$(awk -v t="$train" -v p="$prep" 'BEGIN{printf "%.6f", t-p}')

        # accumulate
        sum_train=$(awk -v s="$sum_train" -v v="$train" 'BEGIN{print s+v}')
        sum_prep=$( awk -v s="$sum_prep"  -v v="$prep"  'BEGIN{print s+v}')
        sum_model=$(awk -v s="$sum_model" -v v="$model" 'BEGIN{print s+v}')
        sum_acc=$(  awk -v s="$sum_acc"   -v v="$acc"   'BEGIN{print s+v}')

        (( ok_runs++ ))
      done
      set -e   # restore strict-error behaviour

      if (( ok_runs == 0 )); then
        echo "  !! All $RUNS runs failed – skipping dataset"
        continue
      fi

      # averages
      avg_train=$(awk -v s="$sum_train" -v n="$ok_runs" 'BEGIN{printf "%.6f", s/n}')
      avg_prep=$( awk -v s="$sum_prep"  -v n="$ok_runs" 'BEGIN{printf "%.6f", s/n}')
      avg_model=$(awk -v s="$sum_model" -v n="$ok_runs" 'BEGIN{printf "%.6f", s/n}')
      avg_acc=$(   awk -v s="$sum_acc"  -v n="$ok_runs" 'BEGIN{printf "%.6f", s/n}')

      # number of rows = part before first ‘x’ in file name
      rows=$(basename "$CSV"); rows=${rows%%x*}

      # write CSV line
      echo "$GPU,$SPLIT,$CSV,$rows,$avg_train,$avg_prep,$avg_model,$avg_acc,$ok_runs" \
        >> "$OUTFILE"
    done
  done
done

echo
echo "All experiments finished. Results stored in $OUTFILE"