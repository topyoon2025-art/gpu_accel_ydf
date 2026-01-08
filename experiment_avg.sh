#!/usr/bin/env bash
set -euo pipefail

RUNS="${1:?Usage: $0 <runs>}"                       # repetitions for every (dataset, split_type)

BIN=./bazel-bin/examples/train_oblique_forest
CSV_DIR=/home/ubuntu/projects/gpu_accel_ydf/yggdrasil-oblique-forests

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

# Split modes we want to test
SPLIT_TYPES=("Exact" "Random" "Equal Width")
GPU_MODES=(0 1)   # <-- CPU + GPU

COMMON_ARGS=(
  --input_mode           csv
  --max_num_projections  100
  --num_trees            1
  --label_col            target
  --num_threads          1
  --tree_depth           2
  --histogram_num_bins   255
  --computation_method   0
  #--GPU_usage            1
)

OUTFILE="results_all_splits.csv"
echo "gpu_usage,split_type,dataset,rows,avg_train_s,avg_prep_s,avg_model_s,avg_accuracy,successful_runs" \
  > "$OUTFILE"

###############################################################################
for GPU in "${GPU_MODES[@]}"; do
  echo
  echo "#################################################################"
  echo "###  GPU_usage = $GPU"
  echo "#################################################################"

  for ST in "${SPLIT_TYPES[@]}"; do
    echo
    echo "==================================================="
    echo "Split type: $ST"
    echo "---------------------------------------------------"

    for csv in "${CSV_FILES[@]}"; do
      echo
      echo "Dataset: $(basename "$csv")"

      sum_train=0 sum_prep=0 sum_model=0 sum_acc=0
      ok_runs=0

      set +e
      for ((r=1; r<=RUNS; ++r)); do
        printf "  Run %2d/%d ...\n" "$r" "$RUNS"
        log=$(mktemp)

        "$BIN" --train_csv "$csv" \
               --numerical_split_type "$ST" \
               --GPU_usage "$GPU" \
               "${COMMON_ARGS[@]}" 2>&1 | tee "$log"

        train=$(grep -oP 'Training time:\s*\K[0-9.]+'                     "$log" | tail -1)
        prep=$( grep -oP 'Total Data and Label Prep Time:\s*\K[0-9.]+'    "$log" | tail -1)
        acc=$(  grep -oP 'Final OOB metrics: accuracy:\K[0-9.]+'          "$log" | tail -1)
        rm -f "$log"

        if [[ -z $train || -z $prep || -z $acc ]]; then
          echo "    ↳ could not parse output, ignoring this run"
          continue
        fi

        model=$(awk -v t="$train" -v p="$prep" 'BEGIN{print t-p}')

        sum_train=$(awk -v s="$sum_train" -v v="$train" 'BEGIN{print s+v}')
        sum_prep=$( awk -v s="$sum_prep"  -v v="$prep"  'BEGIN{print s+v}')
        sum_model=$(awk -v s="$sum_model" -v v="$model" 'BEGIN{print s+v}')
        sum_acc=$(  awk -v s="$sum_acc"   -v v="$acc"   'BEGIN{print s+v}')
        ((ok_runs++))
      done
      set -e

      if (( ok_runs == 0 )); then
        echo "  !! All $RUNS runs failed – skipping dataset"
        continue
      fi

      avg_train=$(awk -v s="$sum_train" -v n="$ok_runs" 'BEGIN{print s/n}')
      avg_prep=$( awk -v s="$sum_prep"  -v n="$ok_runs" 'BEGIN{print s/n}')
      avg_model=$(awk -v s="$sum_model" -v n="$ok_runs" 'BEGIN{print s/n}')
      avg_acc=$(   awk -v s="$sum_acc"  -v n="$ok_runs" 'BEGIN{print s/n}')

      rows=$(basename "$csv"); rows=${rows%%x*}

      echo "$GPU,$ST,$csv,$rows,$avg_train,$avg_prep,$avg_model,$avg_acc,$ok_runs" >> "$OUTFILE"
    done
  done
done

echo
echo "All experiments finished. Results stored in $OUTFILE"