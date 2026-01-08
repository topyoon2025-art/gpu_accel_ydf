# #!/usr/bin/env bash
# set -euo pipefail

# # ---------------------------- user settings ----------------------------
# BIN=./bazel-bin/examples/train_oblique_forest      # compiled executable
# CSV_DIR=/home/ubuntu/projects/gpu_accel_ydf/yggdrasil-oblique-forests    # folder with the datasets

# CSV_FILES=(
#   "$CSV_DIR/100000x100.csv"
#   "$CSV_DIR/125000x100.csv"
#   "$CSV_DIR/250000x100.csv"
#   "$CSV_DIR/500000x100.csv"
#   "$CSV_DIR/1000000x100.csv"
# )

#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------
# Split-type = every word given to the script, default = “Exact”
SPLIT_TYPE="${*:-Exact}"

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

# -------- store each option in its own ARRAY element ------------------
COMMON_ARGS=(
  --input_mode           csv
  --max_num_projections  100
  --num_trees            1
  --label_col            target
  --numerical_split_type "$SPLIT_TYPE"
  --num_threads          1
  --tree_depth           2
)
# ---------------------------------------------------------------------

OUTFILE="/home/ubuntu/projects/results/results_${SPLIT_TYPE// /_}.csv"
echo "dataset,rows,train_time_s,data_prep_s,model_only_s,accuracy" > "$OUTFILE"

for csv in "${CSV_FILES[@]}"; do
    echo "Running on $csv with split type '$SPLIT_TYPE' ..."
    log=$(mktemp)

    # run learner and capture output
    "$BIN" --train_csv "$csv" "${COMMON_ARGS[@]}" 2>&1 | tee "$log"

    # ------------------------------------------------------------------
    # extract times (both already in SECONDS)
    train_time=$(grep -oP 'Training time:\s+\K[0-9.]+(?=\s*s)' \
                 "$log" | tail -1)
    prep_time=$( grep -oP 'Total Data and Label Prep Time:\s+\K[0-9.]+(?=\s*s)' \
                 "$log" | tail -1)
    accuracy=$(grep -oP 'Final OOB metrics: accuracy:\K[0-9.]+' \
                 "$log" | tail -1)

    if [[ -z $train_time || -z $prep_time || -z $accuracy ]]; then
        echo "Could not parse times for $csv – skipping" >&2
        rm -f "$log";  continue
    fi

    # pure model training time (seconds − seconds)
    model_only=$(awk -v t="$train_time" -v p="$prep_time" \
                      'BEGIN{printf "%.6f", t - p}')

    # Number of rows = part before first 'x'
    rows=$(basename "$csv"); rows=${rows%%x*}

    echo "$csv,$rows,$train_time,$prep_time,$model_only,$accuracy" >> "$OUTFILE"
    rm -f "$log"
done

echo "All runs finished. Results written to $OUTFILE"