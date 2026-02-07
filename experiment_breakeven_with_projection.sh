#!/usr/bin/env bash
# Row-sweep benchmark for train_oblique_forest → wide CSV
# -----------------------------------------------------------------------------
set -u
set -o pipefail

BINARY="bazel-bin/examples/train_oblique_forest"   # adjust to your build path

# Fixed parameters
COLS=4096
LABEL="target"
DEPTH=2
THREADS=1
RUNS=1
NUM_TREES=1

# Projection sweep
PROJ_START=5      # inclusive
PROJ_END=6       # inclusive
PROJ_STEP=1

DTG=$(date -u +"%Y%m%dT%H%M%SZ")
OUT="/home/ubuntu/projects/results/row_sweep_results_$DTG.csv"

echo "rows,proj,Random_CPU,Random_GPU,Exact" >"$OUT"

# -----------------------------------------------------------------------------
extract_final_time() {
  awk '
    /Training block took:/ {
      if (match($0,/took:[[:space:]]*([0-9hms\.]+)/,a)) {
        t=a[1]
        if      (t~/h/) { split(t,p,/[hms]/); secs=p[1]*3600+p[2]*60+p[3] }
        else if (t~/m/) { split(t,p,/[ms]/);  secs=p[1]*60+p[2]          }
        else            { gsub(/s/,"",t);     secs=t                     }
        print secs
      }
    }'
}

# -----------------------------------------------------------------------------
# run_case rows split gpu proj trees
#   -> prints the average time or the string NA
run_case() {
  local rows=$1 split=$2 gpu=$3 proj=$4 trees=$5
  local total=0 count=0

  for (( i=1; i<=RUNS; i++ )); do
    output="$(
      "$BINARY" \
        --input_mode "trunk" \
        --label_col "$LABEL" \
        --numerical_split_type "$split" \
        --num_trees "$trees" \
        --num_threads "$THREADS" \
        --tree_depth "$DEPTH" \
        --run_gpu_accel="$gpu" \
        --max_num_projections "$proj" \
        --rows "$rows" \
        --cols "$COLS" \
        2>&1
    )"

    t=$(printf '%s' "$output" | extract_final_time)
    [[ -z $t ]] && continue

    total=$(awk -v a="$total" -v b="$t" 'BEGIN{print a+b}')
    ((count++))
  done

  if (( count == 0 )); then
    echo "NA"
  else
    awk -v s="$total" -v n="$count" 'BEGIN{printf "%.8f", s/n}'
  fi
}

# -----------------------------------------------------------------------------
for (( proj=PROJ_START; proj<=PROJ_END; proj+=PROJ_STEP )); do
  echo "======= PROJ = $proj ======="
  iter=0
  for (( rows=0; rows<=2000; rows+=100 )); do
    ((iter++))
    (( iter % 10 == 0 )) && echo "  progress: ${iter} rows done (rows=$rows)"

    time_rand_cpu=$(run_case "$rows" "Random" false "$proj" "$NUM_TREES")
    time_rand_gpu=$(run_case "$rows" "Random" true  "$proj" "$NUM_TREES")
    time_exact_cpu=$(run_case "$rows" "Exact"  false "$proj" "$NUM_TREES")

    printf '%s,%s,%s,%s,%s\n' \
           "$rows" "$proj" "$time_rand_cpu" "$time_rand_gpu" "$time_exact_cpu" >>"$OUT"
  done
done

echo "Sweep finished. Table written to $OUT"