#!/usr/bin/env python3
"""Run YDF with parallel-chrono, write per-tree-depth CSV (thread-pivoted)."""

from __future__ import annotations
import argparse, csv, os, re, subprocess, sys, time
from pathlib import Path
from pathlib import Path
import subprocess, sys, logging

import pandas as pd
import utils.utils as utils

log = logging.getLogger(__name__)


def get_args():
    # Get base parser as parent
    parent_parser = utils.get_base_parser()
    
    # Create this script's parser with the base as parent
    p = argparse.ArgumentParser(parents=[parent_parser])
    
    # Add script-specific arguments
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--save_log", action="store_true")
    
    # Override defaults if needed
    p.set_defaults(num_trees=5)  # This script uses 5 trees by default
    
    return p.parse_args()


# log parsing
TIMING_RX = re.compile(
    r"thread\s+(\d+)\s+tree\s+(\d+)\s+depth\s+(\d+)\s+"
    r"nodes\s+(\d+)\s+samples\s+(\d+)\s+"
    r"SampleProj\s+([0-9.eE+-]+)s\s+"
    r"ProjEval\s+([0-9.eE+-]+)s\s+"
    r"EvalProj\s+([0-9.eE+-]+)s"
)
# TRAIN_RX = re.compile(r"Training wall-time:\s*([0-9.eE+-]+)s")


def parse_parallel_chrono(log: str) -> pd.DataFrame:
    # ------------------------------------------------------
    # 1. collect the information from the log
    # ------------------------------------------------------
    rows = []
    for m in TIMING_RX.finditer(log):
        tid, tree, depth, nodes, samples, sp, pe, ep = m.groups()
        rows.append(dict(thread   = int(tid),
                         tree     = int(tree),
                         depth    = int(depth),
                         nodes    = int(nodes),
                         samples  = int(samples),
                         SampleProjection   = float(sp),
                         ProjectionEvaluate = float(pe),
                         EvaluateProjection = float(ep)))

    if not rows:
        raise ValueError("no parallel-chrono lines found in log")

    df = pd.DataFrame(rows)

    # ------------------------------------------------------
    # 2. build one block per thread
    # ------------------------------------------------------
    blocks = []

    for tid, g in df.groupby("thread", sort=True):
        g = g.sort_values(["tree", "depth"]).reset_index(drop=True)
        
        # Rename columns
        g = g.rename(columns={
            "samples": "Active Samples",
            "ProjectionEvaluate": "ApplyProjection"
        })
        
        # Drop the thread column
        g = g.drop(columns=["thread"])
        
        # Create a DataFrame with thread ID in first row, column names in second row
        thread_header = pd.DataFrame([[f"Thread {tid}"] + [""] * (len(g.columns) - 1)], 
                                   columns=g.columns)
        col_names = pd.DataFrame([g.columns.tolist()], columns=g.columns)
        
        # Stack: thread header, column names, then data
        g_with_headers = pd.concat([thread_header, col_names, g], ignore_index=True)
        
        blocks.append(g_with_headers)

    # ------------------------------------------------------
    # 3. concatenate blocks side-by-side, insert a blank
    #    column between two successive blocks
    # ------------------------------------------------------
    max_len = max(len(b) for b in blocks)
    gap = pd.DataFrame({"": [""] * max_len})

    padded = []
    for i, b in enumerate(blocks):
        padded.append(b.reindex(range(max_len)).fillna(""))
        if i + 1 < len(blocks):
            padded.append(gap)

    wide = pd.concat(padded, axis=1)

    return wide


def write_csv(left: pd.DataFrame, params: dict[str, object], path: str):
    right = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
    n = max(len(left), len(right))
    gap = pd.DataFrame({"": [""] * n, "  ": [""] * n})
    
    # Add headers for the params section
    params_with_headers = pd.concat([
        pd.DataFrame([["", "", "Parameter", "Value"]], columns=["", "  ", "Parameter", "Value"]),
        right
    ], ignore_index=True)
    
    # Adjust n to account for the extra header row
    n = max(len(left), len(params_with_headers))
    gap = pd.DataFrame({"": [""] * n, "  ": [""] * n})
    
    (left.reindex(range(n)).fillna("")
         .pipe(lambda l: pd.concat([l, gap,
                                    params_with_headers.reindex(range(n)).fillna("")], axis=1))
     ).to_csv(path, index=False, header=False, quoting=csv.QUOTE_MINIMAL)  # header=False


if __name__ == "__main__":
    utils.setup_signal_handlers()
    a = get_args()

    if not utils.build_binary(a, chrono_mode=True):
        print("❌ build failed", file=sys.stderr)
        sys.exit(1)

    exp = f"{a.sample_projection_mode} projections | {a.feature_split_type} | {a.numerical_split_type} | {a.experiment_name}"
    
    cmd = ["./bazel-bin/examples/train_oblique_forest",
           f"--num_trees={a.num_trees}",
           f"--feature_split_type={a.feature_split_type}",
           f"--compute_oob_performances=false"]
    
    if a.max_num_projections is not None:
        cmd.append(f"--max_num_projections={a.max_num_projections}")
    if a.projection_density_factor is not None:
        cmd.append(f"--projection_density_factor={a.projection_density_factor}")
    if a.num_threads is not None:
        cmd.append(f"--num_threads={a.num_threads}")
    if a.tree_depth is not None:
        cmd.append(f"--tree_depth={a.tree_depth}")

    cmd.append("--numerical_split_type=Exact"
               if (a.numerical_split_type == "Dynamic Random Histogramming" or a.numerical_split_type == "Dynamic Equal Width Histogramming")
               else f"--numerical_split_type={a.numerical_split_type}")
 

    # Use CSV filename (without extension) if using CSV input, otherwise use matrix dimensions
    if a.input_mode == "csv":
        csv_filename = Path(a.train_csv).stem  # Gets filename without extension
        dataset_name = csv_filename

        cmd += ["--input_mode=csv",
        f"--train_csv={a.train_csv}",
        f"--label_col={a.label_col}"]
    elif a.input_mode == "uniform" or a.input_mode == "trunk":
        dataset_name = f"{a.input_mode}_{a.rows}_x_{a.cols}"
        cmd += [f"--input_mode={a.input_mode}", f"--rows={a.rows}", f"--cols={a.cols}"]

    out_dir = Path("benchmarks/results/per_function_timing") / utils.get_cpu_model_proc() / exp / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)   

    print("\nRunning binary with command: \n", cmd)
    print("\n\n")

    try:
        utils.configure_cpu_for_benchmarks(True)
        t0 = time.perf_counter()
        proc = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False)
        log = proc.stdout

        if a.save_log:
            # ------------------------------------------------------------------
            #  Save the raw log (without ANSI colour codes) next to the CSV file
            # ------------------------------------------------------------------
            ansi_rx = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
            log_plain = ansi_rx.sub("", log)

            # Timestamped file name to avoid accidental overwrites
            ts = time.strftime("%Y%m%d-%H%M%S")
            log_fp = out_dir / f"{a.feature_split_type}-{a.numerical_split_type}-{a.num_threads}t-{ts}.log"

            try:
                log_fp.write_text(log_plain, encoding="utf-8")
                print("Raw log saved to", log_fp)
            except Exception as err:
                print(f"⚠️  Could not write log file: {err}", file=sys.stderr)

        if proc.returncode < 0:
            print(f"binary died with signal {-proc.returncode}")

        dt = time.perf_counter() - t0
        print(f"\n⏱  Binary subprocess ran for {dt:.4f} s\n")
        log_plain = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', log)
        table = parse_parallel_chrono(log_plain)

        # ------------------------------------------------------------------
        #  >>>  NEW: build file name from arguments instead of wall-time
        # ------------------------------------------------------------------
        fname  = f"{a.feature_split_type}-{a.numerical_split_type}-{a.num_threads}Threads.csv"
        out_fp = out_dir / fname

        write_csv(table, vars(a), out_fp)
        print("CSV written to", out_fp)

    except Exception as e:
        print("❌", e, file=sys.stderr)
    finally:
        utils.cleanup_and_exit()
