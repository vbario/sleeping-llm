#!/bin/bash
# EGO-SELECT matrix driver (notes/131-ego-selector-experiment §7.2).
#
# 8 conditions x 3 seeds {41, 42, 43} = 24 runs, sequential, INTERLEAVED
# (C0s41, C1s41, ..., C6s41, C0s42, ...) so thermal drift never aligns with
# condition. Failure-tolerant and resumable: cells whose result JSON exists
# and is valid are skipped. __pycache__ cleared between runs. Per-run stdout
# (including full valuation-call prompts/outputs printed by the policies)
# is teed to experiments/results/logs/.
#
# Run as:  ./experiments/run_ego_matrix.sh        (self-wraps in caffeinate)
# or:      caffeinate -is ./experiments/run_ego_matrix.sh
#
# Options: pass --quick to run the pilot variant of every cell.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/experiments/results"
LOG_DIR="$RESULTS_DIR/logs"
STREAM="$REPO_ROOT/experiments/data/ego_corpus_stream.json"

SEEDS=(41 42 43)
CONDITIONS=(random surprise borrowed judge judge_rules ego_static ego_full oracle)

QUICK=""
SUFFIX=""
if [ "${1:-}" = "--quick" ]; then
    QUICK="--quick"
    SUFFIX="_quick"
fi

# Prevent macOS idle sleep for the whole matrix (§11 R26). Self-wrap so a
# bare invocation is still protected; lid open + plugged in still required.
if [ -z "${EGO_MATRIX_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
    export EGO_MATRIX_CAFFEINATED=1
    exec caffeinate -is "$0" "$@"
fi

if [ ! -f "$STREAM" ]; then
    echo "FATAL: frozen stream not found: $STREAM"
    echo "Run the freeze phase first: python experiments/ego_freeze.py"
    exit 1
fi

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

clear_pycache() {
    find "$REPO_ROOT/src" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
}

# A cell is complete when its result JSON parses and contains the final
# battery + all harness assertions passing (the results manifest, §7.2).
cell_done() {
    local f="$1"
    [ -f "$f" ] || return 1
    python3 - "$f" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    ok = ("final_battery" in d and d["final_battery"]
          and d.get("harness_assertions")
          and all(d["harness_assertions"].values()))
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
EOF
}

total=0
skipped=0
failed=0
start_ts=$(date +%s)
echo "=== EGO-SELECT matrix: ${#CONDITIONS[@]} conditions x ${#SEEDS[@]} seeds ==="

for seed in "${SEEDS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        total=$((total + 1))
        result_json="$RESULTS_DIR/ego_${cond}_s${seed}${SUFFIX}.json"
        log_file="$LOG_DIR/ego_${cond}_s${seed}${SUFFIX}.log"

        if cell_done "$result_json"; then
            echo "[skip] $cond seed $seed — valid result exists"
            skipped=$((skipped + 1))
            continue
        fi

        echo ""
        echo "=== [$(date '+%H:%M:%S')] cell: $cond seed $seed ==="
        clear_pycache

        if python3 "$REPO_ROOT/experiments/ego_matrix.py" \
                --condition "$cond" --seed "$seed" $QUICK \
                2>&1 | tee "$log_file"; then
            if cell_done "$result_json"; then
                echo "[done] $cond seed $seed"
            else
                echo "[FAIL] $cond seed $seed — result invalid (see $log_file)"
                failed=$((failed + 1))
            fi
        else
            echo "[FAIL] $cond seed $seed — nonzero exit (see $log_file)"
            failed=$((failed + 1))
        fi
    done
done

clear_pycache
elapsed=$(( $(date +%s) - start_ts ))
echo ""
echo "=== Matrix complete: $total cells, $skipped skipped, $failed failed"
echo "=== Elapsed: $((elapsed / 3600))h $(( (elapsed % 3600) / 60 ))m"
[ "$failed" -eq 0 ] || exit 1
