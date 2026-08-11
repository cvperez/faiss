#!/usr/bin/env bash
# submit_perf_study.sh — submit the whole CTE performance-study campaign
# (validation gates, then the size x topology grid, then the plot job) as
# ONE strictly serial sbatch chain: every job depends afterany on the
# previous one, so at no point do two jobs overlap on Ares.
#
# Grid (same cells as the mmap ondisk_step4 study):
#   n1: 44 64 92 130 185
#   n2: 44 64 92 130 185 262 370
#   n4: 44 64 92 130 185 262 370 430 500
#   n8: 44 64 92 130 185 262 370 430 500
#
# Usage (login node):
#   bash submit_perf_study.sh              # gates + grid + plot
#   DRY_RUN=1 bash submit_perf_study.sh    # print the plan only
#   SKIP_GATES=1 bash submit_perf_study.sh # grid only (gates already green)
#
# Cells whose exp1 AND exp2 JSONs are already complete are skipped at
# submit time; the cell script re-checks at run time (idempotent), so
# resubmitting after a partial campaign is safe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CELL="$SCRIPT_DIR/run_perf_study_cell.sh"
OUT_DIR="$ROOT/benchmarks/results/performance_study"
DRY_RUN="${DRY_RUN:-0}"
DEP=""
# Chain link type: gates use afterok (a failed gate halts the campaign,
# --kill-on-invalid-dep cancels the queued rest); the grid uses afterany
# (one flaky cell must not kill the remaining cells — its JSON is simply
# missing and a resubmit retries it). Still strictly serial either way.
DEP_TYPE="afterok"

# After the gates, GATES_OK holds the gatecheck2 job id: every grid cell
# ANDs afterok:$GATES_OK into its dependency, so red gates invalidate the
# whole grid (kill-on-invalid-dep cancels it) while a single failed grid
# cell still lets its afterany successors run.
GATES_OK=""

submit() {  # submit <walltime> <nodes> <export-string> [extra sbatch args...]
    local wall="$1" nodes="$2" exports="$3"
    shift 3
    local depflag=()
    if [ -n "$DEP" ]; then
        local dep="$DEP_TYPE:$DEP"
        [ -n "$GATES_OK" ] && [ "$DEP_TYPE" = afterany ] && dep="$dep,afterok:$GATES_OK"
        depflag=(--dependency="$dep" --kill-on-invalid-dep=yes)
    fi
    if [ "$DRY_RUN" = 1 ]; then
        echo "[dry] sbatch -N$nodes --time=$wall ${depflag[*]:-} $* --export=ALL,$exports run_perf_study_cell.sh"
        DEP="dry"
        return
    fi
    local out
    out="$(sbatch -N "$nodes" --time="$wall" "${depflag[@]}" "$@" \
           --export=ALL,"$exports" "$CELL")"
    echo "$out  (nodes=$nodes, $exports)"
    DEP="$(awk '{print $4}' <<<"$out")"
}

submit_wrap() {  # submit_wrap <name> <command...>
    local name="$1"
    shift
    local depflag=()
    [ -n "$DEP" ] && depflag=(--dependency="$DEP_TYPE:$DEP" --kill-on-invalid-dep=yes)
    if [ "$DRY_RUN" = 1 ]; then
        echo "[dry] sbatch -N1 ${depflag[*]:-} --wrap '$*'  ($name)"
        DEP="dry"
        return
    fi
    local out
    out="$(sbatch -N1 --time=00:20:00 "${depflag[@]}" --job-name="$name" \
           --partition=compute \
           --output="$ROOT/benchmarks/results/logs/%j_${name}.out" \
           --wrap "$*")"
    echo "$out  ($name)"
    DEP="$(awk '{print $4}' <<<"$out")"
}

exp_done() {  # exp_done <1|2> <json>
    python3 - "$1" "$2" <<'EOF'
import json, sys
exp, path = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception:
    sys.exit(1)
if exp == "1":
    warm = [p for p in d.get("passes") or []
            if p.get("label", "").startswith("warm")
            and p.get("qps") is not None
            and p.get("disk_active_pct") is not None]
    sys.exit(0 if len(warm) >= 2 else 1)
series = d.get("qps_series") or []
sys.exit(0 if len(series) >= 2 and d.get("mixed_disk_active") else 1)
EOF
}

# --- validation gates (Phase 8 of the plan; each blocks the chain) -----------
if [ "${SKIP_GATES:-0}" != 1 ]; then
    VAL="$OUT_DIR/validation"
    # Gate 1: selftest (incl. the Add leg) + single-shard exp1 reference
    # on nb44M/n1 with a FULL ingest verify. NOTE: build the contrib on
    # the HEAD node before submitting (cmake reconfigure fails on compute
    # nodes — no boost headers there); BUILD_FIRST is not used here.
    submit 04:00:00 1 "NB_M=44,RUN_EXP=1,RUN_SELFTEST=1,VERIFY_N=all,OUT_DIR=$VAL,JSON_SUFFIX=_singleshard,SHARD_SET=1"
    # Gate 2: multi-shard ingest equivalence — same cell fed by the n2
    # shard pair. Rank slices are contiguous ascending id ranges, so the
    # concatenated per-list order equals the merged order: di_hash must be
    # BITWISE identical to gate 1's (checked by the compare job below).
    submit 04:00:00 1 "NB_M=44,RUN_EXP=1,VERIFY_N=all,OUT_DIR=$VAL,JSON_SUFFIX=_twoshard,SHARD_SET=2"
    # Gates 3+4: first-ever 4- and 8-node clusters, cheap volume, full
    # verify (verify reads are mostly cross-node — the cluster gate).
    # Owner-merge may reorder equal-distance ties, so these must match
    # gate 1 on canon_hash (order-independent), not di_hash.
    submit 04:00:00 4 "NB_M=44,RUN_EXP=1,RUN_SELFTEST=1,VERIFY_N=all,OUT_DIR=$VAL"
    submit 04:00:00 8 "NB_M=44,RUN_EXP=1,RUN_SELFTEST=1,VERIFY_N=all,OUT_DIR=$VAL"
    # Automated equivalence checks over the gate JSONs.
    submit_wrap perf_gatechk "python3 '$SCRIPT_DIR/perf_study_gatecheck.py' '$VAL'"
    # Gate 5: short mixed runs — write path end-to-end on 1 and 4 nodes.
    # Check: achieved_write_fraction ~0.2, windows tagged, no rc=8/9.
    submit 04:00:00 1 "NB_M=44,RUN_EXP=2,MIXED_DURATION_S=600,OUT_DIR=$VAL"
    submit 04:00:00 4 "NB_M=44,RUN_EXP=2,MIXED_DURATION_S=600,OUT_DIR=$VAL"
    # Mixed-run sanity over the two exp2 gate JSONs.
    submit_wrap perf_gatechk2 "python3 '$SCRIPT_DIR/perf_study_gatecheck.py' '$VAL' --exp2"
    GATES_OK="$DEP"
fi

# --- the grid ----------------------------------------------------------------
# The first grid cell hangs off the final gate with afterok (a red gate
# stops the campaign); grid cells then chain with afterany between
# themselves (see DEP_TYPE comment above).
grid_row() {  # grid_row <nodes> <sizes...>
    local nodes="$1"
    shift
    for nb in "$@"; do
        local e1="$OUT_DIR/perf_study_exp1_nb${nb}M_n${nodes}.json"
        local e2="$OUT_DIR/perf_study_exp2_nb${nb}M_n${nodes}.json"
        local run=""
        exp_done 1 "$e1" || run="1"
        exp_done 2 "$e2" || run="${run}2"
        if [ -z "$run" ]; then
            echo "[skip] nb${nb}M n${nodes} — both JSONs complete"
            continue
        fi
        # Walltime: ingest (~NB*520B over NFS) + passes + 1 h mixed stage.
        local wall="06:00:00"
        [ "$nb" -ge 262 ] && wall="10:00:00"
        # Disk-bound cells (per-node ratio >= 1): long, slow passes.
        if awk -v nb="$nb" -v n="$nodes" 'BEGIN{exit !(nb/n/92.3 >= 1.0)}'; then
            wall="12:00:00"
        fi
        # Mixed stage: 1 h in-RAM, 2 h disk-bound (windows are long there).
        local mixed=3600
        if awk -v nb="$nb" -v n="$nodes" 'BEGIN{exit !(nb/n/92.3 >= 1.0)}'; then
            mixed=7200
        fi
        # Heavy-spill cells (> 30 GB NVMe per node) must avoid nodes whose
        # local NVMe is crowded by other users (comp-10 proved short of
        # nb185M/n1's 66 GB — 0-byte tier writes mid-ingest). The partition
        # has 20+ nodes, so excluding a couple never starves an allocation.
        local excl=()
        if awk -v nb="$nb" -v n="$nodes" 'BEGIN{exit !((nb*520/n/1000-30) > 30)}'; then
            excl=(--exclude="${LOWSPACE_NODES:-ares-comp-10}")
        fi
        submit "$wall" "$nodes" "NB_M=$nb,RUN_EXP=$run,MIXED_DURATION_S=$mixed" "${excl[@]}"
        DEP_TYPE="afterany"
    done
}

# ROWS: which topology rows to submit (default all) — lets independent
# row-lanes run concurrently when extra nodes are authorized. DEP_START:
# colon-separated job ids the first cell of this invocation must wait for
# (afterany). NO_PLOT=1: skip the closing plot job (only the final lane
# plots).
if [ -n "${DEP_START:-}" ]; then
    DEP="$DEP_START"
    DEP_TYPE="afterany"
fi
in_rows() { case " ${ROWS:-1 2 4 8} " in *" $1 "*) return 0;; *) return 1;; esac; }

in_rows 1 && grid_row 1 44 64 92 130 185
in_rows 2 && grid_row 2 44 64 92 130 185 262 370
in_rows 4 && grid_row 4 44 64 92 130 185 262 370 430 500
in_rows 8 && grid_row 8 44 64 92 130 185 262 370 430 500

# --- plot job ----------------------------------------------------------------
if [ "${NO_PLOT:-0}" != 1 ]; then
    DEP_TYPE="afterany"
    submit_wrap perf_plot "python3 '$SCRIPT_DIR/perf_study_plot.py' --results-dir '$OUT_DIR' --out-dir '$OUT_DIR'"
fi
echo "chain submitted (last job: $DEP)"
