#!/usr/bin/env python3
"""40_plot_qps.py — grouped-bar QPS comparison: mmap baseline vs IOWarp levels.

Purpose:
    Read the benchmark CSVs produced by `bench_ivf_qps --protocol step3`
    (via scripts/30_run_bench.sh) plus the user's pre-existing mmap baseline
    CSV, and render warm-pass QPS as grouped bars:
    x = index volume, series = system. Baseline numbers are an INPUT file,
    never hardcoded, so they stay versioned next to the new measurements.

CSV schemas:
    Baseline (--baseline, required):
        volume,system,pass,qps
        e.g.  ondisk_nb50M,mmap-baseline,warm1,93.1
        `pass` is one of: cold, warm1, warm2 (any pass starting with "warm"
        counts toward the warm mean; cold rows are ignored by this plot).

    Our runs (results/qps_<volume>.csv, written by bench_ivf_qps):
        one row per (backend, pass). Required columns: backend, pass, qps.
        Optional columns (used if present, ignored otherwise): volume,
        nlist, nprobe, k, threads, nq, elapsed_s, majflt, read_bytes,
        cte_bytes_fetched, notes. If `volume` is absent it is derived from
        the filename (qps_<volume>.csv). Backends map to series labels:
        iowarp -> "iowarp-L0", chimod -> "chimod-L1",
        mmap -> "mmap-harness" (the §7 cross-check run, plotted only if present).

Inputs:
    --baseline PATH     baseline CSV (schema above) — required
    --results-dir DIR   directory with qps_*.csv (default: ../results)
    --linear            force a linear y axis (default: log if max/min > 50,
                        which the 0.2-vs-93 QPS regime always triggers)

Outputs:
    <results-dir>/qps_comparison.png
    <results-dir>/qps_comparison.pdf

Example:
    python3 scripts/40_plot_qps.py --baseline results/baseline_panelA.csv
"""

import argparse
import csv
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- palette (validated categorical slots, fixed order; ink/chrome tokens) ---
SERIES_ORDER = ["mmap-baseline", "iowarp-L0", "chimod-L1", "mmap-harness"]
SERIES_COLOR = {
    "mmap-baseline": "#2a78d6",  # slot 1 blue
    "iowarp-L0": "#1baf7a",      # slot 2 aqua
    "chimod-L1": "#eda100",      # slot 3 yellow
    "mmap-harness": "#4a3aa7",   # slot 5 violet (cross-check only)
}
BACKEND_TO_SERIES = {
    "iowarp": "iowarp-L0",
    "chimod": "chimod-L1",
    "mmap": "mmap-harness",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"


def is_warm(pass_name):
    return pass_name.strip().lower().startswith("warm")


def read_baseline(path):
    """-> {(volume, system): [warm qps...]}"""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not is_warm(row["pass"]):
                continue
            key = (row["volume"].strip(), row["system"].strip())
            out.setdefault(key, []).append(float(row["qps"]))
    return out


def read_run_csv(path):
    """-> {(volume, series): [warm qps...]} from one qps_<volume>.csv."""
    m = re.match(r"qps_(.+)\.csv$", os.path.basename(path))
    vol_from_name = m.group(1) if m else os.path.basename(path)
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not is_warm(row.get("pass", "")):
                continue
            backend = row.get("backend", row.get("system", "")).strip()
            series = BACKEND_TO_SERIES.get(backend, backend)
            volume = row.get("volume", "").strip() or vol_from_name
            out.setdefault((volume, series), []).append(float(row["qps"]))
    return out


def volume_sort_key(volume):
    """Sort volumes by dataset size (nb millions) when parseable."""
    m = re.search(r"nb(\d+)M", volume)
    return (0, int(m.group(1)), volume) if m else (1, 0, volume)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", required=True,
                    help="baseline CSV (volume,system,pass,qps)")
    ap.add_argument("--results-dir",
                    default=os.path.join(here, os.pardir, "results"))
    ap.add_argument("--linear", action="store_true",
                    help="force linear y axis (default: log when range is wide)")
    args = ap.parse_args()

    data = read_baseline(args.baseline)
    run_csvs = sorted(glob.glob(os.path.join(args.results_dir, "qps_*.csv")))
    if not run_csvs:
        print(f"warning: no qps_*.csv under {args.results_dir} — "
              "plotting baseline only", file=sys.stderr)
    for path in run_csvs:
        for key, vals in read_run_csv(path).items():
            data.setdefault(key, []).extend(vals)
    if not data:
        sys.exit("no warm-pass rows found in any input")

    # mean warm QPS per (volume, series)
    warm = {k: sum(v) / len(v) for k, v in data.items()}
    volumes = sorted({v for v, _ in warm}, key=volume_sort_key)
    present = {s for _, s in warm}
    series = [s for s in SERIES_ORDER if s in present]
    series += sorted(present - set(series))  # unknown labels, if any

    # --- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6.4, 1.9 * len(volumes)), 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    group_w = 0.78
    bar_w = group_w / max(len(series), 1)
    qps_vals = list(warm.values())
    use_log = (not args.linear) and max(qps_vals) / max(min(qps_vals), 1e-9) > 50

    for j, s in enumerate(series):
        xs, ys = [], []
        for i, vol in enumerate(volumes):
            if (vol, s) in warm:
                xs.append(i - group_w / 2 + (j + 0.5) * bar_w)
                ys.append(warm[(vol, s)])
        color = SERIES_COLOR.get(s, MUTED)
        ax.bar(xs, ys, width=bar_w * 0.9,  # ~2px surface gap between bars
               color=color, label=s, zorder=3)
        # direct value labels (log axis makes magnitudes hard to read off)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}" if y >= 1 else f"{y:.2f}",
                        (x, y), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK_2)

    if use_log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(volumes)))
    ax.set_xticklabels(volumes, fontsize=9, color=INK)
    ax.set_ylabel("warm-pass QPS (500 queries / elapsed)"
                  + ("  [log]" if use_log else ""), fontsize=10, color=INK)
    ax.set_title("IVF on-disk search: OS page cache (mmap) vs IOWarp/CTE",
                 fontsize=11, color=INK, pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK,
              loc="upper right", ncol=min(len(series), 2))
    fig.text(0.01, 0.01,
             "k=10, nprobe=nlist/64, 8 threads, 1 batched call/pass; "
             "warm = mean of 2 post-cold passes. Budgets: page cache ~44 GiB "
             "(mmap) vs CTE 30 GiB RAM tier + NVMe (iowarp/chimod).",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    for ext in ("png", "pdf"):
        out = os.path.join(args.results_dir, f"qps_comparison.{ext}")
        fig.savefig(out, dpi=200, facecolor=SURFACE)
        print("wrote", out)


if __name__ == "__main__":
    main()
