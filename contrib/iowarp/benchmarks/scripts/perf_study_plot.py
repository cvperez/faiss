#!/usr/bin/env python3
"""
FAISS x IOWarp performance study — plots (CTE ChiMod backend).

Same figures, metrics and JSON schema as the mmap study's
ondisk_step4_plot.py, parameterized by --prefix (default perf_study)
so it reads perf_study_exp{1,2}_nb{N}M_n{K}.json records.

Produces six figures from the exp1 and exp2 JSON results:

  Exp1 Panel A          — read-only QPS vs total DB size, 1-node only
  Exp1 Panel B          — read-only QPS vs total DB size, 2/4/8 nodes (one line each)
  Exp2 Panel A          — mixed 80/20 QPS vs total DB size, 1-node only
  Exp2 Panel B          — mixed 80/20 QPS vs total DB size, 2/4/8 nodes
  active_time_exp1/2    — per-node SSD active time (%util) vs DB size
  disk_throughput_exp1/2— aggregate SSD transfer rate (MB/s) vs DB size

Metric definitions:
  exp1 QPS — mean over warm passes ("passes" with label warm*).
  exp2 QPS — sustained mixed QPS: mean of qps_series windows with in_write=False.

Records whose relevant metric is missing (crashed/timed-out runs that saved
metadata but no passes) are dropped, not plotted as gaps.

Usage:
    python3 perf_study_plot.py [--results-dir <dir>] [--out-dir <dir>]
                               [--prefix perf_study]
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DEFAULT = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "results", "performance_study"))
PREFIX = "perf_study"  # overridden by --prefix

# Node counts and display colours
NODE_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c", 8: "#d62728"}
NODE_LABELS = {1: "1 node", 2: "2 nodes", 4: "4 nodes", 8: "8 nodes"}
# Per-node RAM threshold for shading
RAM_THRESHOLD_M = 92.3   # million vectors per node
SIZE_TICKS = [44, 64, 92, 130, 185, 262, 370, 430, 500]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_exp(results_dir, exp_num, n_nodes=None):
    """Load all exp{exp_num} JSONs, optionally filtered by n_nodes.

    Returns list of dicts (raw JSON), sorted by (n_nodes, nb_M) ascending.
    """
    pattern = os.path.join(results_dir, f"{PREFIX}_exp{exp_num}_nb*M_n*.json")
    files   = glob.glob(pattern)
    records = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            warnings.warn(f"Skipping {f}: {e}")
            continue
        if n_nodes is not None and d.get("n_nodes") != n_nodes:
            continue
        records.append(d)
    records.sort(key=lambda d: (d.get("n_nodes", 0), d.get("nb_M", 0)))
    return records


def get_warm_qps(record):
    """Mean QPS of read-only warm passes ('passes' for exp1, baseline for exp2)."""
    key = "passes" if "passes" in record else "baseline_passes"
    warm = [p["qps"] for p in record.get(key) or [] if p["label"].startswith("warm")]
    return float(np.mean(warm)) if warm else None


def get_mixed_qps(record):
    """Sustained mixed-workload QPS: mean over qps_series windows outside writes.

    This is the exp2 headline metric.  Falls back to None (record dropped) when
    the run died before the mixed stage produced any windows.
    """
    series = record.get("qps_series") or []
    # Guard against a degenerate mixed stage that produced ~no windows (e.g. the
    # writer terminated Stage 3 before the readers ramped up): a lone window is
    # not a throughput measurement, so drop the record rather than plot a false 0.
    if len(series) < 2:
        return None
    vals = [w["qps"] for w in series if not w.get("in_write")]
    return float(np.mean(vals)) if vals else None


def xy_points(records, metric):
    """(nb_M, value) pairs for records where metric is available, size-sorted."""
    pts = [(d["nb_M"], metric(d)) for d in records]
    pts = [(x, y) for x, y in pts if y is not None]
    pts.sort(key=lambda p: p[0])
    return pts


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def style_size_axis(ax):
    """Log size axis with the actual sweep sizes as ticks."""
    ax.set_xscale("log")
    ax.set_xticks(SIZE_TICKS)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    # 9 ticks crowd the log axis — rotate so labels stay legible
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlabel("Total database size (M vectors)")


def style_qps_axis(ax):
    """Log QPS axis — warm in-RAM (~100 QPS) and disk-bound (~0.1 QPS) span
    three decades; a linear axis flattens the disk-bound points onto zero."""
    ax.set_yscale("log")


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def fig_panel_a(records_1node, exp_num, out_dir, metric, metric_note):
    """Single-node QPS vs DB size (Panel A for exp1 or exp2)."""
    pts = xy_points(records_1node, metric)
    if not pts:
        print(f"[plot] no usable 1-node exp{exp_num} data, skipping Panel A")
        return

    x, y = zip(*pts)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, "o-", color=NODE_COLORS[1], lw=2, ms=6)

    ax.axvline(RAM_THRESHOLD_M, color="gray", linestyle="--", alpha=0.6, lw=1.0)
    style_size_axis(ax)
    style_qps_axis(ax)
    ax.text(RAM_THRESHOLD_M * 1.03, ax.get_ylim()[1] * 0.7,
            "per-node RAM", fontsize=7, color="gray", va="top")

    ax.set_ylabel(f"Aggregate QPS ({metric_note})")
    if exp_num == 1:
        ax.set_title("Read-only search throughput vs. database size\n1 node",
                     fontsize=10)
    else:
        ax.set_title("Search throughput under a mixed read/write workload\n1 node",
                     fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{PREFIX}_exp{exp_num}_panel_a.{ext}")
        fig.savefig(p, dpi=150)
        print(f"[plot] saved {p}")
    plt.close(fig)


def fig_panel_b(records_multi, exp_num, out_dir, metric, metric_note):
    """Multi-node QPS vs DB size (Panel B, 2/4/8 nodes)."""
    by_n = {}
    for d in records_multi:
        by_n.setdefault(d["n_nodes"], []).append(d)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False

    for n in sorted(by_n.keys()):
        pts = xy_points(by_n[n], metric)
        if not pts:
            continue
        plotted = True
        x, y = zip(*pts)
        ax.plot(x, y, "o-",
                color=NODE_COLORS.get(n, "grey"),
                label=NODE_LABELS.get(n, f"{n} nodes"),
                lw=2, ms=6)
        boundary = RAM_THRESHOLD_M * n
        if min(x) <= boundary <= max(x) * 1.2:
            ax.axvline(boundary, color=NODE_COLORS.get(n, "grey"),
                       linestyle=":", alpha=0.4, lw=1.0)

    if not plotted:
        print(f"[plot] no usable multi-node exp{exp_num} data, skipping Panel B")
        plt.close(fig)
        return

    style_size_axis(ax)
    style_qps_axis(ax)
    ax.set_ylabel(f"Aggregate QPS ({metric_note})")
    workload = ("Read-only search throughput" if exp_num == 1
                else "Search throughput under a mixed read/write workload")
    ax.set_title(f"{workload}\n"
                 f"2/4/8 nodes (dotted lines: per-node RAM limit × node count)",
                 fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{PREFIX}_exp{exp_num}_panel_b.{ext}")
        fig.savefig(p, dpi=150)
        print(f"[plot] saved {p}")
    plt.close(fig)


def _disk_active_row(record, exp_num):
    """(active_pct, read_mbps, write_mbps) for the measured workload window of
    one record.  exp1 → mean over read-only warm passes; exp2 → the whole
    mixed-stage snapshot (mixed_disk_active).  None entries where unavailable.

    All values come from node-local /proc/diskstats deltas (io_ticks → %util,
    sectors → MB/s), averaged across the shards' disks.
    """
    if exp_num == 2:
        m = record.get("mixed_disk_active") or {}
        return m.get("active_pct"), m.get("read_mbps"), m.get("write_mbps")
    warm = [p for p in record.get("passes") or [] if p["label"].startswith("warm")]
    def _mean(key):
        vals = [p.get(key) for p in warm if p.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None
    return _mean("disk_active_pct"), _mean("disk_read_mbps"), _mean("disk_write_mbps")


def _active_disk_by_n(records, exp_num):
    """{n_nodes: [(nb_M, active_pct, read_mbps, write_mbps), ...]} for records
    that carry a local-disk active measurement."""
    by_n = {}
    for d in records:
        a, r, w = _disk_active_row(d, exp_num)
        if a is None:
            continue
        by_n.setdefault(d.get("n_nodes", 1), []).append((d["nb_M"], a, r, w))
    return by_n


def _save(fig, out_dir, stem):
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=150)
        print(f"[plot] saved {p}")
    plt.close(fig)


def fig_active_time(records, exp_num, out_dir, nodes, tag, tag_title):
    """Per-node disk active time vs DB size, one line per topology, for the
    topologies in `nodes` (1-node and multi-node go to separate figures).

    active_pct is the mean fraction of wall-clock time a *single* node's disk had
    ≥1 I/O in flight (Σ Δio_ticks over the N shard disks / (N × elapsed)), so it
    is a per-node 0-100 % figure regardless of node count — each node has its own
    independent SSD.  Measured over the warm passes (exp1) / mixed stage (exp2).
    """
    by_n = {n: v for n, v in _active_disk_by_n(records, exp_num).items() if n in nodes}
    if not by_n:
        print(f"[plot] no local-disk active data in exp{exp_num} {tag} — skipping")
        return
    fig, ax = plt.subplots(figsize=(6, 4) if tag == "1node" else (7, 4.5))
    for n in sorted(by_n.keys()):
        rows = sorted(by_n[n])
        ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-",
                color=NODE_COLORS.get(n, "grey"), lw=2, ms=6,
                label=NODE_LABELS.get(n, f"{n} nodes"))
    ax.axvline(RAM_THRESHOLD_M, color="gray", linestyle="--", alpha=0.5, lw=1.0)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Per-node disk active time (% of wall time,\nmean over the N shard disks)")
    workload = "read-only workload" if exp_num == 1 else "mixed read/write workload"
    ax.set_title(f"Per-node SSD active time vs. database size\n{workload}, {tag_title}",
                 fontsize=10)
    style_size_axis(ax)
    if len(by_n) > 1:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    _save(fig, out_dir, f"{PREFIX}_active_time_exp{exp_num}_{tag}")


def fig_disk_throughput(records, exp_num, out_dir, nodes, tag, tag_title):
    """SSD transfer rate vs DB size, one line per topology (read; + write for
    exp2), for the topologies in `nodes` (1-node and multi-node go to separate
    figures).  Aggregate across the N shard disks (they are read in parallel)."""
    by_n = {n: v for n, v in _active_disk_by_n(records, exp_num).items() if n in nodes}
    if not by_n:
        print(f"[plot] no local-disk transfer data in exp{exp_num} {tag} — skipping")
        return
    fig, ax = plt.subplots(figsize=(6, 4) if tag == "1node" else (7, 4.5))
    for n in sorted(by_n.keys()):
        rows = sorted(by_n[n])
        x  = [r[0] for r in rows]
        rd = [r[2] for r in rows]
        wr = [r[3] for r in rows]
        color = NODE_COLORS.get(n, "grey")
        lbl   = NODE_LABELS.get(n, f"{n} nodes")
        if any(v is not None for v in rd):
            ax.plot(x, [v if v is not None else np.nan for v in rd], "o-",
                    color=color, lw=2, ms=6, label=f"{lbl} read")
        if exp_num == 2 and any(v for v in wr if v is not None):
            ax.plot(x, [v if v is not None else np.nan for v in wr], "s--",
                    color=color, lw=1.2, ms=4, alpha=0.75, label=f"{lbl} write")
    ax.axvline(RAM_THRESHOLD_M, color="gray", linestyle="--", alpha=0.5, lw=1.0)
    workload = "read-only workload" if exp_num == 1 else "mixed read/write workload"
    ax.set_title(f"Aggregate SSD transfer rate vs. database size\n{workload}, {tag_title}",
                 fontsize=10)
    ax.set_ylabel("SSD transfer rate (MB/s, summed over N disks)")
    style_size_axis(ax)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    _save(fig, out_dir, f"{PREFIX}_disk_throughput_exp{exp_num}_{tag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DEFAULT)
    parser.add_argument("--out-dir",     default=RESULTS_DEFAULT)
    parser.add_argument("--prefix",      default=PREFIX)
    args = parser.parse_args()

    PREFIX = args.prefix

    os.makedirs(args.out_dir, exist_ok=True)

    exp1_1node = load_exp(args.results_dir, 1, n_nodes=1)
    exp1_multi = [d for d in load_exp(args.results_dir, 1) if d["n_nodes"] in (2, 4, 8)]
    exp2_1node = load_exp(args.results_dir, 2, n_nodes=1)
    exp2_multi = [d for d in load_exp(args.results_dir, 2) if d["n_nodes"] in (2, 4, 8)]

    print(f"[plot] exp1 1-node: {len(exp1_1node)} records")
    print(f"[plot] exp1 multi:  {len(exp1_multi)} records")
    print(f"[plot] exp2 1-node: {len(exp2_1node)} records")
    print(f"[plot] exp2 multi:  {len(exp2_multi)} records")

    fig_panel_a(exp1_1node, 1, args.out_dir, get_warm_qps, "warm passes")
    fig_panel_b(exp1_multi, 1, args.out_dir, get_warm_qps, "warm passes")
    fig_panel_a(exp2_1node, 2, args.out_dir, get_mixed_qps,
                "sustained, outside write windows")
    fig_panel_b(exp2_multi, 2, args.out_dir, get_mixed_qps,
                "sustained, outside write windows")

    # Disk figures — per-node active time + aggregate throughput, each split into
    # a 1-node figure and a multi-node (2/4/8) figure (same as the QPS panels).
    for e in (1, 2):
        recs = load_exp(args.results_dir, e)
        for nodes, tag, tag_title in (((1,), "1node", "1 node"),
                                      ((2, 4, 8), "multinode", "2/4/8 nodes")):
            fig_active_time(recs, e, args.out_dir, nodes, tag, tag_title)
            fig_disk_throughput(recs, e, args.out_dir, nodes, tag, tag_title)

    print("\n[plot] All figures complete.")


if __name__ == "__main__":
    main()
