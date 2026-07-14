#!/usr/bin/env python3
"""telemetry_sampler.py — system/tier telemetry alongside ingest + bench.

Purpose: the IOWarp analogue of the baseline's Figure-3a instrumentation
(page-cache residency + disk I/O over time). Samples every INTERVAL s:
  * NVMe device throughput and ACTIVE TIME (/proc/diskstats io_ticks)
  * CTE NVMe file-tier occupancy (size of the tier directory)
  * RAM-tier proxy: /dev/shm used bytes + clio_run RSS
  * NFS read rate on the volume source (nfs mount counters)
  * MemAvailable
  * current phase (read from a phase file the run scripts update:
    setup / ingest / bench_<backend> / done)

Inputs (flags): --out CSV --phase-file PATH [--interval 5]
                [--nvme-dev nvme0n1] [--tier-dir DIR] [--nfs-mount /mnt/common]
Output: CSV with one row per tick:
  ts,phase,nvme_rd_mb_s,nvme_wr_mb_s,nvme_active_pct,cte_file_tier_gb,
  devshm_used_gb,clio_rss_gb,mem_avail_gb,nfs_rd_mb_s
Runs until killed (the sbatch wrapper traps and kills it at job end).

Example:
  python3 telemetry_sampler.py --out results/telemetry_x.csv \
      --phase-file results/.phase_123 --tier-dir /mnt/nvme/$USER/cte_tier &
"""
import argparse
import os
import subprocess
import sys
import time


def diskstats(dev):
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                if len(p) > 13 and p[2] == dev:
                    # sectors read, sectors written, io_ticks(ms)
                    return int(p[5]), int(p[9]), int(p[12])
    except OSError:
        pass
    return 0, 0, 0


def dir_bytes(path):
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for fn in files:
                try:
                    total += os.stat(os.path.join(root, fn)).st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def devshm_used():
    try:
        st = os.statvfs("/dev/shm")
        return (st.f_blocks - st.f_bfree) * st.f_frsize
    except OSError:
        return 0


def clio_rss():
    try:
        pids = subprocess.run(
            ["pgrep", "-f", "iowarp_core/bin/clio_run"],
            capture_output=True, text=True).stdout.split()
        rss = 0
        for pid in pids:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss += int(line.split()[1]) * 1024
                        break
        return rss
    except (OSError, ValueError):
        return 0


def nfs_read_bytes(mount):
    try:
        with open("/proc/self/mountstats") as f:
            in_section = False
            for line in f:
                if line.startswith("device ") and f" {mount} " in line:
                    in_section = True
                elif line.startswith("device "):
                    in_section = False
                elif in_section and line.strip().startswith("bytes:"):
                    return int(line.split()[1])  # normalreadbytes
    except (OSError, ValueError, IndexError):
        pass
    return 0


def mem_avail():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def read_phase(path):
    try:
        with open(path) as f:
            return f.read().strip() or "unknown"
    except OSError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase-file", required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--nvme-dev", default="nvme0n1")
    ap.add_argument("--tier-dir", default="")
    ap.add_argument("--nfs-mount", default="/mnt/common")
    a = ap.parse_args()

    fresh = not os.path.exists(a.out)
    out = open(a.out, "a", buffering=1)
    if fresh:
        out.write("ts,phase,nvme_rd_mb_s,nvme_wr_mb_s,nvme_active_pct,"
                  "cte_file_tier_gb,devshm_used_gb,clio_rss_gb,"
                  "mem_avail_gb,nfs_rd_mb_s\n")

    rd0, wr0, ticks0 = diskstats(a.nvme_dev)
    nfs0 = nfs_read_bytes(a.nfs_mount)
    t0 = time.time()
    while True:
        time.sleep(a.interval)
        t1 = time.time()
        dt = t1 - t0
        rd1, wr1, ticks1 = diskstats(a.nvme_dev)
        nfs1 = nfs_read_bytes(a.nfs_mount)
        row = (
            f"{int(t1)},{read_phase(a.phase_file)},"
            f"{(rd1 - rd0) * 512 / dt / 2**20:.2f},"
            f"{(wr1 - wr0) * 512 / dt / 2**20:.2f},"
            f"{min(100.0, (ticks1 - ticks0) / (dt * 1000) * 100):.1f},"
            f"{dir_bytes(a.tier_dir) / 2**30 if a.tier_dir else 0:.3f},"
            f"{devshm_used() / 2**30:.3f},"
            f"{clio_rss() / 2**30:.3f},"
            f"{mem_avail() / 2**30:.3f},"
            f"{max(0, nfs1 - nfs0) / dt / 2**20:.2f}\n")
        out.write(row)
        rd0, wr0, ticks0, nfs0, t0 = rd1, wr1, ticks1, nfs1, t1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
