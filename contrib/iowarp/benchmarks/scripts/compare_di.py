#!/usr/bin/env python3
"""compare_di.py — diff two bench_ivf_qps --dump-di result dumps.

Usage: compare_di.py A.bin B.bin [--max-report N]

Format (written by dump_di_file in bench_ivf_qps.cpp):
  magic "DIQ1", u64 nq, u64 k, float32 D[nq*k], int64 I[nq*k]

Per query, classifies the two k-length result lists as:
  exact        raw bytes equal (same order, same values)
  order-only   same (D, I) pairs after canonical (D, then I) sort
  boundary-tie same D multiset; I differences confined to entries whose
               distance equals the k-th (worst kept) distance — the k-th
               neighbor is ambiguous under ties, both answers are exact
  MISMATCH     anything else (a real result difference)

Exit code 0 iff no MISMATCH queries.
"""

import struct
import sys


def read_dump(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"DIQ1":
            sys.exit(f"ERROR: {path}: bad magic {magic!r}")
        nq, k = struct.unpack("<QQ", f.read(16))
        n = nq * k
        D = struct.unpack(f"<{n}f", f.read(4 * n))
        I = struct.unpack(f"<{n}q", f.read(8 * n))
    return nq, k, D, I


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    max_report = 5
    for i, a in enumerate(sys.argv[1:]):
        if a == "--max-report":
            max_report = int(sys.argv[1:][i + 1])
    if len(args) != 2:
        sys.exit(__doc__)
    nq_a, k_a, D_a, I_a = read_dump(args[0])
    nq_b, k_b, D_b, I_b = read_dump(args[1])
    if (nq_a, k_a) != (nq_b, k_b):
        sys.exit(f"ERROR: shape mismatch: {nq_a}x{k_a} vs {nq_b}x{k_b}")
    nq, k = nq_a, k_a

    exact = order_only = boundary_tie = 0
    mismatches = []
    for qi in range(nq):
        da = D_a[qi * k:(qi + 1) * k]
        ia = I_a[qi * k:(qi + 1) * k]
        db = D_b[qi * k:(qi + 1) * k]
        ib = I_b[qi * k:(qi + 1) * k]
        if da == db and ia == ib:
            exact += 1
            continue
        pa = sorted(zip(da, ia))
        pb = sorted(zip(db, ib))
        if pa == pb:
            order_only += 1
            continue
        # Same distance multiset, id differences only at the boundary
        # distance? (the k-th neighbor is ambiguous under ties; the worst
        # kept distance is max(D) for L2, min(D) for IP — accept either
        # extreme since the dump does not record the metric)
        if sorted(da) == sorted(db):
            diff_ids = {p for p in pa if p not in pb} | {
                p for p in pb if p not in pa}
            diff_dists = {d for d, _ in diff_ids}
            if len(diff_dists) == 1 and diff_dists <= {max(da), min(da)}:
                boundary_tie += 1
                continue
        mismatches.append((qi, pa, pb))

    print(f"queries={nq} k={k}: exact={exact} order_only={order_only} "
          f"boundary_tie={boundary_tie} MISMATCH={len(mismatches)}")
    for qi, pa, pb in mismatches[:max_report]:
        print(f"  q{qi}:")
        print(f"    A: {pa}")
        print(f"    B: {pb}")
    if mismatches:
        print("RESULT: MISMATCH")
        sys.exit(1)
    print("RESULT: PASS (result sets identical up to k-th-distance ties)")


if __name__ == "__main__":
    main()
