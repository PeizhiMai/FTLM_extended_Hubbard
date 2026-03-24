#!/usr/bin/env python3
"""
Time / peak-RSS report for block exact diagonalization (NumPy eigvalsh per N_up,N_dn sector)
+ grand-canonical n(μ) loop — same Hamiltonian as ftlm build_hubbard_geometry.
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from hubbard_rect_ed import all_eigenlevels_rect, grand_canonical_n


def max_rss_mib() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(r) / (1024.0 * 1024.0)
    return float(r) / 1024.0


def default_params() -> dict[str, float]:
    return {"t": 1.0, "tp": -0.35, "U": 5.75, "V": 0.9}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lx", type=int, default=4)
    ap.add_argument("--ly", type=int, default=2)
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--mu-min", type=float, default=-5.0)
    ap.add_argument("--mu-max", type=float, default=25.0)
    ap.add_argument("--n-mu", type=int, default=221)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--tp", type=float, default=-0.35)
    ap.add_argument("--U", type=float, default=5.75)
    ap.add_argument("--V", type=float, default=0.9)
    args = ap.parse_args()

    p = {"t": args.t, "tp": args.tp, "U": args.U, "V": args.V}
    n_sites = args.lx * args.ly
    rss0 = max_rss_mib()

    if n_sites >= 8:
        print(
            f"# ED: building all ({n_sites + 1})² particle sectors + eigvalsh (several minutes on 8+ sites)...",
            file=sys.stderr,
            flush=True,
        )

    t0 = time.perf_counter()
    levels = all_eigenlevels_rect(args.lx, args.ly, p, 0.0, 0.0)
    t_ed = time.perf_counter() - t0
    rss_after_ed = max_rss_mib()

    mus = []
    t_mu0 = time.perf_counter()
    for i in range(args.n_mu):
        tmu = float(i) / max(1, args.n_mu - 1)
        mus.append(args.mu_min + tmu * (args.mu_max - args.mu_min))
    for mu in mus:
        _ = grand_canonical_n(args.beta, mu, levels, n_sites)
    t_gc = time.perf_counter() - t_mu0
    rss_final = max_rss_mib()

    # Machine-parseable line for run_4x2_resource_compare.py
    print(
        f"METRIC kind=ED_numpy_block lx={args.lx} ly={args.ly} "
        f"wall_ed_s={t_ed:.6g} wall_gc_mu_loop_s={t_gc:.6g} "
        f"wall_total_s={t_ed + t_gc:.6g} levels={len(levels)} n_mu={args.n_mu} beta={args.beta} "
        f"max_rss_mib={rss_final:.3f} rss_after_ed_mib={rss_after_ed:.3f} rss_baseline_mib={rss0:.3f}"
    )
    print(
        f"# ED: block-diagonal eigvalsh per (N_up,N_down); grand-canonical loop over {args.n_mu} μ values."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
