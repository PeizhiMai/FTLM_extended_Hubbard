#!/usr/bin/env python3
"""
Grand-canonical n(μ) at fixed β: exact diagonalization (NumPy, block sectors).

Optionally overlays the C++ `ftlm_bench_2x2_nmu` TSV when present (default for 2×2 only).
For 4×2 and larger, use ED only (dense Jacobi in C++ would be impractical for the largest blocks).
For FTLM-only plots (no ED), see `plot_n_vs_mu_ftlm_only.py` + `ftlm_bench_ftlm_nmu_rect` (reports wall time and peak RSS by default).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from hubbard_rect_ed import all_eigenlevels_rect, grand_canonical_n


def parse_tsv_header(line: str) -> dict[str, float]:
    """Parse '# ... beta=20  t=1 tp=-0.35 U=5.75 V=0.9' style header."""
    out: dict[str, float] = {}
    for key in ("beta", "t", "tp", "U", "V"):
        m = re.search(rf"\b{key}=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", line)
        if m:
            out[key] = float(m.group(1))
    return out


def load_tsv(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mus: list[float] = []
    ns: list[float] = []
    meta: dict[str, float] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                meta.update(parse_tsv_header(line))
                continue
            parts = line.split()
            mus.append(float(parts[0]))
            ns.append(float(parts[1]))
    return np.array(mus), np.array(ns), meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot n(mu) from grand-canonical ED (+ optional C++ TSV)")
    ap.add_argument("--lx", type=int, default=2)
    ap.add_argument("--ly", type=int, default=2)
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--tp", type=float, default=-0.35)
    ap.add_argument("--U", type=float, default=5.75)
    ap.add_argument("--V", type=float, default=0.9)
    ap.add_argument("--mu-min", type=float, default=-5.0)
    ap.add_argument("--mu-max", type=float, default=20.0)
    ap.add_argument("--n-mu", type=int, default=221)
    ap.add_argument(
        "--cpp-tsv",
        type=Path,
        default=None,
        help="Optional overlay from ftlm_bench_*_nmu (defaults to 2x2 TSV only for lx=ly=2)",
    )
    ap.add_argument("--out", type=Path, default=None, help="PNG path (default under benchmarks/data/)")
    ap.add_argument("--no-show", action="store_true", help="Do not open interactive window")
    args = ap.parse_args()

    n_sites = args.lx * args.ly
    cpp_tsv = args.cpp_tsv
    if cpp_tsv is None and args.lx == 2 and args.ly == 2:
        cpp_tsv = REPO_ROOT / "benchmarks/data/n_vs_mu_beta20.tsv"

    use_cpp = cpp_tsv is not None and cpp_tsv.is_file()
    if cpp_tsv is not None and not cpp_tsv.is_file():
        print(f"Note: missing --cpp-tsv={cpp_tsv}, ED-only plot.", file=sys.stderr)

    if use_cpp:
        mu_grid, n_cpp, meta = load_tsv(cpp_tsv)
        if not meta:
            print("TSV has no parameter header line starting with #", file=sys.stderr)
            return 1
        beta = meta["beta"]
        p = {k: meta[k] for k in ("t", "tp", "U", "V") if k in meta}
    else:
        beta = args.beta
        p = {"t": args.t, "tp": args.tp, "U": args.U, "V": args.V}
        mu_grid = np.linspace(args.mu_min, args.mu_max, max(2, args.n_mu))

    if args.out is None:
        if args.lx == 2 and args.ly == 2 and use_cpp:
            args.out = REPO_ROOT / "benchmarks/data/n_vs_mu_beta20_ed_compare.png"
        else:
            args.out = REPO_ROOT / f"benchmarks/data/n_vs_mu_{args.lx}x{args.ly}_beta{beta:g}_ed.png"

    if n_sites >= 8:
        print(
            f"# ED: all eigenlevels for {args.lx}×{args.ly} ({n_sites} sites) — expect minutes of CPU time...",
            file=sys.stderr,
            flush=True,
        )

    levels = all_eigenlevels_rect(args.lx, args.ly, p, 0.0, 0.0)
    n_ed = np.array([grand_canonical_n(beta, float(m), levels, n_sites) for m in mu_grid])

    if use_cpp:
        diff = np.max(np.abs(n_ed - n_cpp))
        print(f"max|n_python_ED - n_cpp_TSV| = {diff:.3e}  (levels={len(levels)})")
    else:
        print(f"ED levels={len(levels)}  beta={beta}  grid_points={len(mu_grid)}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    if use_cpp:
        ax.plot(mu_grid, n_cpp, "o", ms=3, mfc="none", color="C0", label="C++ benchmark (TSV)")
    ax.plot(
        mu_grid,
        n_ed,
        "-" if use_cpp else "-",
        lw=2.0,
        color="C1" if use_cpp else "C0",
        alpha=0.9,
        label="Exact diagonalization (NumPy)" + (" (overlay)" if use_cpp else ""),
    )
    ax.set_xlabel(r"chemical potential $\mu$")
    ax.set_ylabel(r"$n = \langle \hat N\rangle / N_{\mathrm{sites}}$")
    ax.set_title(
        rf"{args.lx}$\times${args.ly} PBC grand canonical, $\beta={beta:g}$  "
        rf"$t={p['t']}\ t'={p['tp']}\ U={p['U']}\ V={p['V']}$"
    )
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    ax.set_ylim(-0.05, 2.15)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
