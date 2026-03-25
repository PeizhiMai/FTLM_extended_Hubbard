#!/usr/bin/env python3
"""Plot 3x2 n(mu): FTLM non-momentum + momentum k-blocks + ED (same grid as TSVs)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from hubbard_rect_ed import all_eigenlevels_rect, grand_canonical_n
from plot_n_vs_mu import load_tsv


def main() -> int:
    ap = argparse.ArgumentParser(description="3x2 three-way n(mu) plot (nomom + k + ED)")
    ap.add_argument(
        "--nomom-tsv",
        type=Path,
        default=REPO_ROOT / "benchmarks/data/n_vs_mu_3x2_beta20_ftlm_nomom.tsv",
    )
    ap.add_argument(
        "--k-tsv",
        type=Path,
        default=REPO_ROOT / "benchmarks/data/n_vs_mu_3x2_beta20_ftlm_k.tsv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks/data/n_vs_mu_3x2_beta20_ftlm_nomom_vs_k_vs_ed.png",
    )
    args = ap.parse_args()

    lx, ly = 3, 2
    mu_n, n_n, meta = load_tsv(args.nomom_tsv)
    mu_k, n_k, _ = load_tsv(args.k_tsv)
    if not np.allclose(mu_n, mu_k):
        print("error: mu grids differ between nomom and k TSVs", file=sys.stderr)
        return 1
    beta = meta["beta"]
    p = {k: meta[k] for k in ("t", "tp", "U", "V") if k in meta}
    n_sites = lx * ly
    levels = all_eigenlevels_rect(lx, ly, p, 0.0, 0.0)
    n_ed = np.array([grand_canonical_n(beta, float(m), levels, n_sites) for m in mu_n])

    d_ne = float(np.max(np.abs(n_n - n_ed)))
    d_ke = float(np.max(np.abs(n_k - n_ed)))
    d_nk = float(np.max(np.abs(n_n - n_k)))

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(mu_n, n_n, "-", lw=2.0, color="C0", label="FTLM non-momentum")
    ax.plot(mu_n, n_ed, ":", lw=2.0, color="C2", label="Exact diagonalization (ED)")
    ax.plot(mu_n, n_k, "--", lw=2.0, color="C3", label="FTLM momentum blocks (pure)")
    ax.set_xlabel(r"chemical potential $\mu$")
    ax.set_ylabel(r"$n = \langle \hat N\rangle / N_{\mathrm{sites}}$")
    ax.set_title(f"{lx}×{ly} benchmark: FTLM non-momentum vs momentum vs ED")
    ax.set_ylim(-0.05, 2.15)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.text(
        0.5,
        0.02,
        rf"max|nonmom−ED| = {d_ne:.3e}   max|k−ED| = {d_ke:.3e}   max|nonmom−k| = {d_nk:.3e}",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"max|nonmom-ED|={d_ne:.3e} max|k-ED|={d_ke:.3e} max|nonmom-k|={d_nk:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
