#!/usr/bin/env python3
"""Plot dense vs Lanczos Ritz spectrum from `ftlm_bench_2x2 --write-plot-data=DIR`."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing dense_evals.txt and lanczos_ritz.txt",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("bench_2x2_spectrum.png"),
        help="Output PNG path",
    )
    args = p.parse_args()
    ddir = args.data_dir
    dense = np.loadtxt(ddir / "dense_evals.txt")
    ritz = np.loadtxt(ddir / "lanczos_ritz.txt")
    if dense.ndim == 0:
        dense = np.array([float(dense)])
    if ritz.ndim == 0:
        ritz = np.array([float(ritz)])
    dense.sort()
    ritz.sort()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.plot(np.arange(len(dense)), dense, "o-", ms=4, lw=1.2, label="dense (full H)")
    ax.plot(np.arange(len(ritz)), ritz, "s", ms=5, mfc="none", lw=0, label="Lanczos Ritz (T_k)")
    ax.set_xlabel("sorted index")
    ax.set_ylabel("energy")
    ax.set_title("Spectrum match (2×2 cluster sector)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.axvline(dense[0], color="C0", ls="-", lw=2, alpha=0.85, label=f"E_min dense = {dense[0]:.6f}")
    ax2.axvline(dense[-1], color="C0", ls="--", lw=2, alpha=0.85, label=f"E_max dense = {dense[-1]:.6f}")
    ax2.axvline(ritz[0], color="C1", ls="-", lw=1.5, alpha=0.9, label=f"E_min Ritz = {ritz[0]:.6f}")
    ax2.axvline(ritz[-1], color="C1", ls="--", lw=1.5, alpha=0.9, label=f"E_max Ritz = {ritz[-1]:.6f}")
    ax2.set_xlabel("energy")
    ax2.set_yticks([])
    ax2.set_title("Extrema (vertical lines)")
    ax2.legend(loc="upper center", fontsize=8)
    ax2.set_xlim(min(dense[0], ritz[0]) - 0.5, max(dense[-1], ritz[-1]) + 0.5)

    meta = ddir / "meta.txt"
    if meta.is_file():
        fig.suptitle(meta.read_text().strip().replace("\n", "  "), fontsize=8, y=1.02)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
