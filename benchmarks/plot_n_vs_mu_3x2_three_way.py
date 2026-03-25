#!/usr/bin/env python3
"""Plot 3x2 n(mu): FTLM non-momentum + momentum k-blocks + ED (same grid as TSVs)."""
from __future__ import annotations

import argparse
import re
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

_METRIC_LINE = re.compile(
    r"^METRIC\s+kind=(?P<kind>\S+)\s+.*?"
    r"wall_time_s=(?P<wall>[\d.eE+-]+)\s+.*?"
    r"peak_rss_bytes=(?P<rss_b>[\d.eE+-]+)\s+.*?"
    r"peak_rss_mib=(?P<rss_mib>[\d.eE+-]+)",
    re.DOTALL,
)


def parse_monitoring_metrics(log_text: str) -> dict[str, tuple[float, float, float]]:
    """kind -> (wall_time_s, peak_rss_bytes, peak_rss_mib) from METRIC lines."""
    out: dict[str, tuple[float, float, float]] = {}
    for line in log_text.splitlines():
        m = _METRIC_LINE.match(line.strip())
        if not m:
            continue
        kind = m.group("kind")
        out[kind] = (float(m.group("wall")), float(m.group("rss_b")), float(m.group("rss_mib")))
    return out


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
    ap.add_argument(
        "--monitoring-log",
        type=Path,
        default=REPO_ROOT / "benchmarks/data/n_vs_mu_3x2_beta20_three_way_monitoring.txt",
        help="Log from run_3x2_three_way_monitoring.sh (METRIC lines for wall time + peak RSS).",
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

    metrics: dict[str, tuple[float, float, float]] = {}
    log_path = args.monitoring_log
    if log_path.is_file():
        try:
            metrics = parse_monitoring_metrics(log_path.read_text(encoding="utf-8", errors="replace"))
        except OSError as e:
            print(f"warning: could not read monitoring log {log_path}: {e}", file=sys.stderr)

    print("")
    print("--- 3x2 three-way report (numerics + run time + peak RSS) ---")
    print(
        f"  numerics: max|nonmom-ED|={d_ne:.3e}  max|k-ED|={d_ke:.3e}  max|nonmom-k|={d_nk:.3e}"
    )
    if "FTLM_nmu_rect" in metrics:
        w, rb, rm = metrics["FTLM_nmu_rect"]
        print(
            f"  FTLM non-momentum: wall_time_s={w:.4g}  peak_RSS_MiB={rm:.4g}  peak_RSS_bytes={rb:.5g}"
        )
    else:
        print(
            "  FTLM non-momentum: (no METRIC kind=FTLM_nmu_rect in monitoring log — run benchmarks/run_3x2_three_way_monitoring.sh)"
        )
    if "FTLM_nmu_rect_k" in metrics:
        w, rb, rm = metrics["FTLM_nmu_rect_k"]
        print(
            f"  FTLM k-blocks:     wall_time_s={w:.4g}  peak_RSS_MiB={rm:.4g}  peak_RSS_bytes={rb:.5g}"
        )
    else:
        print(
            "  FTLM k-blocks:     (no METRIC kind=FTLM_nmu_rect_k in monitoring log — run benchmarks/run_3x2_three_way_monitoring.sh)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
