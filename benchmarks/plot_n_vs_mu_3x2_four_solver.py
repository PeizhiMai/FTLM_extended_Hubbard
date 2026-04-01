#!/usr/bin/env python3
"""Plot 3×2 n(μ): four C++ solvers; writes performance TSV (wall time, peak RSS) from METRIC lines."""
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

from plot_n_vs_mu import load_tsv

_GIB = 1024.0**3


def peak_rss_bytes_to_gb(rss_b: float) -> float:
    """Binary GB (GiB): peak RSS bytes / 1024^3."""
    return rss_b / _GIB


# METRIC kind=FTLM_nmu_rect_k solver=k-gram wall_time_s=... wall_sector_logZ_s=... peak_rss_bytes=... peak_rss_mib=...
_METRIC = re.compile(
    r"^METRIC\s+kind=\S+\s+solver=(?P<solver>[^\s]+)\s+"
    r".*?wall_time_s=(?P<wall>[\d.eE+-]+)\s+.*?wall_sector_logZ_s=(?P<wall_sec>[\d.eE+-]+)\s+"
    r".*?peak_rss_bytes=(?P<rss_b>[\d.eE+-]+)\s+.*?peak_rss_mib=(?P<rss_mib>[\d.eE+-]+)",
    re.DOTALL,
)


def parse_four_solver_metrics(log_text: str) -> dict[str, tuple[float, float, float, float]]:
    """solver -> (wall_time_s, wall_sector_logZ_s, peak_rss_bytes, peak_rss_mib). Last METRIC per solver wins."""
    out: dict[str, tuple[float, float, float, float]] = {}
    for line in log_text.splitlines():
        m = _METRIC.match(line.strip())
        if not m:
            continue
        sol = m.group("solver")
        out[sol] = (
            float(m.group("wall")),
            float(m.group("wall_sec")),
            float(m.group("rss_b")),
            float(m.group("rss_mib")),
        )
    return out


def write_performance_tsv(path: Path, metrics: dict[str, tuple[float, float, float, float]]) -> None:
    """Tab-separated summary for scripts; includes readable comment block."""
    order = [("nonmom", "non-momentum FTLM"), ("k-gram", "k-block Gram"), ("k-orbit", "k-block orbit"), ("ed", "ED (dense per k-block)")]
    lines = [
        "# 3×2 four-solver benchmark — runtime and peak RSS from METRIC lines (C++ getrusage RUSAGE_SELF).",
        "# Executable: ftlm_bench_ftlm_nmu_rect_k",
        "# peak_rss_gb = peak_rss_bytes / 1024^3 (binary GB, i.e. GiB).",
        "#",
        "# solver\tlabel\twall_time_s\twall_sector_logZ_s\tpeak_rss_gb",
    ]
    for key, label in order:
        if key in metrics:
            w, ws, rb, _rm = metrics[key]
            rg = peak_rss_bytes_to_gb(rb)
            lines.append(f"{key}\t{label}\t{w:.8g}\t{ws:.8g}\t{rg:.10g}")
        else:
            lines.append(f"{key}\t{label}\t(missing)\t(missing)\t(missing)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="3×2 four-solver n(mu) comparison plot")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks/data",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks/data/n_vs_mu_3x2_beta20_four_solver.png",
    )
    ap.add_argument(
        "--monitoring-log",
        type=Path,
        default=REPO_ROOT / "benchmarks/data/n_vs_mu_3x2_beta20_four_solver_monitoring.txt",
    )
    ap.add_argument(
        "--performance-out",
        type=Path,
        default=REPO_ROOT / "benchmarks/data/n_vs_mu_3x2_beta20_four_solver_performance.tsv",
        help="TSV of wall time + peak RSS (GB) per solver (from METRIC lines in monitoring log).",
    )
    args = ap.parse_args()
    d = args.data_dir

    paths = {
        "nonmom": d / "n_vs_mu_3x2_beta20_solver_nonmom.tsv",
        "k-gram": d / "n_vs_mu_3x2_beta20_solver_k_gram.tsv",
        "k-orbit": d / "n_vs_mu_3x2_beta20_solver_k_orbit.tsv",
        "ed": d / "n_vs_mu_3x2_beta20_solver_ed.tsv",
    }
    for label, p in paths.items():
        if not p.is_file():
            print(f"error: missing TSV for {label}: {p} (run benchmarks/run_3x2_four_solver_benchmark.sh)", file=sys.stderr)
            return 1

    mu_ref, n_nonmom, _ = load_tsv(paths["nonmom"])
    n_kg = load_tsv(paths["k-gram"])[1]
    n_ko = load_tsv(paths["k-orbit"])[1]
    n_ed = load_tsv(paths["ed"])[1]

    for label in ("k-gram", "k-orbit", "ed"):
        mu_i, _, _ = load_tsv(paths[label])
        if not np.allclose(mu_ref, mu_i):
            print(f"error: μ grid mismatch vs nonmom for {label}", file=sys.stderr)
            return 1

    curves: list[tuple[str, np.ndarray, str]] = [
        ("nonmom", n_nonmom, "C0"),
        ("k-gram", n_kg, "C1"),
        ("k-orbit", n_ko, "C2"),
        ("ed", n_ed, "C3"),
    ]
    d_g_o = float(np.max(np.abs(n_kg - n_ko)))
    d_g_e = float(np.max(np.abs(n_kg - n_ed)))
    d_o_e = float(np.max(np.abs(n_ko - n_ed)))
    d_n_g = float(np.max(np.abs(n_nonmom - n_kg)))

    styles = [("-", 2.0), ("--", 2.0), ("-.", 2.0), (":", 2.2)]
    labels_plot = [
        "non-momentum FTLM",
        "k-block Gram",
        "k-block orbit",
        "ED (per k-block, dense)",
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=140)
    for i, ((key, n_arr, color), (ls, lw), lab) in enumerate(zip(curves, styles, labels_plot)):
        ax.plot(mu_ref, n_arr, ls, lw=lw, color=color, label=lab)

    ax.set_xlabel(r"chemical potential $\mu$")
    ax.set_ylabel(r"$n = \langle \hat N\rangle / N_{\mathrm{sites}}$")
    ax.set_title("3×2 benchmark: four solvers (β=20)")
    ax.set_ylim(-0.05, 2.15)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)

    fig.text(
        0.5,
        0.02,
        rf"max|nonmom−k-gram|={d_n_g:.3e}  max|k-gram−k-orbit|={d_g_o:.3e}  max|k-gram−ed|={d_g_e:.3e}  max|k-orbit−ed|={d_o_e:.3e}",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

    metrics: dict[str, tuple[float, float, float, float]] = {}
    if args.monitoring_log.is_file():
        try:
            metrics = parse_four_solver_metrics(args.monitoring_log.read_text(encoding="utf-8", errors="replace"))
        except OSError as e:
            print(f"warning: could not read {args.monitoring_log}: {e}", file=sys.stderr)

    if metrics:
        write_performance_tsv(args.performance_out, metrics)
        print(f"wrote {args.performance_out}")

    print("")
    print("--- 3×2 four-solver report (numerics + wall time + peak RSS in GB) ---")
    print(
        f"  numerics: max|nonmom−k-gram|={d_n_g:.3e}  max|k-gram−k-orbit|={d_g_o:.3e}  "
        f"max|k-gram−ed|={d_g_e:.3e}  max|k-orbit−ed|={d_o_e:.3e}"
    )
    order = [("nonmom", "non-momentum"), ("k-gram", "k-block Gram"), ("k-orbit", "k-block orbit"), ("ed", "ED (dense)")]
    for key, title in order:
        if key in metrics:
            w, ws, rb, _rm = metrics[key]
            rg = peak_rss_bytes_to_gb(rb)
            print(
                f"  {title:22s}  wall_s={w:.4g}  sector_logZ_s={ws:.4g}  "
                f"peak_RSS_GB={rg:.6g}"
            )
        else:
            print(f"  {title:22s}  (no METRIC solver={key} in monitoring log)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
