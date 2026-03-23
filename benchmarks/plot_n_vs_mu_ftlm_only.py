#!/usr/bin/env python3
"""
Plot n(μ) from a `ftlm_bench_ftlm_nmu_rect` TSV (FTLM only — no ED).

Reads optional `# wall_time_s=... peak_rss_mib=...` footer written by the benchmark when monitoring is on.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_tsv_header(line: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("beta", "t", "tp", "U", "V", "wall_time_s", "peak_rss_mib", "peak_rss_bytes", "sector_time_s"):
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
    ap = argparse.ArgumentParser(description="Plot n(mu) from FTLM TSV (no ED)")
    ap.add_argument("--tsv", type=Path, required=True, help="Output of ftlm_bench_ftlm_nmu_rect")
    ap.add_argument("--out", type=Path, default=None, help="PNG path (default: benchmarks/data/...)")
    ap.add_argument("--no-show", action="store_true", help="Do not open interactive window")
    args = ap.parse_args()

    if not args.tsv.is_file():
        print(f"Missing TSV: {args.tsv}", file=sys.stderr)
        return 1

    mu, n, meta = load_tsv(args.tsv)
    beta = meta.get("beta")
    if beta is None:
        print("TSV missing beta= in a # header line", file=sys.stderr)
        return 1

    t = meta.get("t", float("nan"))
    tp = meta.get("tp", float("nan"))
    u = meta.get("U", float("nan"))
    v = meta.get("V", float("nan"))
    wall = meta.get("wall_time_s")
    rss_mib = meta.get("peak_rss_mib")

    if args.out is None:
        args.out = REPO_ROOT / "benchmarks" / "data" / f"n_vs_mu_ftlm_only_{args.tsv.stem}.png"

    subtitle = ""
    if wall is not None and rss_mib is not None:
        subtitle = rf"wall {wall:.1f} s, peak RSS {rss_mib:.1f} MiB"
    elif wall is not None:
        subtitle = rf"wall {wall:.1f} s"

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(mu, n, "-", lw=2.0, color="C0", label="FTLM")
    ax.set_xlabel(r"chemical potential $\mu$")
    ax.set_ylabel(r"$n = \langle \hat N\rangle / N_{\mathrm{sites}}$")
    title = rf"FTLM grand canonical, $\beta={beta:g}$  $t={t}\ t'={tp}\ U={u}\ V={v}$"
    ax.set_title(title + ("\n" + subtitle if subtitle else ""))
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
