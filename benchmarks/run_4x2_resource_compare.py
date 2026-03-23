#!/usr/bin/env python3
"""
Run 4×2 ED (NumPy block diagonalization) vs C++ matrix-free Lanczos sampling (FTLM precursor)
and write a short comparison of wall time and peak RSS.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_BUILD = REPO_ROOT / "build" / "ftlm_bench_4x2_ftlm"


def run_ed(py: str, argv: list[str]) -> str:
    r = subprocess.run([py, str(BENCH_DIR / "ed_rect_gc_report.py")] + argv, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise SystemExit(r.returncode)
    return r.stdout


def run_ftlm(bin_path: Path, argv: list[str]) -> str:
    r = subprocess.run([str(bin_path)] + argv, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise SystemExit(r.returncode)
    return r.stdout


def extract_metric(line: str) -> dict[str, str]:
    if "METRIC " not in line:
        return {}
    out = {}
    for part in line.split():
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ftlm-bin", type=Path, default=DEFAULT_BUILD)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", type=Path, default=BENCH_DIR / "data" / "4x2_resource_report.txt")
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--n-mu", type=int, default=221)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--lanczos-steps", type=int, default=128)
    args = ap.parse_args()

    if not args.ftlm_bin.is_file():
        print(f"Missing {args.ftlm_bin} — build with: cmake --build build --target ftlm_bench_4x2_ftlm", file=sys.stderr)
        return 1

    ed_argv = [
        "--lx=4",
        "--ly=2",
        f"--beta={args.beta}",
        f"--n-mu={args.n_mu}",
    ]
    ftlm_argv = [
        f"--samples={args.samples}",
        f"--lanczos-steps={args.lanczos_steps}",
    ]

    out_ed = run_ed(args.python, ed_argv)
    out_ft = run_ftlm(args.ftlm_bin, ftlm_argv)

    lines = []
    lines.append("# 4×2 PBC resource comparison (same Hubbard defaults as 2×2 n(μ) benchmark)\n")
    lines.append("# ED: NumPy eigvalsh in each (N_up,N_down) block + grand-canonical μ loop.\n")
    lines.append("# FTLM: stochastic Lanczos extrema in (4,4) sector — precursor workload, not full FTLM trace.\n\n")

    ed_m = {}
    for ln in out_ed.splitlines():
        lines.append(f"ed: {ln}\n")
        if ln.startswith("METRIC"):
            ed_m = extract_metric(ln)

    ft_m = {}
    for ln in out_ft.splitlines():
        lines.append(f"ftlm: {ln}\n")
        if ln.startswith("METRIC"):
            ft_m = extract_metric(ln)

    lines.append("\n## Summary\n")
    if ed_m:
        lines.append(
            f"| ED (NumPy block)   | wall_total_s={ed_m.get('wall_total_s', '?')} | max_rss_mib={ed_m.get('max_rss_mib', '?')} | levels={ed_m.get('levels', '?')} |\n"
        )
    if ft_m:
        lines.append(
            f"| FTLM precursor     | wall_s={ft_m.get('wall_s', '?')} | max_rss_mib={ft_m.get('max_rss_mib', '?')} | matvec={ft_m.get('matvec_count', '?')} | dim={ft_m.get('dim', '?')} |\n"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines))
    print(f"wrote {args.out}")
    print("".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
