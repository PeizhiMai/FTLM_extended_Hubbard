"""
Rectangular PBC extended Hubbard: geometry matching src/lattice.cpp and sector exact diagonalization.
Used by plot_n_vs_mu.py and ed_rect_gc_report.py.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np


def site_index(x: int, y: int, lx: int) -> int:
    return y * lx + x


def peierls(wx: int, wy: int, phi_x: float, phi_y: float) -> complex:
    return complex(math.cos(wx * phi_x + wy * phi_y), math.sin(wx * phi_x + wy * phi_y))


def append_nn_pair(pairs: list[tuple[int, int]], a: int, b: int) -> None:
    if a == b:
        return
    if a > b:
        a, b = b, a
    pairs.append((a, b))


def add_directed(
    hops: list[tuple[int, int, int, complex]],
    i: int,
    j: int,
    phase: complex,
    t_strength: float,
) -> None:
    if abs(t_strength) < 1e-15:
        return
    c_ij = -t_strength * phase
    c_ji = -t_strength * phase.conjugate()
    for spin in (0, 1):
        hops.append((i, j, spin, c_ij))
        hops.append((j, i, spin, c_ji))


def build_hubbard_geometry_py(
    lx: int, ly: int, t: float, tp: float, phi_x: float, phi_y: float
) -> tuple[list[tuple[int, int, int, complex]], list[tuple[int, int]]]:
    """Mirror src/lattice.cpp build_hubbard_geometry (directed hops + NN pairs for V)."""
    hops: list[tuple[int, int, int, complex]] = []
    pairs: list[tuple[int, int]] = []
    for y in range(ly):
        for x in range(lx):
            i = site_index(x, y, lx)
            if lx > 1:
                wx, nx = 0, x + 1
                if nx >= lx:
                    nx = 0
                    wx = 1
                j = site_index(nx, y, lx)
                if i != j:
                    append_nn_pair(pairs, i, j)
                    ph = peierls(wx, 0, phi_x, phi_y)
                    add_directed(hops, i, j, ph, t)
            if ly > 1:
                wy, ny = 0, y + 1
                if ny >= ly:
                    ny = 0
                    wy = 1
                j = site_index(x, ny, lx)
                if i != j:
                    append_nn_pair(pairs, i, j)
                    ph = peierls(0, wy, phi_x, phi_y)
                    add_directed(hops, i, j, ph, t)

            def diag(dx: int, dy: int) -> None:
                wx = wy = 0
                nx, ny = x + dx, y + dy
                if nx >= lx:
                    nx -= lx
                    wx = 1
                elif nx < 0:
                    nx += lx
                    wx = -1
                if ny >= ly:
                    ny -= ly
                    wy = 1
                elif ny < 0:
                    ny += ly
                    wy = -1
                j = site_index(nx, ny, lx)
                if i != j:
                    ph = peierls(wx, wy, phi_x, phi_y)
                    add_directed(hops, i, j, ph, tp)

            for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                diag(dx, dy)

    return hops, pairs


def fermions_between(ma: int, mb: int, u: int, d: int) -> int:
    if ma > mb:
        ma, mb = mb, ma
    c = 0
    for m in range(ma + 1, mb):
        site = m // 2
        if m % 2 == 0:
            if (u >> site) & 1:
                c += 1
        else:
            if (d >> site) & 1:
                c += 1
    return c


def enumerate_sector(n_sites: int, nu: int, nd: int) -> list[tuple[int, int]]:
    ups = []
    for comb in combinations(range(n_sites), nu):
        u = 0
        for k in comb:
            u |= 1 << k
        ups.append(u)
    downs = []
    for comb in combinations(range(n_sites), nd):
        d = 0
        for k in comb:
            d |= 1 << k
        downs.append(d)
    return [(u, d) for u in ups for d in downs]


def build_h_sector(
    basis: list[tuple[int, int]],
    hops: list[tuple[int, int, int, complex]],
    pairs: list[tuple[int, int]],
    u_hub: float,
    v_nn: float,
) -> np.ndarray:
    imap = {st: k for k, st in enumerate(basis)}
    dim = len(basis)
    h = np.zeros((dim, dim), dtype=np.complex128)

    def occ(u: int, d: int, site: int) -> int:
        return ((u >> site) & 1) + ((d >> site) & 1)

    for j, (u0, d0) in enumerate(basis):
        du = u_hub * bin(u0 & d0).count("1")
        for i_s, j_s in pairs:
            du += v_nn * occ(u0, d0, i_s) * occ(u0, d0, j_s)
        h[j, j] = du

        for fr, to, spin, coeff in hops:
            c = complex(coeff)
            if spin == 0:
                if not ((u0 >> fr) & 1):
                    continue
                if (u0 >> to) & 1:
                    continue
                up = u0 ^ (1 << fr) ^ (1 << to)
                dn = d0
            else:
                if not ((d0 >> fr) & 1):
                    continue
                if (d0 >> to) & 1:
                    continue
                up = u0
                dn = d0 ^ (1 << fr) ^ (1 << to)
            ma = 2 * fr + spin
            mb = 2 * to + spin
            sg = -1.0 if fermions_between(ma, mb, u0, d0) & 1 else 1.0
            irow = imap[(up, dn)]
            h[irow, j] += sg * c

    if np.max(np.abs(h - np.conjugate(h.T))) > 1e-10:
        raise ValueError("H is not Hermitian (check geometry / signs)")
    return h


def all_eigenlevels_rect(
    lx: int,
    ly: int,
    p: dict[str, float],
    phi_x: float = 0.0,
    phi_y: float = 0.0,
) -> list[tuple[float, int]]:
    n_sites = lx * ly
    hops, pairs = build_hubbard_geometry_py(lx, ly, p["t"], p["tp"], phi_x, phi_y)
    levels: list[tuple[float, int]] = []
    for nu in range(n_sites + 1):
        for nd in range(n_sites + 1):
            basis = enumerate_sector(n_sites, nu, nd)
            if not basis:
                continue
            h = build_h_sector(basis, hops, pairs, p["U"], p["V"])
            w = np.linalg.eigvalsh((h + np.conjugate(h.T)) * 0.5)
            n_el = nu + nd
            for ev in w:
                levels.append((float(ev.real), n_el))
    return levels


def grand_canonical_n(
    beta: float, mu: float, levels: list[tuple[float, int]], n_sites: int
) -> float:
    max_a = max(-beta * (e - mu * float(n)) for e, n in levels)
    sum_w = 0.0
    sum_nw = 0.0
    for e, n in levels:
        a = -beta * (e - mu * float(n))
        w = math.exp(a - max_a)
        sum_w += w
        sum_nw += float(n) * w
    return (sum_nw / sum_w) / float(n_sites)
