"""独立参考实现（测试专用）。

本文件刻意 **不导入** convex_distance：所有期望值由解析公式或独立的
凸组合/活动集枚举推导。它只允许被测试调用，绝不能反过来调用被测核心。

包含：
- ``point_segment_distance``：点-线段解析最近距离；
- ``aabb_info``：轴对齐盒的解析最近距离与最近点（角点情形）；
- ``enumerative_distance``：在重心变量上枚举活动集、解 KKT 小系统，
  求两小点云凸包间距离（每个物体最近点至多含 4 个仿射独立点）；
- ``in_convex_hull``：枚举 <=4 个输入点的凸组合表示给定点（仿射解），
  用于独立核验被测返回的见证点确实落在凸包内。
"""

from __future__ import annotations

import itertools
import math
from typing import Sequence

Vec3 = tuple[float, float, float]


# ---------- 基础线性代数（独立于被测包） ----------

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(a[i] * b[i] for i in range(len(a)))


def _solve(A: list[list[float]], b: list[float]) -> list[float] | None:
    """高斯-约当消元（全选主元）；奇异返回 None。"""
    n = len(b)
    m = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-13:
            return None
        m[col], m[piv] = m[piv], m[col]
        d = m[col][col]
        for c in range(col, n + 1):
            m[col][c] /= d
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            if f != 0.0:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] for i in range(n)]


# ---------- 解析参考 ----------

def point_segment_distance(p: Vec3, a: Vec3, b: Vec3) -> tuple[float, Vec3]:
    """点 p 到线段 [a,b] 的解析最近距离与线段上最近点。"""
    ab = [b[i] - a[i] for i in range(3)]
    ap = [p[i] - a[i] for i in range(3)]
    denom = _dot(ab, ab)
    if denom == 0.0:
        return math.sqrt(_dot(ap, ap)), a
    t = _dot(ap, ab) / denom
    t = min(1.0, max(0.0, t))
    q = (a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2])
    return math.sqrt(sum((p[i] - q[i]) ** 2 for i in range(3))), q


def aabb_info(lo1: Vec3, hi1: Vec3, lo2: Vec3, hi2: Vec3
              ) -> tuple[float, Vec3, Vec3]:
    """两个轴对齐盒的解析最近距离及一对最近点。"""
    qa: list[float] = []
    qb: list[float] = []
    for k in range(3):
        if hi1[k] < lo2[k]:
            qa.append(hi1[k]); qb.append(lo2[k])
        elif hi2[k] < lo1[k]:
            qa.append(lo1[k]); qb.append(hi2[k])
        else:
            # 区间重叠：两最近点在该轴取同一值（取重叠区间内任意点，
            # 这里用中点的确定性选取）。
            lo = max(lo1[k], lo2[k])
            hi = min(hi1[k], hi2[k])
            v = 0.5 * (lo + hi)
            qa.append(v); qb.append(v)
    a = (qa[0], qa[1], qa[2])
    b = (qb[0], qb[1], qb[2])
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3))), a, b


# ---------- 枚举式凸包距离（独立 KKT 活动集） ----------

def enumerative_distance(P: Sequence[Vec3], Q: Sequence[Vec3]
                         ) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """枚举 KKT 活动集求 min |Σλp − Σμq|。

    3 维凸包最近点处的活动点各不超过 4（Carathéodory），
    对每个候选活动集解含两个等式约束的 KKT 线性系统，
    再独立验证对偶可行性与权重非负。
    返回 (distance, λ, μ)。
    """
    m, n = len(P), len(Q)
    cols: list[Vec3] = list(P) + [(-q[0], -q[1], -q[2]) for q in Q]
    N = m + n
    H = [[_dot(cols[i], cols[j]) for j in range(N)] for i in range(N)]
    best: tuple[float, tuple[float, ...], tuple[float, ...]] | None = None

    rng = range
    for ps in rng(1, min(m, 4) + 1):
        for qs in rng(1, min(n, 4) + 1):
            for I in itertools.combinations(range(m), ps):
                for J in itertools.combinations(range(m, m + n), qs):
                    F = list(I) + list(J)
                    k = len(F)
                    M = [[0.0] * (k + 2) for _ in range(k + 2)]
                    for a in range(k):
                        for b in range(k):
                            M[a][b] = H[F[a]][F[b]]
                        M[a][k] = 1.0 if F[a] < m else 0.0
                        M[a][k + 1] = 0.0 if F[a] < m else 1.0
                    for a in range(k):
                        M[k][a] = 1.0 if F[a] < m else 0.0
                        M[k + 1][a] = 0.0 if F[a] < m else 1.0
                    rhs = [0.0] * k + [1.0, 1.0]
                    x = _solve(M, rhs)
                    if x is None:
                        continue
                    wf, alpha, beta = x[:k], x[k], x[k + 1]
                    if any(w < -1e-9 for w in wf):
                        continue
                    w = [0.0] * N
                    for t, fi in enumerate(F):
                        w[fi] = max(0.0, wf[t])
                    # KKT 对偶可行性：g_i + alpha >= 0 (i∈P)，
                    # g_j + beta >= 0 (j∈Q)。
                    g = [sum(H[i][j] * w[j] for j in F) for i in range(N)]
                    if any(g[i] + (alpha if i < m else beta) < -1e-7
                           for i in range(N)):
                        continue
                    z = [sum(w[i] * cols[i][d] for i in range(N))
                         for d in range(3)]
                    d2 = _dot(z, z)
                    lam = tuple(w[:m])
                    mu = tuple(w[m:])
                    if best is None or d2 < best[0] ** 2:
                        best = (math.sqrt(d2), lam, mu)
    if best is None:
        raise AssertionError("枚举参考失败（不应发生，至少单顶点集可行）")
    return best


def in_convex_hull(points: Sequence[Vec3], x: Vec3, tol: float) -> bool:
    """x 是否能表示为 points 中 <=4 个点的非负凸组合（独立枚举）。

    枚举所有 <=4 个点的子集，解仿射方程 Σλp=x、Σλ=1，
    要求全部 λ >= -tol。仅适用于小点云（测试里 n<=10）。
    """
    pts = list(points)
    for size in range(1, min(4, len(pts)) + 1):
        for combo in itertools.combinations(range(len(pts)), size):
            sel = [pts[i] for i in combo]
            if size == 1:
                if all(abs(sel[0][d] - x[d]) <= tol for d in range(3)):
                    return True
                continue
            # 解 Σλp=x、Σλ=1（k 个未知量，4 个方程）。
            # 枚举从 3 个坐标方程中选 k-1 个的所有组合，避免选到
            # 坐标相关的行而误判奇异（如两点 x 坐标相同）。
            k = size
            for chosen in itertools.combinations(range(3), k - 1):
                A = [[sel[j][r] for j in range(k)] for r in chosen]
                A.append([1.0] * k)
                b = [x[r] for r in chosen] + [1.0]
                lam = _solve(A, b)
                if lam is None:
                    continue
                if any(w < -tol for w in lam):
                    continue
                for d in range(3):
                    got = sum(lam[j] * sel[j][d] for j in range(k))
                    if abs(got - x[d]) > tol:
                        break
                else:
                    return True
    return False
