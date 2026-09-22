"""GJK 单纯形上的最近点子算法（Johnson 距离算法风格的子集枚举）。

降维策略（不使用随机抖动）：
- 对当前单纯形（<=4 个 Minkowski 差顶点）枚举全部非空子集；
- 用「棱向量 Gram 矩阵」求原点到子集仿射包的最近点：
  以 y0 为基点，G_ij = e_i·e_j（e_i = y_i - y0），解 G t = -E^T y0。
  这样即使仿射包穿过原点（两物体刚好接触），矩阵也保持良态——
  若改用环境坐标 Y^T Y 会在此情形奇异；
- 棱向量 Gram 奇异 = 仿射相关（重复/共线/共面的冗余点），
  跳过该子集即可：同一凸包必由其某个仿射独立真子集表示，
  而真子集同样在枚举范围内。于是退化线段自动退到端点、
  退化三角形自动退到边/顶点，无需扰动坐标；
- 权值全部非负的子集才可行，取最近点范数最小者。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .vectors import Vec3, dot, norm_sq, sub

# 棱向量 Gram 主元相对阈值：小于最大棱长平方的该比例即视为仿射相关。
RANK_REL = 1e-13
# 权值负零点：绝对值小于该比例的负权按舍入噪声处理。
NEG_W_REL = 1e-12
# 两个候选点视为等距的相对噪声带。
TIE_REL = 1e-18

_SINGULAR = "singular"
_INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class MinkowskiVertex:
    """Minkowski 差 A-B 的一个顶点 p = a - b 及其来源见证。"""

    p: Vec3
    ai: int  # A 上取到 support 的顶点下标
    bi: int  # B 上取到 support 的顶点下标


@dataclass(frozen=True)
class ClosestSolution:
    """最近点子算法结果。"""

    point: Vec3                     # conv(simplex) 上距原点最近的点
    weights: tuple[float, ...]      # 与 simplex 顶点对齐的非负权重，和为 1
    mask: int                       # 获胜子集位掩码
    subset_size: int
    norm: float
    singular_skipped: int = 0       # 诊断：因仿射相关被跳过的子集数


def _solve_symmetric(
    g: list[list[float]], rhs: list[float], pivot_scale: float
) -> list[float] | None:
    """部分主元高斯消元解小对称系统；数值秩亏时返回 None。"""
    k = len(g)
    a = [row[:] + [rhs[i]] for i, row in enumerate(g)]
    threshold = RANK_REL * max(pivot_scale, 1e-300)
    for col in range(k):
        piv = col
        best = abs(a[col][col])
        for r in range(col + 1, k):
            if abs(a[r][col]) > best:
                best = abs(a[r][col])
                piv = r
        if best <= threshold:
            return None
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
        pivot = a[col][col]
        for r in range(col + 1, k):
            factor = a[r][col] / pivot
            if factor != 0.0:
                for c in range(col, k + 1):
                    a[r][c] -= factor * a[col][c]
    x = [0.0] * k
    for i in range(k - 1, -1, -1):
        s = a[i][k]
        for j in range(i + 1, k):
            s -= a[i][j] * x[j]
        piv = a[i][i]
        if abs(piv) <= threshold:
            return None
        x[i] = s / piv
    return x


def _subset_closest(
    verts: Sequence[MinkowskiVertex], mask: int
) -> tuple[Vec3, tuple[float, ...]] | str:
    """原点到某子集凸包的最近点。

    返回 (point, local_weights)；权为负返回 ``_INFEASIBLE``；
    仿射相关返回 ``_SINGULAR``（交给真子集，无抖动）。
    """
    idxs = [i for i in range(len(verts)) if (mask >> i) & 1]
    k = len(idxs)
    y0 = verts[idxs[0]].p
    if k == 1:
        return y0, (1.0,)

    edges = [sub(verts[idxs[j]].p, y0) for j in range(1, k)]
    m = k - 1
    g = [[dot(edges[a], edges[b]) for b in range(m)] for a in range(m)]
    rhs = [-dot(edges[a], y0) for a in range(m)]
    pivot_scale = max(dot(e, e) for e in edges)
    t = _solve_symmetric(g, rhs, pivot_scale)
    if t is None:
        return _SINGULAR

    w_local = [1.0 - sum(t)] + t
    max_abs = max((abs(w) for w in w_local), default=1.0)
    neg_tol = NEG_W_REL * max(max_abs, 1.0)
    if any(w < -neg_tol for w in w_local):
        return _INFEASIBLE
    w_local = [0.0 if w < 0.0 else w for w in w_local]
    total = sum(w_local)
    if total <= 0.0:
        return _INFEASIBLE
    w_local = [w / total for w in w_local]

    px = py = pz = 0.0
    for j in range(k):
        v = verts[idxs[j]].p
        w = w_local[j]
        px += w * v[0]
        py += w * v[1]
        pz += w * v[2]
    return (px, py, pz), tuple(w_local)


def closest_on_simplex(verts: Sequence[MinkowskiVertex]) -> ClosestSolution:
    """枚举所有可行子集，返回距原点最近的代表。"""
    n = len(verts)
    best: tuple[Vec3, tuple[float, ...], int, int, float] | None = None
    singular = 0
    for mask in range(1, 1 << n):
        res = _subset_closest(verts, mask)
        if res == _SINGULAR:
            singular += 1
            continue
        if res == _INFEASIBLE:
            continue
        point, w_local = res  # type: ignore[misc]
        d2 = norm_sq(point)
        size = mask.bit_count()
        idxs = [i for i in range(n) if (mask >> i) & 1]
        w_full = [0.0] * n
        for j, i in enumerate(idxs):
            w_full[i] = w_local[j]
        candidate = (point, tuple(w_full), mask, size, d2)
        if best is None:
            best = candidate
        elif d2 < best[4] - TIE_REL * max(best[4], d2, 1.0):
            best = candidate
        elif abs(d2 - best[4]) <= TIE_REL * max(best[4], d2, 1.0) and size < best[3]:
            # 距离相同：优先更简单（顶点更少）的见证
            best = candidate
    if best is None:  # 理论上不可能：单点子集恒可行
        raise RuntimeError("最近点子算法失败：无可行子集（不应发生）")
    return ClosestSolution(
        point=best[0],
        weights=best[1],
        mask=best[2],
        subset_size=best[3],
        norm=math.sqrt(best[4]),
        singular_skipped=singular,
    )
