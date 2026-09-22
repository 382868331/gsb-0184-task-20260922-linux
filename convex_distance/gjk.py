"""GJK 距离主循环：Minkowski 差 A-B 到原点的最近点。

每轮迭代：
1. 在当前单纯形（<=4 顶点）上用 Johnson 风格子集枚举求最近点 v
   与凸组合权重（见 simplex.py），权重同时给出两物体上的见证点；
2. 沿 d = -v/|v| 取新 support 顶点 w = h_A(d) - h_B(-d)；
3. 残差 r = |v| + d·w（当前上界 |v| 与 support 下界 -d·w 之差），
   r <= 容差即给出可核验的分离证书；
4. 否则按获胜子集裁剪单纯形并加入 w，继续。

状态语义严格区分：
- ``intersecting``：原点在凸包内的数值证书（最近点范数 <= 机器接触地板）；
- ``separated``：support 残差收敛（含结构条件成立的停滞）的分离证书；
- ``not_converged``：不发任何证书，``reason`` 进一步区分
  ``budget_exhausted``（迭代预算用完）、``machine_floor_ambiguity``
  （残差舍入地板与当前距离同阶，双精度无法区分接触与微小缝隙）、
  ``numerical_contradiction``（停滞但残差超出地板的数值矛盾）。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

from .bodies import ConvexBody
from .errors import (
    MAX_ITERATIONS,
    validate_float_param,
    validate_int_param,
)
from .simplex import MinkowskiVertex, closest_on_simplex
from .vectors import (
    Vec3,
    dot,
    norm,
    norm_sq,
    scale,
    sub,
)

SEPARATED = "separated"
INTERSECTING = "intersecting"
NOT_CONVERGED = "not_converged"

# IEEE-754 双精度机器精度。
_EPS = sys.float_info.epsilon
# 原点是否「在凸包内」以及活动迭代残差的机器舍入地板（平坦，O(eps*R)，
# 常数由尺度扫描实验标定，见 README「误差口径」）。
CONTACT_FLOOR = 64.0
ACTIVE_FLOOR = 64.0
# 停滞时 Gram 方向误差造成的残差放大地板：
# eps*R*max(1, R/|v|)。只在该地板严格小于 |v| 时允许发分离证书——
# 即要求结构条件 d·w < -(1-AMBIGUITY_RATIO)*|v|（支撑面越过原点）。
# 地板与 |v| 同阶意味着双精度已无法区分「接触」与「微小缝隙」，
# 此时返回 not_converged，绝不在无结构保证下硬判分离。
STAG_FLOOR = 32.0
AMBIGUITY_RATIO = 0.25


@dataclass(frozen=True)
class DistanceResult:
    """``distance`` 的返回值（不可变）。

    字段：
    - ``status``：``"separated" | "intersecting" | "not_converged"``；
    - ``converged``：是否在预算内取得证书（separated/intersecting 为 True）；
    - ``distance``：最近距离；intersecting 为 0.0；not_converged 为估计值；
    - ``witness_a`` / ``witness_b``：两物体上的最近点见证（凸组合）；
    - ``separation_direction``：单位向量 n（由 B 指向 A，满足
      witness_a = witness_b + distance*n），intersecting/not_converged 为 None；
    - ``residual``：最终 support 残差 r = |v| + d·w（>=0）；
    - ``witness_residual``：|witness_a - witness_b - v|，证书自洽残差；
    - ``iterations``：实际迭代（最近点子问题求解）次数；
    - ``max_iterations``：本次预算；
    - ``singular_skipped``：累计被拒绝的退化（仿射相关）子集数——
      这些情形通过降维到真子集处理，没有任何坐标抖动；
    - ``reason``：``"certified"`` / ``"budget_exhausted"`` /
      ``"machine_floor_ambiguity"``（后者：残差地板与当前距离同阶，
      双精度无法区分接触与微小缝隙，按未收敛处理，不发证书）；
    - ``message``：状态的文字说明。
    """

    status: str
    converged: bool
    distance: float
    witness_a: Vec3
    witness_b: Vec3
    separation_direction: Vec3 | None
    residual: float
    witness_residual: float
    iterations: int
    max_iterations: int
    singular_skipped: int
    reason: str
    message: str

    @property
    def intersecting(self) -> bool:
        return self.status == INTERSECTING

    @property
    def separated(self) -> bool:
        return self.status == SEPARATED


def _coerce(body: ConvexBody | Iterable[Sequence[float]], name: str) -> ConvexBody:
    if isinstance(body, ConvexBody):
        return body
    return ConvexBody(body, name=name)  # 校验在构造内完成


def _centroid(points: Sequence[Vec3]) -> Vec3:
    n = float(len(points))
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


def _support_pair(
    A: ConvexBody, B: ConvexBody, direction: Vec3
) -> MinkowskiVertex:
    a, ai = A.support(direction)
    neg_d = (-direction[0], -direction[1], -direction[2])
    b, bi = B.support(neg_d)
    return MinkowskiVertex(sub(a, b), ai, bi)


def _witnesses(
    weights: Sequence[float],
    simplex: Sequence[MinkowskiVertex],
    A: ConvexBody,
    B: ConvexBody,
) -> tuple[Vec3, Vec3]:
    """按单纯形上的凸权重组合出两物体上的见证点。"""
    total = sum(weights)
    if total <= 0.0:
        raise RuntimeError("内部错误：见证权重和非正")
    ax = ay = az = bx = by = bz = 0.0
    for w_lambda, mv in zip(weights, simplex):
        if w_lambda == 0.0:
            continue
        lam = w_lambda / total
        a = A.points[mv.ai]
        b = B.points[mv.bi]
        ax += lam * a[0]; ay += lam * a[1]; az += lam * a[2]
        bx += lam * b[0]; by += lam * b[1]; bz += lam * b[2]
    return (ax, ay, az), (bx, by, bz)


def distance(
    A: ConvexBody | Iterable[Sequence[float]],
    B: ConvexBody | Iterable[Sequence[float]],
    *,
    max_iterations: int = MAX_ITERATIONS,
    tolerance_rel: float = 1e-12,
    tolerance_abs: float = 1e-12,
) -> DistanceResult:
    """求 conv(A) 与 conv(B) 的最近距离与分离证书。

    参数：
    - ``A``、``B``：``ConvexBody`` 或长度 3 点的 list/tuple（各 1..64 点，
      坐标有限，拒绝 NaN/Infinity，拒绝 bool 坐标）；
    - ``max_iterations``：1..128 的 int（拒绝 bool）；
    - ``tolerance_rel`` / ``tolerance_abs``：非负有限收敛容差。

    抛出：``TypeError`` / ``InvalidBodyError`` / ``InvalidParameterError``。
    本函数无状态、无副作用，输入非法时不会创建任何可观察的部分结果。
    """
    body_a = _coerce(A, "A")
    body_b = _coerce(B, "B")
    budget = validate_int_param(
        max_iterations, "max_iterations", lo=1, hi=MAX_ITERATIONS
    )
    tol_rel = validate_float_param(tolerance_rel, "tolerance_rel", positive=False)
    tol_abs = validate_float_param(tolerance_abs, "tolerance_abs", positive=False)

    pa, pb = body_a.points, body_b.points
    ca, cb = _centroid(pa), _centroid(pb)
    d0 = sub(ca, cb)
    if norm_sq(d0) == 0.0:
        d0 = (1.0, 0.0, 0.0)

    first = _support_pair(body_a, body_b, d0)
    simplex: list[MinkowskiVertex] = [first]
    singular_total = 0

    v = first.p
    weights = (1.0,)
    iterations = 0
    residual = math.inf

    while iterations < budget:
        iterations += 1
        sol = closest_on_simplex(simplex)
        singular_total += sol.singular_skipped
        v = sol.point
        weights = sol.weights
        nv = sol.norm

        # 构型半径尺度（Minkowski 差顶点相对原点的最大范数）。
        length_scale = max(norm(mv.p) for mv in simplex)
        # 接触地板：|v| 不经过大数相减，平坦 O(eps*R) 即可。
        contact_floor = tol_abs + CONTACT_FLOOR * _EPS * length_scale
        # 活动迭代（support 仍在推进）时残差 r=|v|+d·w 的舍入地板。
        active_floor = max(
            tol_abs + tol_rel * nv, ACTIVE_FLOOR * _EPS * length_scale
        )
        # 停滞时方向来自病态 Gram，方向误差放大残差，地板带 R/|v| 项。
        stag_floor = max(
            tol_abs + tol_rel * nv,
            STAG_FLOOR * _EPS * length_scale
            * max(1.0, length_scale / max(nv, contact_floor)),
        )

        # 接触判定只认「绝对用户容差 + 机器地板」，刻意不乘 tol_rel：
        # 否则在 R=1e12 的构型上 tol_rel*R 会把 1.5 的真实缝隙吞成相交，
        # 用宽容差掩盖边界错误（见 README「误差口径」）。
        if nv <= contact_floor:
            wa, wb = _witnesses(weights, simplex, body_a, body_b)
            witness_res = norm(sub(sub(wa, wb), v))
            return DistanceResult(
                status=INTERSECTING,
                converged=True,
                distance=0.0,
                witness_a=wa,
                witness_b=wb,
                separation_direction=None,
                residual=0.0,
                witness_residual=witness_res,
                iterations=iterations,
                max_iterations=budget,
                singular_skipped=singular_total,
                reason="certified",
                message=(
                    f"原点在 Minkowski 差凸包内（最近点范数 {nv:.3e} "
                    f"<= 接触地板 {contact_floor:.3e}）：两物体相交/接触"
                ),
            )

        direction = scale(v, -1.0 / nv)
        nxt = _support_pair(body_a, body_b, direction)
        residual = nv + dot(direction, nxt.p)
        if residual < 0.0 and residual > -active_floor:
            residual = 0.0  # 纯舍入造成的微小负值归零

        # 真正的停滞：新 support 顶点已是「当前单纯形」的成员。
        # 注意不能用全局历史集合——此前裁剪掉的顶点可以在新方向下
        # 再次成为极值点并带来实质进展（必须允许它重新加入）。
        stagnated = any(
            mv.ai == nxt.ai and mv.bi == nxt.bi for mv in simplex
        )

        note: str | None = None
        if not stagnated and residual <= active_floor:
            note = (
                f"活动迭代 support 残差 {residual:.3e} 收敛"
                f"（舍入地板 {active_floor:.3e}）"
            )
        elif stagnated and residual <= active_floor:
            note = (
                f"support 精确停滞且残差 {residual:.3e} 在平坦舍入地板 "
                f"{active_floor:.3e} 内"
            )
        elif (
            stagnated
            and stag_floor < AMBIGUITY_RATIO * nv
            and residual <= stag_floor
        ):
            # 放大地板严格小于当前距离：结构条件 d·w < -(1-c)|v|
            # 仍然成立（支撑面确实越过原点），方向误差只是噪声。
            note = (
                f"support 停滞；结构分离条件成立（残差 {residual:.3e} "
                f"<= 病态方向噪声地板 {stag_floor:.3e}，且地板 < "
                f"{AMBIGUITY_RATIO}*|v|={AMBIGUITY_RATIO * nv:.3e}）"
            )

        if note is not None:
            wa, wb = _witnesses(weights, simplex, body_a, body_b)
            n_vec = sub(wa, wb)
            n_len = norm(n_vec)
            witness_res = norm(sub(sub(wa, wb), v))
            d_dist = norm(sub(wa, wb))
            if n_len <= 0.0:
                # 见证重合却 v!=0：证书不自洽，绝不返回 separated。
                exit_kind = "contradiction"
                break
            sep_dir: Vec3 | None = scale(n_vec, 1.0 / n_len)
            return DistanceResult(
                status=SEPARATED,
                converged=True,
                distance=d_dist,
                witness_a=wa,
                witness_b=wb,
                separation_direction=sep_dir,
                residual=max(residual, 0.0),
                witness_residual=witness_res,
                iterations=iterations,
                max_iterations=budget,
                singular_skipped=singular_total,
                reason="certified",
                message=f"{note}：分离距离证书成立",
            )

        if stagnated:
            # 无法再推进又拿不到证书：区分两种精度失败，均与
            # 「证明相交」严格分开，且不冒充分离。
            if stag_floor >= AMBIGUITY_RATIO * nv:
                exit_kind = "ambiguous"
            else:
                exit_kind = "contradiction"
            break

        if iterations == budget:
            exit_kind = "budget"
            break

        # 按获胜子集降维裁剪，再加入新顶点。
        pruned = [simplex[i] for i in range(len(simplex))
                  if (sol.mask >> i) & 1]
        pruned.append(nxt)
        simplex = pruned
    else:
        exit_kind = "budget"  # while 条件耗尽（理论上被内部 break 覆盖）

    # 未取得证书：预算耗尽 / 机器精度不可判定 / 数值矛盾，
    # 三者均与「证明相交」严格区分，不发分离方向。
    wa, wb = _witnesses(weights, simplex, body_a, body_b)
    witness_res = norm(sub(sub(wa, wb), v))
    if exit_kind == "budget":
        reason = "budget_exhausted"
        msg = (
            f"迭代预算 {budget} 耗尽未收敛（当前残差 {residual:.3e}）；"
            "距离为当前最佳估计，不构成证书，与相交结论严格区分"
        )
    elif exit_kind == "ambiguous":
        reason = "machine_floor_ambiguity"
        msg = (
            f"support 停滞于 |v|={nv:.3e}，而病态方向噪声地板 "
            f"{stag_floor:.3e} 已与之同阶（>= {AMBIGUITY_RATIO}*|v|）："
            "双精度无法区分「接触」与「微小缝隙」，按未收敛处理，"
            "不冒发分离或相交证书（与相交证明严格区分）"
        )
    else:
        reason = "numerical_contradiction"
        msg = (
            f"support 停滞但残差 {residual:.3e} 明显超出噪声地板 "
            f"{stag_floor:.3e}：出现数值矛盾，按未收敛处理"
        )
    return DistanceResult(
        status=NOT_CONVERGED,
        converged=False,
        distance=norm(sub(wa, wb)),
        witness_a=wa,
        witness_b=wb,
        separation_direction=None,
        residual=max(residual, 0.0) if math.isfinite(residual) else math.inf,
        witness_residual=witness_res,
        iterations=iterations,
        max_iterations=budget,
        singular_skipped=singular_total,
        reason=reason,
        message=msg,
    )
