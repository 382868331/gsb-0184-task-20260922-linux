"""固定输入演示（确定性，纯离线真实计算，无 sleep、无预录结果）。

运行：python demo.py

内容：
1. 点-线段与轴对齐盒：真实 GJK 计算，打印距离、两物体见证点、
   单位分离方向，并用独立的解析/支撑面核验分离证书；
2. 退化降维：重复点、共线点集、共面点云，不做随机抖动，
   以枚举跳过仿射相关子集（singular_skipped）的方式降维，距离精确；
3. 实际触发的拒绝边界：
   - 非法输入（NaN）抛 InvalidBodyError，错误信息可定位到坐标；
   - 大尺度近接触构型：双精度无法区分「接触」与「微小缝隙」时，
     返回 status=not_converged / reason=machine_floor_ambiguity，
     不冒发证书（与相交证明、预算耗尽严格分开）。
"""

from __future__ import annotations

import math

from convex_distance import (
    ConvexBody,
    InvalidBodyError,
    NOT_CONVERGED,
    box,
    distance,
)


def _fmt(v: tuple[float, float, float]) -> str:
    return f"({v[0]: .10g}, {v[1]: .10g}, {v[2]: .10g})"


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show(name: str, A, B, *, note: str = "") -> None:
    r = distance(A, B)
    print(f"[{name}] {note}")
    print(f"  status               : {r.status}")
    print(f"  converged / reason   : {r.converged} / {r.reason}")
    print(f"  distance             : {r.distance:.12g}")
    print(f"  witness_a (A 上)     : {_fmt(r.witness_a)}")
    print(f"  witness_b (B 上)     : {_fmt(r.witness_b)}")
    if r.separation_direction is not None:
        print(f"  separation direction : {_fmt(r.separation_direction)} (单位)")
        # 独立核验（解析）：min n·a - max n·b 必须等于 distance
        pts_a = A.points if isinstance(A, ConvexBody) else ConvexBody(A).points
        pts_b = B.points if isinstance(B, ConvexBody) else ConvexBody(B).points
        n = r.separation_direction
        gap = (
            min(sum(n[k] * p[k] for k in range(3)) for p in pts_a)
            - max(sum(n[k] * p[k] for k in range(3)) for p in pts_b)
        )
        print(f"  证书核验 min n·a-max n·b = {gap:.12g} "
              f"(应等于 distance，差 {abs(gap - r.distance):.2e})")
        # 见证方向自洽
        w = tuple(r.witness_a[k] - r.witness_b[k] for k in range(3))
        print(f"  见证自洽 |wa-wb|     : {math.sqrt(sum(c*c for c in w)):.12g}")
    print(f"  support 残差         : {r.residual:.3e}")
    print(f"  见证自洽残差         : {r.witness_residual:.3e}")
    print(f"  迭代次数 / 上限      : {r.iterations} / {r.max_iterations}")
    print(f"  跳过的退化(仿射相关)子集: {r.singular_skipped}")
    return r


def main() -> None:
    section("1) 正常结果：真实 GJK 分离证书（点-线段 / 轴对齐盒）")

    # 点到斜线段的垂线在线段中部：解析距离 1，方向 (0,1,0)
    show("点-线段", [(0.0, 1.0, 0.0)],
         [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
         note="点 (0,1,0) 到 x 轴线段 [-1,1]，解析距离 1")

    # 轴对齐盒：x 方向相距 2，最近集为面；解析距离 2
    show("轴对齐盒", box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), name="A"),
         box((3.0, 0.0, 0.0), (4.0, 1.0, 1.0), name="B"),
         note="两盒 x 方向面间距 2（最近集为整张面，见证为面上任意点）")

    section("2) 退化降维：重复点 / 共线 / 共面，无随机抖动，距离精确")

    # 共线 5 点 + 1 个重复点（确定性去重），Minkowski 差退化但仍精确
    line = [(float(k), 0.0, 0.0) for k in range(-2, 3)] + [(2.0, 0.0, 0.0)]
    show("共线点集(含重复)", line, [(0.0, 2.0, 0.0)],
         note="x 轴 [-2,2] 共线点与点 (0,2,0)，解析距离 2；重复点确定性去重")

    # 两个共面正方形（z 为常数）：三维凸包体积为零，降维处理
    sq1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
           (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    sq2 = [(0.2, 0.2, 3.0), (0.8, 0.2, 3.0),
           (0.8, 0.8, 3.0), (0.2, 0.8, 3.0)]
    show("共面正方形", sq1, sq2,
         note="两个零体积共面正方形，z 向解析距离 3（仿射相关子集被枚举跳过）")

    section("3a) 实际触发的异常边界：NaN 输入被拒绝（错误可定位）")

    bad = [(0.0, float("nan"), 0.0)]
    try:
        distance(bad, [(1.0, 0.0, 0.0)])
    except InvalidBodyError as exc:
        print("  捕获 InvalidBodyError：")
        print(f"    {exc}")
        print("  -> 不静默纠正、不抖动、不留部分结果；定位到 A[0][1]")
    else:
        raise AssertionError("NaN 必须被拒绝")

    # 整数参数拒绝 bool（bool 虽是 int 子类型）
    try:
        distance([(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0)],
                 max_iterations=True)  # type: ignore[arg-type]
    except TypeError as exc:
        print(f"  整数参数拒绝 bool：TypeError -> {exc}")

    section("3b) 实际触发的拒绝边界：双精度机器地板，拒绝冒发证书")

    # 两个 R=200000 的轴对齐盒，解析上沿 x=s 面真实接触（y/z 区间重叠）。
    # 在该尺度下接触与 ~6e-7 的微小缝隙在双精度内不可区分：
    # 模块必须返回 not_converged / machine_floor_ambiguity，
    # 既不冒充分离证书，也不与「证明相交」混为一谈。
    s = 100000.0
    oy, oz = -0.08086095642853475, 0.040685885532142474  # 固定（种子 7 第 9 例）
    A = box((-s, -s, -s), (s, s, s), name="bigA")
    B = box((s, oy, oz), (s + 1.0, oy + 1.0, oz + 1.0), name="bigB")
    # 解析核验：轴对齐盒区间在三个轴上均重叠 => 真实接触/相交
    overlap = all(
        max(-s, lo) <= min(s, hi)
        for lo, hi in ((s, s + 1.0), (oy, oy + 1.0), (oz, oz + 1.0))
    )
    print(f"  解析判定：两盒三轴区间均重叠 = {overlap}（真实距离为 0）")
    r = distance(A, B)
    print(f"  GJK status = {r.status}，converged = {r.converged}")
    print(f"  reason     = {r.reason}")
    print(f"  当前最佳距离估计 = {r.distance:.6e}（无证书意义）")
    print(f"  separation_direction = {r.separation_direction}（拒绝发证时为 None）")
    print(f"  {r.message}")
    assert r.status == NOT_CONVERGED and not r.converged
    assert r.separation_direction is None
    print("  -> 搜索预算耗尽(budget_exhausted)、机器地板不可判定"
          "(machine_floor_ambiguity)、证明相交(intersecting) 三者严格分开")

    print()
    print("演示完成：以上数值均为现场真实计算，输入固定且确定性可复现。")


if __name__ == "__main__":
    main()
