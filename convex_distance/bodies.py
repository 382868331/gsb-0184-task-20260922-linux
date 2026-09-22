"""凸体：一组三维点的凸包，只通过 support 函数访问。

设计要点：
- 不预先构造任何面/半空间；GJK 需要的一切几何信息都由 ``support``
  （沿给定方向的极值点）提供。
- 退化输入（点、共线、共面）是合法的：support 仍良定义，真正的降维
  发生在单纯形子集枚举阶段（见 simplex.py），不做任何随机抖动。
- 顶点去重在输入校验阶段确定性完成（errors.validate_points）。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .errors import InvalidBodyError, reject_number, validate_points
from .vectors import Vec3, dot


class ConvexBody:
    """由有限点集隐式表示的凸体 conv(points)。

    构造时完整校验并冻结点集；构造失败不会产生半成品对象
    （校验在任何赋值之前完成）。
    """

    __slots__ = ("_points", "_name")

    def __init__(self, points: Iterable[Sequence[float]], *, name: str = "body"):
        frozen = validate_points(points, name)
        self._points = frozen
        self._name = name

    @property
    def points(self) -> tuple[Vec3, ...]:
        return self._points

    @property
    def name(self) -> str:
        return self._name

    def support(self, direction: Vec3) -> tuple[Vec3, int]:
        """h(direction) = argmax_{p in points} dot(p, direction)。

        返回 (极点, 下标)。方向为零时确定性取 index 0（不抛错——
        算法不应以零方向调用 support，这里只做稳健兜底）。
        并列最大值确定性地取最小下标，保证可复现、不依赖 dict 顺序。
        """
        points = self._points
        best_i = 0
        best_v = dot(points[0], direction)
        for i in range(1, len(points)):
            v = dot(points[i], direction)
            if v > best_v:
                best_v = v
                best_i = i
        return points[best_i], best_i

    def __len__(self) -> int:
        return len(self._points)

    def __repr__(self) -> str:
        return f"ConvexBody(name={self._name!r}, n={len(self._points)})"


def box(
    lo: Sequence[float], hi: Sequence[float], *, name: str = "box"
) -> ConvexBody:
    """便利构造：轴对齐长方体 [lo, hi] 的 8 个角点（确定性顺序）。

    坐标同样拒绝 bool/NaN/Infinity；lo==hi 的维度退化为点也合法
    （去重后自然降维）。lo[k] > hi[k] 时角点集合不变，凸包仍为
    以两值为界的轴对齐盒。
    """
    if not isinstance(lo, (list, tuple)) or not isinstance(hi, (list, tuple)):
        raise TypeError(f"{name}: lo/hi 必须是长度为 3 的 list/tuple")
    if len(lo) != 3 or len(hi) != 3:
        raise InvalidBodyError(f"{name}: lo/hi 必须长度为 3")
    lo_f = tuple(reject_number(lo[k], f"{name}.lo[{k}]") for k in range(3))
    hi_f = tuple(reject_number(hi[k], f"{name}.hi[{k}]") for k in range(3))
    points = [
        (lo_f[0] if (k & 1) else hi_f[0],
         lo_f[1] if (k & 2) else hi_f[1],
         lo_f[2] if (k & 4) else hi_f[2])
        for k in range(8)
    ]
    return ConvexBody(points, name=name)
