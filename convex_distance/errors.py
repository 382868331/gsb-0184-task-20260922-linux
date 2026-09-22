"""错误类型与输入校验。

错误定位约定：所有非法输入错误信息都带有参数名与下标，例如
``A[3][1]``、``B``、``max_iterations``，便于调用方定位。
模块是无状态的：任何对象构造在完整校验通过后才提交，失败不留部分变更。
"""

from __future__ import annotations

import math
from typing import Any

# 题面接口边界：每个物体至多 64 个顶点。
MAX_POINTS = 64
# 题面接口边界：GJK 主循环至多 128 次迭代。
MAX_ITERATIONS = 128


class ConvexDistanceError(Exception):
    """本模块所有自定义异常的公共基类。"""


class InvalidBodyError(ConvexDistanceError, ValueError):
    """物体点集非法（空、超长、非有限数、维度错误等）。"""


class InvalidParameterError(ConvexDistanceError, ValueError):
    """标量参数非法（取值越界等）。"""


def reject_number(value: Any, where: str) -> float:
    """校验单个数值坐标：拒绝 bool、非数值、NaN/Infinity；返回 float。"""
    if isinstance(value, bool):
        raise TypeError(
            f"{where}: bool 不是合法数值坐标（整数参数也不接受 bool）"
        )
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{where}: 期望 int/float 有限数值，实际为 {type(value).__name__}"
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise InvalidBodyError(f"{where}: 坐标必须有限，拒绝 NaN/Infinity")
    return fv


def _reject_point(point: Any, name: str, i: int) -> tuple[float, float, float]:
    if not isinstance(point, (list, tuple)):
        raise TypeError(
            f"{name}[{i}]: 点必须是长度为 3 的 list/tuple，"
            f"实际为 {type(point).__name__}"
        )
    if len(point) != 3:
        raise InvalidBodyError(
            f"{name}[{i}]: 点必须恰好包含 3 个坐标，实际长度 {len(point)}"
        )
    return tuple(
        reject_number(point[j], f"{name}[{i}][{j}]") for j in range(3)
    )  # type: ignore[return-value]


def validate_points(points: Any, name: str) -> tuple[tuple[float, float, float], ...]:
    """校验并冻结一个物体的点集。

    - 必须是 list/tuple，非空，长度 <= 64；
    - 每个点是长度 3 的 list/tuple；
    - 坐标为有限 int/float（bool 按 TypeError 拒绝）；
    - 完全相同的点确定性去重（保留首次出现），其余顺序不变。
    """
    if not isinstance(points, (list, tuple)):
        raise TypeError(
            f"{name}: 点集必须是 list/tuple，实际为 {type(points).__name__}"
        )
    n = len(points)
    if n == 0:
        raise InvalidBodyError(f"{name}: 点集不能为空，至少需要 1 个点")
    if n > MAX_POINTS:
        raise InvalidBodyError(
            f"{name}: 点集长度 {n} 超过接口上限 {MAX_POINTS}"
        )
    frozen: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for i in range(n):
        p = _reject_point(points[i], name, i)
        if p not in seen:
            seen.add(p)
            frozen.append(p)
    return tuple(frozen)


def validate_int_param(value: Any, name: str, *, lo: int, hi: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name}: 整数参数拒绝 bool，实际为 {value!r}")
    if not isinstance(value, int):
        raise TypeError(
            f"{name}: 期望 int，实际为 {type(value).__name__}"
        )
    if not lo <= value <= hi:
        raise InvalidParameterError(
            f"{name}: 取值 {value} 超出允许范围 [{lo}, {hi}]"
        )
    return value


def validate_float_param(value: Any, name: str, *, positive: bool) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name}: 数值参数拒绝 bool，实际为 {value!r}")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name}: 期望 int/float，实际为 {type(value).__name__}")
    fv = float(value)
    if not math.isfinite(fv):
        raise InvalidParameterError(f"{name}: 必须有限，拒绝 NaN/Infinity")
    if positive and fv <= 0.0:
        raise InvalidParameterError(f"{name}: 必须为正数，实际为 {fv!r}")
    if not positive and fv < 0.0:
        raise InvalidParameterError(f"{name}: 不能为负，实际为 {fv!r}")
    return fv
