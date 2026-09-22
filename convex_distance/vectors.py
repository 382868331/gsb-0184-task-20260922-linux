"""三维向量小工具，全部使用 float 运算。

主算法基于 IEEE-754 双精度浮点，容差只用于收敛/终止判断与证书分类，
绝不用于“修正”顶点坐标或抖动退化构型。
"""

from __future__ import annotations

import math

Vec3 = tuple[float, float, float]

# 相对长度比例容差：见 README「误差口径」。
EPS_REL = 1e-12
# 绝对容差（尺度与 1 同量级时生效）。
EPS_ABS = 1e-12


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm_sq(a: Vec3) -> float:
    return dot(a, a)


def norm(a: Vec3) -> float:
    return math.sqrt(norm_sq(a))
