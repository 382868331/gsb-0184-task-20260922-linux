"""凸多面体的最近点与分离证书（GJK，纯标准库）。

公开接口：
- ``distance(A, B, ...)``：主入口，返回 ``DistanceResult``；
- ``ConvexBody``：由 <=64 个三维点表示的凸体，只通过 support 访问；
- ``DistanceResult``：距离、见证点、分离方向、残差、迭代状态；
- 状态常量 ``SEPARATED`` / ``INTERSECTING`` / ``NOT_CONVERGED``；
- 异常 ``ConvexDistanceError`` / ``InvalidBodyError`` /
  ``InvalidParameterError``；
- 边界常量 ``MAX_POINTS`` (=64) / ``MAX_ITERATIONS`` (=128)。
"""

from __future__ import annotations

from .bodies import ConvexBody, box
from .errors import (
    MAX_ITERATIONS,
    MAX_POINTS,
    ConvexDistanceError,
    InvalidBodyError,
    InvalidParameterError,
)
from .gjk import (
    INTERSECTING,
    NOT_CONVERGED,
    SEPARATED,
    DistanceResult,
    distance,
)

__all__ = [
    "distance",
    "DistanceResult",
    "ConvexBody",
    "box",
    "SEPARATED",
    "INTERSECTING",
    "NOT_CONVERGED",
    "ConvexDistanceError",
    "InvalidBodyError",
    "InvalidParameterError",
    "MAX_POINTS",
    "MAX_ITERATIONS",
]

__version__ = "1.0.0"
