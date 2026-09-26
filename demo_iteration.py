"""Deterministic offline demo for penetration(): GJK + EPA.

Runs three real computations (no sleeps, no prerecorded output):
  1. an oblique penetration (rotated cube through the unit cube) -- prints
     depth bounds, direction and contact points, then translates B by the
     reported upper bound and re-checks separation independently;
  2. a face-to-face touching pair -- explicit zero-depth "touching"
     status, no invented normal;
  3. an exhausted iteration budget -- verifiable bounds are kept and the
     status honestly reports "not_converged".
"""

import math

from convex_distance import (
    STATUS_NOT_CONVERGED,
    STATUS_PENETRATING,
    STATUS_TOUCHING,
    penetration,
)


def fmt(p):
    return "(" + ", ".join(f"{c:+.6f}" for c in p) + ")"


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def norm(u):
    return math.sqrt(dot(u, u))


def rot(axis, angle):
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return (
        (x * x * C + c, x * y * C - z * s, x * z * C + y * s),
        (y * x * C + z * s, y * y * C + c, y * z * C - x * s),
        (z * x * C - y * s, z * y * C + x * s, z * z * C + c),
    )


def cube(lo, hi):
    return [(float(x), float(y), float(z))
            for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])]


def main():
    unit = cube((0, 0, 0), (1, 1, 1))

    # --- Case 1: oblique penetration, rotated cube vs unit cube ---------
    R = rot((1.0, 2.0, 3.0), math.radians(40.0))
    oblique = [tuple(sum(R[i][j] * p[j] for j in range(3)) + off
                     for i, off in enumerate((0.55, 0.35, 0.45)))
               for p in cube((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))]

    res = penetration(unit, oblique)
    print("=== Case 1: oblique penetration (rotated cube vs unit cube) ===")
    print(f"status       : {res.status}")
    assert res.status == STATUS_PENETRATING
    print(f"depth        : {res.depth:.12f}  "
          f"[{res.depth_lower:.12f}, {res.depth_upper:.12f}]")
    print(f"direction    : {fmt(res.direction)}")
    print(f"witness_a    : {fmt(res.witness_a)}")
    print(f"witness_b    : {fmt(res.witness_b)}")
    print(f"iterations   : {res.iterations}")
    gap = tuple(res.witness_a[i] - res.witness_b[i] for i in range(3))
    print(f"|wa - wb|    : {norm(gap):.12f} (== depth_lower, "
          f"aligned with direction: {dot(gap, res.direction):.12f})")

    # translate B by depth_upper * direction and verify separation by an
    # independent support scan over SAT-style axes
    t = res.depth_upper + 1e-6
    moved = [tuple(p[i] + t * res.direction[i] for i in range(3))
             for p in oblique]
    axes = []
    for verts in (unit, moved):
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                for k in range(j + 1, len(verts)):
                    u = tuple(verts[j][c] - verts[i][c] for c in range(3))
                    v = tuple(verts[k][c] - verts[i][c] for c in range(3))
                    n = (u[1] * v[2] - u[2] * v[1],
                         u[2] * v[0] - u[0] * v[2],
                         u[0] * v[1] - u[1] * v[0])
                    if norm(n) > 1e-12:
                        axes.append(n)
    for i in range(len(unit)):
        for j in range(i + 1, len(unit)):
            ea = tuple(unit[j][c] - unit[i][c] for c in range(3))
            for k in range(len(moved)):
                for l in range(k + 1, len(moved)):
                    eb = tuple(moved[l][c] - moved[k][c]
                               for c in range(3))
                    n = (ea[1] * eb[2] - ea[2] * eb[1],
                         ea[2] * eb[0] - ea[0] * eb[2],
                         ea[0] * eb[1] - ea[1] * eb[0])
                    if norm(n) > 1e-12:
                        axes.append(n)
    separated = any(
        max(dot(p, n) for p in unit) < min(dot(p, n) for p in moved) or
        max(dot(p, n) for p in moved) < min(dot(p, n) for p in unit)
        for n in axes)
    print(f"separated after translating B by depth_upper*n: {separated}")
    assert separated

    # --- Case 2: face-to-face contact -> explicit zero-depth status -----
    touching = cube((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    res = penetration(unit, touching)
    print()
    print("=== Case 2: face-to-face contact (no positive-volume overlap) ===")
    print(f"status       : {res.status}")
    assert res.status == STATUS_TOUCHING
    print(f"depth        : {res.depth}  direction: {res.direction}")
    print(f"witness_a    : {fmt(res.witness_a)}")
    print(f"witness_b    : {fmt(res.witness_b)}")

    # --- Case 3: exhausted iteration budget -> honest not_converged -----
    res = penetration(unit, oblique, max_iter=4)
    print()
    print("=== Case 3: iteration budget exhausted (max_iter=4) ===")
    print(f"status       : {res.status}")
    assert res.status == STATUS_NOT_CONVERGED
    print(f"bounds kept  : [{res.depth_lower:.12f}, "
          f"{res.depth_upper:.12f}]")
    print(f"direction    : {fmt(res.direction)} (best known, not certified)")
    print(f"residual     : {res.residual:.6f}")


if __name__ == "__main__":
    main()
