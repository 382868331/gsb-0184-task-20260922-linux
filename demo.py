"""Deterministic offline demo for the convex_distance module.

Runs real computations (no sleeps, no prerecorded output):
  1. a separated pair of convex hulls -> prints the separation certificate
     and verifies it against every input vertex;
  2. degenerate lower-dimensional inputs (segment vs triangle) handled by
     dimension reduction;
  3. ellipsoid Minkowski shells: a segment core and a point core carrying
     different rotated ellipsoid shells -> prints the core/shell witness
     split and the certified [lower, upper] distance interval, verified
     against an independent support-function evaluation;
  4. contract violations (NaN coordinate, too many points, singular
     shell matrix) actually triggering InputError rejections.
"""

import math
import random

from convex_distance import InputError, distance, distance_shell


def fmt(p):
    return "(" + ", ".join(f"{c:+.6f}" for c in p) + ")"


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def norm_matT(L, v):
    """||L^T v|| -- the shell's support radius in direction v."""
    tx = L[0][0] * v[0] + L[1][0] * v[1] + L[2][0] * v[2]
    ty = L[0][1] * v[0] + L[1][1] * v[1] + L[2][1] * v[2]
    tz = L[0][2] * v[0] + L[1][2] * v[1] + L[2][2] * v[2]
    return math.sqrt(tx * tx + ty * ty + tz * tz)


def main():
    rng = random.Random(20260922)  # deterministic sample data

    # --- Case 1: unit cube vs a translated random point cloud -------------
    cube = [(float(x), float(y), float(z))
            for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    cloud = [(rng.uniform(2.5, 4.0), rng.uniform(-1.0, 1.0),
              rng.uniform(-1.0, 1.0)) for _ in range(16)]

    res = distance(cube, cloud)
    print("=== Case 1: unit cube vs translated 16-point cloud ===")
    print(f"status          : {res.status}")
    print(f"distance        : {res.distance:.12f}")
    print(f"witness_a       : {fmt(res.witness_a)}")
    print(f"witness_b       : {fmt(res.witness_b)}")
    print(f"separating_axis : {fmt(res.separating_axis)}")
    print(f"iterations      : {res.iterations}   residual: {res.residual:.3e}")

    n = res.separating_axis
    wa_n = dot(res.witness_a, n)
    wb_n = dot(res.witness_b, n)
    min_a = min(dot(a, n) for a in cube)
    max_b = max(dot(b, n) for b in cloud)
    print("certificate check over all vertices:")
    print(f"  min_A a.n = {min_a:.12f}  >=  wa.n = {wa_n:.12f}"
          f"  -> {'OK' if min_a >= wa_n - 1e-9 else 'VIOLATED'}")
    print(f"  max_B b.n = {max_b:.12f}  <=  wb.n = {wb_n:.12f}"
          f"  -> {'OK' if max_b <= wb_n + 1e-9 else 'VIOLATED'}")
    print(f"  separation gap min_A a.n - max_B b.n = {min_a - max_b:.12f} > 0")
    print()

    # --- Case 2: degenerate inputs (dimension reduction, no jitter) -------
    segment = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    triangle = [(3.0, -1.0, 0.0), (3.0, 1.0, -1.0), (3.0, 1.0, 1.0)]
    res2 = distance(segment, triangle)
    print("=== Case 2: segment vs triangle (degenerate, lower-dimensional) ===")
    print(f"status          : {res2.status}")
    print(f"distance        : {res2.distance:.12f}  (exact value: 1.0)")
    print(f"witness_a       : {fmt(res2.witness_a)}")
    print(f"witness_b       : {fmt(res2.witness_b)}")
    print(f"iterations      : {res2.iterations}   residual: {res2.residual:.3e}")
    print()

    # --- Case 3: ellipsoid Minkowski shells ------------------------------
    # segment core with a rotated thin ellipsoid shell vs a point core
    # with a spherical shell
    seg = [(0.0, 0.0, 0.0), (2.0, 0.5, 0.0)]
    pt = [(1.0, 3.0, 1.0)]
    c, s = math.cos(0.7), math.sin(0.7)
    # L_a = rotation about z by 0.7 rad applied to diag(1.2, 0.3, 0.4)
    L_a = ((1.2 * c, -0.3 * s, 0.0),
           (1.2 * s, 0.3 * c, 0.0),
           (0.0, 0.0, 0.4))
    L_b = ((0.6, 0.0, 0.0), (0.0, 0.6, 0.0), (0.0, 0.0, 0.6))

    res3 = distance_shell(seg, pt, shell_a=L_a, shell_b=L_b)
    print("=== Case 3: segment core + rotated ellipsoid shell vs "
          "point core + sphere shell ===")
    print(f"status          : {res3.status}")
    print(f"distance        : {res3.distance:.12f}")
    print(f"certified bounds: [{res3.lower_bound:.12f}, "
          f"{res3.upper_bound:.12f}]")
    print(f"core_witness_a  : {fmt(res3.core_witness_a)}"
          f"   shell_offset_a: {fmt(res3.shell_offset_a)}")
    print(f"core_witness_b  : {fmt(res3.core_witness_b)}"
          f"   shell_offset_b: {fmt(res3.shell_offset_b)}")
    print(f"witness_a       : {fmt(res3.witness_a)}  (on dilated shell A)")
    print(f"witness_b       : {fmt(res3.witness_b)}  (on dilated shell B)")
    print(f"separating_axis : {fmt(res3.separating_axis)}")
    print(f"iterations      : {res3.iterations}   residual: "
          f"{res3.residual:.3e}")

    # independent certificate check: the gap along n, recomputed from the
    # support functions max a.n + ||L^T n||, must match the distance
    n3 = res3.separating_axis
    min_a = min(dot(a, n3) for a in seg) - norm_matT(L_a, n3)
    max_b = max(dot(b, n3) for b in pt) + norm_matT(L_b, n3)
    print("certificate check (independent support evaluation):")
    print(f"  min_A' a.n = {min_a:.12f}   max_B' b.n = {max_b:.12f}")
    print(f"  certified gap along n = {min_a - max_b:.12f}"
          f"  -> {'OK' if abs((min_a - max_b) - res3.distance) < 1e-6 else 'VIOLATED'}")
    print()

    # --- Case 4: contract violations are rejected, not silently fixed ----
    print("=== Case 4: rejection of invalid input ===")
    try:
        distance([(0.0, 0.0, float("nan"))], cloud)
    except InputError as exc:
        print(f"NaN coordinate rejected  -> InputError: {exc}")
    try:
        distance([(float(i), 0.0, 0.0) for i in range(65)], cloud)
    except InputError as exc:
        print(f"65 points rejected       -> InputError: {exc}")
    try:
        distance(cube, cloud, max_iter=True)
    except InputError as exc:
        print(f"bool max_iter rejected   -> InputError: {exc}")
    try:
        distance_shell(seg, pt, shell_a=((1.0, 0.0, 0.0),
                                         (0.0, 1.0, 0.0),
                                         (0.0, 0.0, 0.0)))
    except InputError as exc:
        print(f"singular shell rejected  -> InputError: {exc}")


if __name__ == "__main__":
    main()
