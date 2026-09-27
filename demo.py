"""Deterministic offline demo for the convex_distance module.

Runs three real computations (no sleeps, no prerecorded output):
  1. a separated pair of convex hulls -> prints the separation certificate
     and verifies it against every input vertex;
  2. degenerate lower-dimensional inputs (segment vs triangle) handled by
     dimension reduction;
  3. contract violations (NaN coordinate, too many points) actually
     triggering InputError rejections.
"""

import random

from convex_distance import InputError, distance


def fmt(p):
    return "(" + ", ".join(f"{c:+.6f}" for c in p) + ")"


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


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

    # --- Case 1b: warm-started continuous frames --------------------------
    print("=== Case 1b: 4 consecutive frames, cold vs warm-started ===")
    prev = None
    for t in range(4):
        off = 2.5 + 0.05 * t
        frame = [(rng.uniform(off, off + 1.5), rng.uniform(-1.0, 1.0),
                  rng.uniform(-1.0, 1.0)) for _ in range(16)]
        cold = distance(cube, frame)
        warm = distance(cube, frame, warm_start=prev) if prev else None
        if warm is None:
            print(f"frame {t}: cold iterations = {cold.iterations}")
        else:
            assert warm.status == cold.status
            assert abs(warm.distance - cold.distance) < 1e-12
            print(f"frame {t}: cold iterations = {cold.iterations}, "
                  f"warm-started iterations = {warm.iterations} "
                  f"(same distance {warm.distance:.12f})")
        prev = cold
        last = warm if warm is not None else cold
    # the last warm-started result still carries a valid certificate
    n = last.separating_axis
    print(f"last frame certificate: min_A a.n - max_B b.n = "
          f"{min(dot(a, n) for a in cube) - max(dot(b, n) for b in frame):.12f}"
          f" > 0 (recomputed, not cached)")
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

    # --- Case 3: contract violations are rejected, not silently fixed -----
    print("=== Case 3: rejection of invalid input ===")
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


if __name__ == "__main__":
    main()
