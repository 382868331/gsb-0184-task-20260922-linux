"""Deterministic offline demo for the convex_distance module.

Runs real computations (no sleeps, no prerecorded output):
  1. a separated pair of convex hulls -> prints the separation certificate
     and verifies it against every input vertex;
  2. degenerate lower-dimensional inputs (segment vs triangle) handled by
     dimension reduction;
  3. contract violations (NaN coordinate, too many points) actually
     triggering InputError rejections;
  4. translational sweep: first contact and contact interval of a cube
     sliding through another, checked against the hand-computed interval;
  5. sweep edge cases: tangential graze (touching), complete miss,
     relative rest;
  6. sweep contract violations actually triggering InputError.
"""

import random

from convex_distance import InputError, distance, sweep


def fmt(p):
    return "(" + ", ".join(f"{c:+.6f}" for c in p) + ")"


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def box(lo, hi):
    return [(x, y, z)
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])]


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
    print()

    # --- Case 4: translational sweep, cube sliding through cube -----------
    # A = [-1,1]^3 static; B = [4,6]x[-1,1]^2 moving at (-1,0,0).
    # Hand-computed: faces meet at t = 3, B clears A again at t = 7.
    a = box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    b = box((4.0, -1.0, -1.0), (6.0, 1.0, 1.0))
    sw = sweep(a, b, (-1.0, 0.0, 0.0), (0.0, 10.0))
    print("=== Case 4: sweep, cube sliding through cube ===")
    print(f"status          : {sw.status}")
    print(f"contact interval: [{sw.t_enter:.9f}, {sw.t_exit:.9f}]"
          f"  (hand-computed: [3.0, 7.0])")
    print(f"t_closest       : {sw.t_closest:.9f}   "
          f"min_distance: {sw.min_distance:.3e}")
    print(f"witness_a@enter : {fmt(sw.witness_a)}")
    print(f"witness_b@enter : {fmt(sw.witness_b)}")
    print(f"GJK evaluations : {sw.evaluations}   "
          f"zero_tolerance: {sw.zero_tolerance:.3e}")
    ok = abs(sw.t_enter - 3.0) < 1e-6 and abs(sw.t_exit - 7.0) < 1e-6
    print(f"matches hand-computed interval -> {'OK' if ok else 'MISMATCH'}")
    print()

    # --- Case 5: sweep edge cases ------------------------------------------
    print("=== Case 5: sweep edge cases ===")
    # tangential graze: point (t, 2-t, 0) touches the cube edge (1,1,0)
    # exactly at t = 1 -> zero-length contact interval
    g = sweep(a, [(0.0, 2.0, 0.0)], (1.0, -1.0, 0.0), (0.0, 4.0))
    print(f"tangential graze : {g.status} at t = {g.t_enter:.9f}"
          f"  (hand-computed: t = 1.0)")
    # complete miss: point passes the origin at distance 3
    m = sweep([(0.0, 0.0, 0.0)], [(5.0, 3.0, 0.0)], (-1.0, 0.0, 0.0),
              (0.0, 10.0))
    print(f"complete miss    : {m.status}, closest approach "
          f"{m.min_distance:.9f} at t = {m.t_closest:.6f}"
          f"  (hand-computed: 3.0 at t = 5.0)")
    # relative rest, overlapping: contact fills the whole window
    s = sweep(a, box((0.5, -1.0, -1.0), (2.5, 1.0, 1.0)),
              (0.0, 0.0, 0.0), (2.0, 9.0))
    print(f"relative rest    : {s.status}, stationary={s.stationary},"
          f" interval [{s.t_enter:.1f}, {s.t_exit:.1f}] = whole window")
    print()

    # --- Case 6: sweep contract violations ---------------------------------
    print("=== Case 6: rejection of invalid sweep input ===")
    try:
        sweep(a, b, (1.0, 0.0), (0.0, 10.0))
    except InputError as exc:
        print(f"2-component velocity rejected -> InputError: {exc}")
    try:
        sweep(a, b, (1.0, 0.0, 0.0), (10.0, 0.0))
    except InputError as exc:
        print(f"reversed window rejected      -> InputError: {exc}")
    try:
        sweep(a, b, (float("nan"), 0.0, 0.0), (0.0, 10.0))
    except InputError as exc:
        print(f"NaN velocity rejected         -> InputError: {exc}")


if __name__ == "__main__":
    main()
