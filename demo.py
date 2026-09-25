"""Deterministic offline demo for the convex_distance module.

Runs real computations only (no sleeps, no prerecorded output):

  1. legacy ``distance`` interface: cube vs point cloud, separation
     certificate checked against every input vertex;
  2. spherical-shell special case checked against the analytic formula
     D - rA - rB;
  3. anisotropic ellipsoidal shell: directional thickness differs, the
     gap is NOT computed with a max-radius sphere, with core/offset
     witnesses split;
  4. rotated shell + degenerate segment core (capsule geometry);
  5. independent direction sampling: support bound and witness
     feasibility only -- sampling is never used as a global-optimality
     proof;
  6. un-converged run keeps its interval; singular matrix and zero
     direction are rejected, not silently repaired.
"""

import math
import random

from convex_distance import (
    InputError,
    distance,
    distance_shelled,
    shell_support_offset,
)


def fmt(p):
    return "(" + ", ".join(f"{c:+.6f}" for c in p) + ")"


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def mat_vec(M, x):
    return tuple(sum(M[i][j] * x[j] for j in range(3)) for i in range(3))


def transpose(M):
    return tuple(tuple(M[j][i] for j in range(3)) for i in range(3))


def solve3(M, b):
    m = [list(M[i]) + [b[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= f * m[col][c]
    x = [0.0, 0.0, 0.0]
    for r in range(2, -1, -1):
        x[r] = (m[r][3] - sum(m[r][c] * x[c]
                               for c in range(r + 1, 3))) / m[r][r]
    return tuple(x)


def fibonacci_dirs(k):
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for i in range(k):
        z = 1.0 - 2.0 * (i + 0.5) / k
        r = math.sqrt(max(0.0, 1.0 - z * z))
        a = golden * i
        out.append((math.cos(a) * r, math.sin(a) * r, z))
    return out


def main():
    rng = random.Random(20260925)  # deterministic sample data

    # --- Case 1: legacy interface, cube vs translated point cloud --------
    cube = [(float(x), float(y), float(z))
            for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    cloud = [(rng.uniform(2.5, 4.0), rng.uniform(-1.0, 1.0),
              rng.uniform(-1.0, 1.0)) for _ in range(16)]
    res = distance(cube, cloud)
    print("=== Case 1: legacy distance(), unit cube vs 16-point cloud ===")
    print(f"status {res.status}  distance {res.distance:.12f}"
          f"  iters {res.iterations}  residual {res.residual:.2e}")
    n = res.separating_axis
    min_a = min(dot(a, n) for a in cube)
    max_b = max(dot(b, n) for b in cloud)
    ok = min_a >= dot(res.witness_a, n) - 1e-9 and \
        max_b <= dot(res.witness_b, n) + 1e-9
    print(f"certificate over all {len(cube) + len(cloud)} vertices: "
          f"{'OK' if ok else 'VIOLATED'} "
          f"(min_A a.n - max_B b.n = {min_a - max_b:.6f} > 0)")
    print()

    # --- Case 2: spherical special case vs analytic D - rA - rB ----------
    D, rA, rB = 5.0, 0.5, 1.0
    eye = lambda r: ((r, 0.0, 0.0), (0.0, r, 0.0), (0.0, 0.0, r))
    res = distance_shelled([(0.0, 0.0, 0.0)], [(D, 0.0, 0.0)],
                           LA=eye(rA), LB=eye(rB))
    print("=== Case 2: point cores + spherical shells (analytic check) ===")
    print(f"status {res.status}  distance {res.distance:.12f}"
          f"  analytic D-rA-rB = {D - rA - rB:.12f}")
    print(f"witness_a {fmt(res.witness_a)}  witness_b {fmt(res.witness_b)}")
    print()

    # --- Case 3: anisotropic shell, thickness differs by direction -------
    # x semi-axis 0.4, y semi-axis 3.0; centres 5 apart on x.  The correct
    # gap is 5 - 0.4 = 4.6; a max-radius sphere would wrongly give 2.0.
    L = ((0.4, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 2.0))
    res = distance_shelled([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)], LA=L)
    print("=== Case 3: anisotropic ellipsoid shell diag(0.4, 3, 2) ===")
    print(f"status {res.status}  distance {res.distance:.12f}"
          f"  (max-radius-sphere would report 2.0 -- wrong)")
    print(f"core_a   {fmt(res.core_a)}     offset_a {fmt(res.offset_a)}")
    print(f"witness_a {fmt(res.witness_a)} = core + offset")
    for name, d in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
        off = shell_support_offset(L, d)
        print(f"  thickness in +{name}: ||L^T e_{name}|| = "
              f"{math.sqrt(dot(mat_vec(transpose(L), d), mat_vec(transpose(L), d))):.2f}"
            f"   support offset {fmt(off)}")
    print(f"bracket [{res.lower_bound:.12f}, {res.upper_bound:.12f}]"
          f"  residual {res.residual:.2e}")
    print()

    # --- Case 4: rotated shell + degenerate segment core (capsule) -------
    r = 0.5
    seg = [(-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    res = distance_shelled(seg, [(0.0, 1.75, 0.0)], LA=eye(r))
    print("=== Case 4: degenerate segment core + spherical shell (capsule) ===")
    print(f"status {res.status}  distance {res.distance:.12f}"
          f"  (analytic 1.75 - 0.5 = 1.25)")
    print(f"core_a {fmt(res.core_a)} (on the segment, a barycentric combo)")
    print(f"offset_a {fmt(res.offset_a)} (radial shell offset)")
    print(f"witness_a {fmt(res.witness_a)} (on the actual inflated body)")
    print()

    # --- Case 5: independent sampling: support bound + feasibility -------
    rng2 = random.Random(424242)
    A = [tuple(rng2.uniform(-2, 2) for _ in range(3)) for _ in range(8)]
    B = [(rng2.uniform(-2, 2) + 6.0, rng2.uniform(-1, 1),
          rng2.uniform(-1, 1)) for _ in range(6)]
    LA = ((0.9, 0.2, 0.0), (0.0, 0.35, 0.1), (0.1, 0.0, 0.6))
    LB = ((0.3, 0.0, 0.1), (0.1, 0.4, 0.0), (0.0, 0.2, 0.5))
    res = distance_shelled(A, B, LA=LA, LB=LB)
    print("=== Case 5: independent direction sampling (NOT a global proof) ===")
    print(f"status {res.status}  distance {res.distance:.9f}"
          f"  bracket [{res.lower_bound:.9f}, {res.upper_bound:.9f}]")
    violations = 0
    for nn in fibonacci_dirs(400):
        off = shell_support_offset(LA, nn)
        # offset is on the boundary: ||L^-1 off|| == 1, and off.n == ||L^T n||
        if abs(math.sqrt(dot(solve3(LA, off), solve3(LA, off))) - 1.0) > 1e-9:
            violations += 1
        if abs(dot(off, nn) - math.sqrt(
                dot(mat_vec(transpose(LA), nn),
                    mat_vec(transpose(LA), nn)))) > 1e-9:
            violations += 1
        # no sampled ellipsoid point exceeds the support offset
        for u in fibonacci_dirs(24):
            if dot(mat_vec(LA, u), nn) > dot(off, nn) + 1e-9:
                violations += 1
    # returned witnesses feasible: cores inside hulls (sampled directions),
    # offsets inside ellipsoids
    za = math.sqrt(dot(solve3(LA, res.offset_a), solve3(LA, res.offset_a)))
    zb = math.sqrt(dot(solve3(LB, res.offset_b), solve3(LB, res.offset_b)))
    core_ok = all(
        dot(res.core_a, nn) <= max(dot(a, nn) for a in A) + 1e-8 and
        dot(res.core_b, nn) >= min(dot(b, nn) for b in B) - 1e-8
        for nn in fibonacci_dirs(400))
    print(f"support-bound violations over 400 directions x 24 samples: "
          f"{violations}")
    print(f"offset feasibility ||L^-1 offset||: A {za:.9f} <= 1, "
          f"B {zb:.9f} <= 1")
    print(f"core witnesses inside core hulls (sampled): "
          f"{'OK' if core_ok else 'VIOLATED'}")
    print(f"lower <= distance <= upper: "
          f"{res.lower_bound - 1e-12 <= res.distance <= res.upper_bound + 1e-12}")
    print()

    # --- Case 6: budget exhaustion keeps interval; invalid input rejected
    print("=== Case 6: un-converged interval and explicit rejections ===")
    tri = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    far = [(5.0, 5.0, 5.0)]
    rc = distance_shelled(tri, far, LA=eye(0.5), max_iter=2)
    print(f"max_iter=2 -> status {rc.status} (never called 'intersecting'),"
          f" distance {rc.distance:.6f}")
    print(f"  preserved interval [{rc.lower_bound:.6f}, {rc.upper_bound:.6f}]"
          f"  residual {rc.residual:.2e}")
    for label, call in (
        ("singular LA", lambda: distance_shelled(
            tri, far, LA=((1.0, 2.0, 0.0), (2.0, 4.0, 0.0),
                          (0.0, 0.0, 1.0)))),
        ("zero direction", lambda: shell_support_offset(eye(1.0),
                                                        (0.0, 0.0, 0.0))),
        ("non-finite entry", lambda: distance_shelled(
            tri, far, LA=((1.0, 0.0, 0.0), (0.0, float('nan'), 0.0),
                          (0.0, 0.0, 1.0)))),
    ):
        try:
            call()
        except InputError as exc:
            print(f"{label:16s} -> InputError: {exc}")


if __name__ == "__main__":
    main()
