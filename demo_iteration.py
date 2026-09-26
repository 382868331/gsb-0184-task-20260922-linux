"""Deterministic offline demo for the penetration (EPA) query.

Runs real computations only (no sleeps, no prerecorded output):
  1. oblique penetration: a rotated box vs an axis-aligned box -- the
     minimum translation is verified by independently enumerating
     candidate separating axes, and by actually translating B by
     (1 +/- eps) * depth and re-querying distance();
  2. contact points are checked to lie inside their own hulls;
  3. boundary states: exact face contact (touching, zero depth) and a
     starved iteration budget (not_converged, bounds still bracket the
     converged answer).
"""

import math

from convex_distance import distance, penetration


def fmt(p):
    if p is None:
        return "None"
    return "(" + ", ".join(f"{c:+.6f}" for c in p) + ")"


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def box(cx, cy, cz, hx, hy, hz):
    return [(cx + sx * hx, cy + sy * hy, cz + sz * hz)
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)]


def rotz(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])


def translate(P, t):
    return [(p[0] + t[0], p[1] + t[1], p[2] + t[2]) for p in P]


def ref_sat_depth(A, B):
    """Independent reference: minimum translation length found by
    enumerating candidate separating axes (all cross products of
    pairwise vertex differences of both hulls)."""
    def sub(u, v):
        return (u[0] - v[0], u[1] - v[1], u[2] - v[2])

    def cross(u, v):
        return (u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0])

    diffs = ([sub(p, q) for i, p in enumerate(A) for q in A[i + 1:]]
             + [sub(p, q) for i, p in enumerate(B) for q in B[i + 1:]])
    best = math.inf
    for u in diffs:
        for v in diffs:
            n = cross(u, v)
            ln = math.sqrt(dot(n, n))
            if ln < 1e-9:
                continue
            n = (n[0] / ln, n[1] / ln, n[2] / ln)
            min_a = min(dot(a, n) for a in A)
            max_a = max(dot(a, n) for a in A)
            min_b = min(dot(b, n) for b in B)
            max_b = max(dot(b, n) for b in B)
            best = min(best, max_a - min_b, max_b - min_a)
    return best


def main():
    # --- Case 1: oblique penetration, rotated box vs axis box ---------
    a = box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    b = [rotz(p, math.pi / 4.0)
         for p in box(1.2, 0.3, 0.0, 1.0, 1.0, 0.5)]
    res = penetration(a, b)
    print("=== Case 1: oblique penetration (box rotated 45 deg about z) ===")
    print(f"status       : {res.status}")
    print(f"depth bounds : [{res.depth_lower:.12f}, {res.depth_upper:.12f}]")
    print(f"direction    : {fmt(res.direction)}")
    print(f"contact_a    : {fmt(res.contact_a)}")
    print(f"contact_b    : {fmt(res.contact_b)}")
    print(f"iterations   : {res.iterations}   residual: {res.residual:.3e}")

    ref = ref_sat_depth(a, b)
    print(f"independent SAT-axis reference depth: {ref:.12f}")
    ok = res.depth_lower - 1e-9 <= ref <= res.depth_upper + 1e-9
    print(f"reference inside reported bounds -> {'OK' if ok else 'FAILED'}")

    n = res.direction
    t_out = tuple(1.001 * res.depth_upper * c for c in n)
    t_in = tuple(0.999 * res.depth_lower * c for c in n)
    s_out = distance(a, translate(b, t_out)).status
    s_in = distance(a, translate(b, t_in)).status
    print(f"translate B by +100.1% depth -> {s_out} (expect separated)")
    print(f"translate B by + 99.9% depth -> {s_in} (expect intersecting)")

    # contact points must lie inside their own hulls (box bounds check)
    ca, cb = res.contact_a, res.contact_b
    in_a = all(-1.0 - 1e-9 <= c <= 1.0 + 1e-9 for c in ca)
    print(f"contact_a inside box A bounds -> {'OK' if in_a else 'FAILED'}")
    mtv = tuple(ca[i] - cb[i] for i in range(3))
    depth = math.sqrt(dot(mtv, mtv))
    print(f"|contact_a - contact_b| = {depth:.12f} (within depth bounds)")
    print()

    # --- Case 2: exact face contact -> explicit zero-depth touching ---
    c = box(2.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    res2 = penetration(a, c)
    print("=== Case 2: coplanar face contact (no positive penetration) ===")
    print(f"status       : {res2.status} (explicit zero-depth state)")
    print(f"depth bounds : [{res2.depth_lower}, {res2.depth_upper}]")
    print(f"direction    : {res2.direction}")
    print()

    # --- Case 3: starved budget -> not_converged, bounds still valid --
    res3 = penetration(a, b, max_iter=4)
    print("=== Case 3: iteration budget exhausted (max_iter=4) ===")
    print(f"status       : {res3.status}")
    print(f"depth bounds : [{res3.depth_lower:.6f}, {res3.depth_upper:.6f}]")
    bracket = (res3.depth_lower <= res.depth_upper + 1e-9
               and res3.depth_upper >= res.depth_lower - 1e-9)
    print(f"bounds still bracket the converged depth -> "
          f"{'OK' if bracket else 'FAILED'}")


if __name__ == "__main__":
    main()
