"""Deterministic offline demo for the translational sweep.

Runs real sweep_contact computations (no sleeps, no prerecorded output):
  1. a head-on impact with entry/exit times and a reviewable contact
     certificate (normal and meeting point);
  2. boundary cases: grazing single-instant contact, initial overlap
     with an analytic exit, relative rest, and a complete miss;
  3. contract violations (2-component velocity, inverted time window,
     non-finite time) actually triggering InputError.

Expected hand-computed values are printed alongside the results.
"""

from convex_distance import InputError, sweep_contact


def box(lo, hi):
    return [(x, y, z)
            for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])]


def fmt(p):
    return "(" + ", ".join(f"{c:+.6f}" for c in p) + ")"


def show(title, res, expected=None):
    print(f"--- {title}")
    print(f"    status         : {res.status}"
          + (f"   (expected {expected})" if expected else ""))
    print(f"    interval       : {res.interval}")
    print(f"    initial state  : {res.initial_state}"
          f"   relative rest: {res.relative_rest}")
    if res.first_contact is not None:
        c = res.first_contact
        print(f"    first contact  : t={c.time:.9f}  kind={c.kind}")
        print(f"                     point={fmt(c.point_a)}"
              f"  normal={fmt(c.normal) if c.normal else None}")
    if res.last_contact is not None and (
            res.first_contact is None
            or res.last_contact.time != res.first_contact.time
            or res.last_contact.kind != res.first_contact.kind):
        c = res.last_contact
        print(f"    last contact   : t={c.time:.9f}  kind={c.kind}")
    print(f"    min separation : {res.closest_distance:.3e}"
          f" at t={res.closest_time:.6f}   GJK iterations: {res.iterations}")
    print()


def main():
    cube = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    target = box((3.0, 0.0, 0.0), (4.0, 1.0, 1.0))

    print("=== 1. Head-on impact: unit cube moves +x at speed 1 ===")
    # Face contact when A's max x reaches B's min x: 1 + t = 3 -> t=2;
    # exit when A's min x passes B's max x: t = 4.
    r = sweep_contact(cube, target, (1.0, 0.0, 0.0), 0.0, 6.0)
    show("window [0, 6]", r, expected="intersecting [2.0, 4.0]")

    print("=== 2. Boundary cases ===")
    # 2a. Grazing corner: path (t, .5, 3-.5t) touches corner (4,.5,1)
    #     at exactly t=4.
    r = sweep_contact([(0.0, 0.5, 3.0)],
                      box((3.0, 0.0, 0.0), (4.0, 1.0, 1.0)),
                      (1.0, 0.0, -0.5), 0.0, 8.0)
    show("grazing corner (single instant, expected touching t=4)", r,
         expected="touching (4.0, 4.0)")

    # 2b. Initial overlap of width 0.5, exit at t = 1.5.
    overlapping = box((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))
    r = sweep_contact(cube, overlapping, (1.0, 0.0, 0.0), 0.0, 6.0)
    show("initial overlap (expected intersecting [0, 1.5])", r,
         expected="intersecting (0.0, 1.5)")

    # 2c. Relative rest, separated: gap is a constant 2.
    r = sweep_contact(cube, target, (0.0, 0.0, 0.0), 0.0, 6.0)
    show("relative rest, separated (expected d=2)", r,
         expected="separated")

    # 2d. Complete miss: transverse motion, gap only grows.
    r = sweep_contact(cube, target, (0.0, 2.0, 0.0), 0.0, 6.0)
    show("complete miss (expected min d=2 at t=0)", r,
         expected="separated")

    # 2e. Contact exactly at the window end.
    r = sweep_contact(cube, target, (1.0, 0.0, 0.0), 0.0, 2.0)
    show("contact exactly at t_end (expected touching t=2)", r,
         expected="touching (2.0, 2.0)")

    print("=== 3. Invalid inputs are rejected, no partial state ===")
    try:
        sweep_contact(cube, target, (1.0, 0.0), 0.0, 6.0)
    except InputError as exc:
        print(f"2-component velocity -> InputError: {exc}")
    try:
        sweep_contact(cube, target, (1.0, 0.0, 0.0), 6.0, 0.0)
    except InputError as exc:
        print(f"inverted window     -> InputError: {exc}")
    try:
        sweep_contact(cube, target, (1.0, 0.0, float("nan")),
                      0.0, 6.0)
    except InputError as exc:
        print(f"NaN velocity        -> InputError: {exc}")

    # The rejected calls leave no state: a valid call still works.
    r = sweep_contact(cube, target, (1.0, 0.0, 0.0), 0.0, 6.0)
    assert r.status == "intersecting" and abs(r.interval[0] - 2.0) < 1e-9
    print("\nvalid call after rejections still reports interval "
          f"[2, 4]: {r.status} {tuple(round(t, 9) for t in r.interval)}")


if __name__ == "__main__":
    main()
