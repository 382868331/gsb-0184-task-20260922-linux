"""Tests for convex_distance.

Independent references (point-segment formula, axis-aligned box formula,
exhaustive active-set enumeration over small convex combinations) are
implemented here from scratch and never call the module under test.
"""

import itertools
import math
import random
import unittest

from convex_distance import (
    InputError,
    distance,
    sweep_contact,
    STATUS_SEPARATED,
    STATUS_INTERSECTING,
    STATUS_NOT_CONVERGED,
    STATUS_TOUCHING,
)


# ---------------------------------------------------------------------------
# independent references
# ---------------------------------------------------------------------------

def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _sub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def ref_point_segment(p, a, b):
    """Analytic distance from point p to segment [a, b]."""
    ab = _sub(b, a)
    denom = _dot(ab, ab)
    if denom == 0.0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, _dot(_sub(p, a), ab) / denom))
    c = (a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2])
    return math.sqrt(_dot(_sub(p, c), _sub(p, c)))


def ref_aabb(lo1, hi1, lo2, hi2):
    """Analytic distance between two axis-aligned boxes."""
    s = 0.0
    for k in range(3):
        gap = max(0.0, lo1[k] - hi2[k], lo2[k] - hi1[k])
        s += gap * gap
    return math.sqrt(s)


def _solve_linear(mat, rhs):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(mat)
    m = [row[:] + [rhs[i]] for i, row in enumerate(mat)]
    scale = max((abs(x) for row in m for x in row[:n]), default=0.0)
    if scale == 0.0:
        return None
    eps = 1e-12 * scale
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) <= eps:
            return None
        m[col], m[piv] = m[piv], m[col]
        inv = 1.0 / m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] * inv
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / m[r][r]
    return x


def _simplex_pair_d2(S, T):
    """Squared distance between conv(S) and conv(T) when both closest
    points lie in the relative interiors; None otherwise.

    Solves the equality-constrained QP  min ||sum a_i s_i - sum b_j t_j||^2
    s.t. sum a = sum b = 1  via its KKT system, then checks a, b >= 0.
    """
    k, l = len(S), len(T)
    n = k + l + 2
    mat = [[0.0] * n for _ in range(n)]
    rhs = [0.0] * n
    for i in range(k):
        for j in range(k):
            mat[i][j] = _dot(S[i], S[j])
        for j in range(l):
            mat[i][k + j] = -_dot(S[i], T[j])
        mat[i][k + l] = 1.0
        mat[k + l][i] = 1.0
    for i in range(l):
        for j in range(k):
            mat[k + i][j] = -_dot(T[i], S[j])
        for j in range(l):
            mat[k + i][k + j] = _dot(T[i], T[j])
        mat[k + i][k + l + 1] = 1.0
        mat[k + l + 1][k + i] = 1.0
    rhs[k + l] = 1.0
    rhs[k + l + 1] = 1.0
    x = _solve_linear(mat, rhs)
    if x is None:
        return None
    alpha, beta = x[:k], x[k:k + l]
    if any(a < -1e-9 for a in alpha) or any(b < -1e-9 for b in beta):
        return None
    pa = [0.0, 0.0, 0.0]
    pb = [0.0, 0.0, 0.0]
    for a, s in zip(alpha, S):
        for c in range(3):
            pa[c] += a * s[c]
    for b, t in zip(beta, T):
        for c in range(3):
            pb[c] += b * t[c]
    return sum((pa[c] - pb[c]) ** 2 for c in range(3))


def ref_convex_distance(A, B):
    """Exhaustive active-set reference: by Caratheodory the closest pair
    is realised on simplices of at most 4 vertices from each hull, so
    enumerating all subset pairs up to size 4 gives the global minimum.
    """
    best = math.inf
    for k in range(1, min(4, len(A)) + 1):
        for S in itertools.combinations(A, k):
            for l in range(1, min(4, len(B)) + 1):
                for T in itertools.combinations(B, l):
                    d2 = _simplex_pair_d2(S, T)
                    if d2 is not None and d2 < best:
                        best = d2
    return math.sqrt(best)


def box_corners(lo, hi):
    return [(x, y, z)
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])]


# ---------------------------------------------------------------------------
# correctness against independent references
# ---------------------------------------------------------------------------

class TestPointSegment(unittest.TestCase):
    def test_point_vs_segment_analytic(self):
        rng = random.Random(101)
        for _ in range(30):
            p = tuple(rng.uniform(-5, 5) for _ in range(3))
            a = tuple(rng.uniform(-5, 5) for _ in range(3))
            b = tuple(rng.uniform(-5, 5) for _ in range(3))
            res = distance([p], [a, b])
            expected = ref_point_segment(p, a, b)
            if expected > 1e-9:
                self.assertEqual(res.status, STATUS_SEPARATED)
                self.assertAlmostEqual(res.distance, expected, delta=1e-8)
            else:
                self.assertEqual(res.status, STATUS_INTERSECTING)

    def test_point_on_segment_is_intersecting(self):
        res = distance([(0.5, 0.0, 0.0)], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertLessEqual(res.distance, 1e-9)

    def test_degenerate_segment_point_point(self):
        # segment whose endpoints coincide = a point
        res = distance([(3.0, 4.0, 0.0)], [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)])
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, math.sqrt(14.0), delta=1e-9)


class TestAxisAlignedBoxes(unittest.TestCase):
    def test_boxes_against_analytic(self):
        rng = random.Random(202)
        cases = 0
        while cases < 25:
            lo1 = tuple(rng.uniform(-4, 4) for _ in range(3))
            hi1 = tuple(lo1[k] + rng.uniform(0.1, 2.0) for k in range(3))
            lo2 = tuple(rng.uniform(-4, 4) for _ in range(3))
            hi2 = tuple(lo2[k] + rng.uniform(0.1, 2.0) for k in range(3))
            expected = ref_aabb(lo1, hi1, lo2, hi2)
            res = distance(box_corners(lo1, hi1), box_corners(lo2, hi2))
            if expected > 1e-9:
                self.assertEqual(res.status, STATUS_SEPARATED)
                self.assertAlmostEqual(res.distance, expected, delta=1e-8)
                cases += 1
            else:
                self.assertEqual(res.status, STATUS_INTERSECTING)
                cases += 1

    def test_touching_boxes(self):
        # share exactly the face x = 1
        a = box_corners((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        b = box_corners((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        res = distance(a, b)
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertLessEqual(res.distance, 1e-9)


class TestExhaustiveSmall(unittest.TestCase):
    def test_random_small_sets(self):
        rng = random.Random(303)
        for trial in range(25):
            na = rng.randint(1, 6)
            nb = rng.randint(1, 6)
            A = [tuple(rng.uniform(-3, 3) for _ in range(3)) for _ in range(na)]
            B = [tuple(rng.uniform(-3, 3) + 2.0 for _ in range(3))
                 for _ in range(nb)]
            expected = ref_convex_distance(A, B)
            res = distance(A, B)
            if expected > 1e-8:
                self.assertEqual(res.status, STATUS_SEPARATED)
                self.assertAlmostEqual(res.distance, expected, delta=1e-7,
                                       msg=f"trial {trial}")
                # witnesses realise the distance
                gap = math.dist(res.witness_a, res.witness_b)
                self.assertAlmostEqual(gap, res.distance, delta=1e-7)
            else:
                self.assertEqual(res.status, STATUS_INTERSECTING)


class TestDegenerateAndRobust(unittest.TestCase):
    def test_coplanar_shapes(self):
        # two squares in the plane z = 0
        A = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        B = [(3.0, 0.0, 0.0), (4.0, 0.0, 0.0),
             (4.0, 1.0, 0.0), (3.0, 1.0, 0.0)]
        res = distance(A, B)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 2.0, delta=1e-9)

    def test_coplanar_touching(self):
        A = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        B = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0)]
        res = distance(A, B)  # share the vertex (1,0,0)
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertLessEqual(res.distance, 1e-9)

    def test_duplicate_points_ignored(self):
        A = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        B = [(5.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 5.0, 0.0)]
        clean_A = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        clean_B = [(5.0, 0.0, 0.0), (5.0, 5.0, 0.0)]
        r1 = distance(A, B)
        r2 = distance(clean_A, clean_B)
        self.assertEqual(r1.status, r2.status)
        self.assertAlmostEqual(r1.distance, r2.distance, delta=1e-12)

    def test_skew_segments(self):
        A = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        B = [(1.0, 3.0, -1.0), (1.0, 3.0, 1.0)]
        res = distance(A, B)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 3.0, delta=1e-9)
        self.assertAlmostEqual(res.witness_a[0], 1.0, delta=1e-9)
        self.assertAlmostEqual(res.witness_b[1], 3.0, delta=1e-9)

    def test_triangle_vs_point(self):
        tri = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
        p = [(0.5, 0.5, 4.0)]
        res = distance(p, tri)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 4.0, delta=1e-9)

    def test_translation_invariance(self):
        rng = random.Random(404)
        A = [tuple(rng.uniform(-2, 2) for _ in range(3)) for _ in range(6)]
        B = [tuple(rng.uniform(-2, 2) + 3.0 for _ in range(3))
             for _ in range(6)]
        t = (7.25, -3.5, 11.75)
        At = [tuple(p[k] + t[k] for k in range(3)) for p in A]
        Bt = [tuple(p[k] + t[k] for k in range(3)) for p in B]
        r1 = distance(A, B)
        r2 = distance(At, Bt)
        self.assertEqual(r1.status, r2.status)
        self.assertAlmostEqual(r1.distance, r2.distance, delta=1e-8)
        for k in range(3):
            self.assertAlmostEqual(r2.witness_a[k] - t[k], r1.witness_a[k],
                                   delta=1e-8)
            self.assertAlmostEqual(r2.witness_b[k] - t[k], r1.witness_b[k],
                                   delta=1e-8)


class TestCertificate(unittest.TestCase):
    def test_separating_axis_certifies_all_vertices(self):
        rng = random.Random(505)
        checked = 0
        while checked < 15:
            A = [tuple(rng.uniform(-2, 2) for _ in range(3))
                 for _ in range(rng.randint(4, 8))]
            B = [tuple(rng.uniform(-2, 2) + rng.uniform(2, 6)
                       for _ in range(3)) for _ in range(rng.randint(4, 8))]
            res = distance(A, B)
            if res.status != STATUS_SEPARATED:
                continue
            n = res.separating_axis
            self.assertAlmostEqual(_dot(n, n), 1.0, delta=1e-9)
            wa_n = _dot(res.witness_a, n)
            wb_n = _dot(res.witness_b, n)
            for a in A:
                self.assertGreaterEqual(_dot(a, n), wa_n - 1e-7)
            for b in B:
                self.assertLessEqual(_dot(b, n), wb_n + 1e-7)
            self.assertAlmostEqual(wa_n - wb_n, res.distance, delta=1e-7)
            self.assertLessEqual(res.residual, 1e-9)
            checked += 1


class TestBudgetAndStatus(unittest.TestCase):
    def test_budget_exhausted_is_not_intersecting(self):
        A = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
             (0.0, 0.0, 1.0)]
        B = [(3.0, 3.0, 3.0), (4.0, 3.0, 3.0), (3.0, 4.0, 3.0),
             (3.0, 3.0, 4.0)]
        res = distance(A, B, max_iter=1)
        self.assertEqual(res.status, STATUS_NOT_CONVERGED)
        self.assertFalse(res.converged)
        self.assertEqual(res.iterations, 1)
        # budget exhaustion must not be confused with a proven result
        self.assertNotEqual(res.status, STATUS_INTERSECTING)
        self.assertGreater(res.residual, 0.0)

    def test_max_iter_capped_at_128(self):
        with self.assertRaises(InputError):
            distance([(0, 0, 0)], [(1, 0, 0)], max_iter=129)


class TestValidation(unittest.TestCase):
    def test_nan_and_inf_rejected(self):
        with self.assertRaisesRegex(InputError, r"A\[0\]\[2\]"):
            distance([(0.0, 0.0, float("nan"))], [(1.0, 0.0, 0.0)])
        with self.assertRaisesRegex(InputError, r"B\[1\]\[0\]"):
            distance([(0.0, 0.0, 0.0)],
                     [(1.0, 0.0, 0.0), (float("inf"), 0.0, 0.0)])

    def test_bool_coordinate_rejected(self):
        with self.assertRaisesRegex(InputError, r"A\[0\]\[0\]"):
            distance([(True, 0.0, 0.0)], [(1.0, 0.0, 0.0)])

    def test_bool_max_iter_and_tol_rejected(self):
        with self.assertRaisesRegex(InputError, "max_iter"):
            distance([(0, 0, 0)], [(1, 0, 0)], max_iter=True)
        with self.assertRaisesRegex(InputError, "tol"):
            distance([(0, 0, 0)], [(1, 0, 0)], tol=False)

    def test_cardinality_limits(self):
        with self.assertRaisesRegex(InputError, "A: expected 1..64"):
            distance([], [(1.0, 0.0, 0.0)])
        with self.assertRaisesRegex(InputError, "B: expected 1..64"):
            distance([(0.0, 0.0, 0.0)],
                     [(float(i), 0.0, 0.0) for i in range(65)])

    def test_shape_errors(self):
        with self.assertRaisesRegex(InputError, r"A\[0\]: expected exactly 3"):
            distance([(0.0, 0.0)], [(1.0, 0.0, 0.0)])
        with self.assertRaisesRegex(InputError, "A: expected a sequence"):
            distance("not points", [(1.0, 0.0, 0.0)])
        with self.assertRaisesRegex(InputError, r"A\[0\]\[1\]"):
            distance([(0.0, "x", 0.0)], [(1.0, 0.0, 0.0)])

    def test_tol_range(self):
        for bad in (0.0, -1e-9, 1.0, float("nan"), float("inf")):
            with self.assertRaises(InputError):
                distance([(0, 0, 0)], [(1, 0, 0)], tol=bad)

    def test_max_iter_range(self):
        for bad in (0, -3, 2.5, "10"):
            with self.assertRaises(InputError):
                distance([(0, 0, 0)], [(1, 0, 0)], max_iter=bad)

    def test_fraction_and_int_inputs_accepted(self):
        from fractions import Fraction as F
        res = distance([(F(0), F(0), F(0))], [(F(3), 4, 0)])
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 5.0, delta=1e-12)


# ===========================================================================
# sweep_contact
#
# Independent references used below (none call sweep_contact to produce an
# expected value):
#   * ref_box_sweep_interval: closed-form overlap interval of two
#     axis-aligned boxes under a constant translation (1D per axis);
#   * point-vs-point / point-vs-box hand-computed examples;
#   * a dense time grid of the *static* distance() as a supplementary
#     cross-check (presence/absence of contact, minimum distance).
# ===========================================================================

def box_pts(lo, hi):
    return [(x, y, z)
            for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])]


def ref_box_sweep_interval(loA, hiA, vel, loB, hiB, t0, t1, eps=1e-9):
    """Closed-form status/interval of A+vel*t overlapping box B.

    Per axis the overlap is an interval of times; intersecting the three
    intervals with the window gives the result.  Returns
    "separated" / "touching" / "intersecting" and (lo, hi) or None.
    """
    enter, exit_ = t0, t1
    for k in range(3):
        vk = vel[k]
        if abs(vk) < 1e-14:
            if loA[k] > hiB[k] + eps or hiA[k] < loB[k] - eps:
                return "separated", None
            continue
        ta = (hiB[k] - loA[k]) / vk
        tb = (loB[k] - hiA[k]) / vk
        a, b = min(ta, tb), max(ta, tb)
        enter, exit_ = max(enter, a), min(exit_, b)
        if exit_ < enter - eps:
            return "separated", None
    if exit_ - enter <= eps:
        return "touching", (enter, enter)
    return "intersecting", (enter, exit_)


def grid_status(A, B, vel, t0, t1, n=160):
    """Supplementary grid oracle built from the static distance only."""
    ts = [t0 + (t1 - t0) * i / n for i in range(n + 1)]
    ds = []
    for t in ts:
        At = [(p[0] + vel[0] * t, p[1] + vel[1] * t,
               p[2] + vel[2] * t) for p in A]
        r = distance(At, B)
        ds.append(0.0 if r.status == STATUS_INTERSECTING else r.distance)
    imin = min(range(n + 1), key=lambda i: ds[i])
    return ts, ds, ts[imin], ds[imin]


def point_in_box(p, lo, hi, eps=1e-7):
    return all(lo[k] - eps <= p[k] <= hi[k] + eps for k in range(3))


CUBE_A = box_pts((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
CUBE_B = box_pts((3.0, 0.0, 0.0), (4.0, 1.0, 1.0))
OVERLAP_B = box_pts((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))


class TestSweepHandComputed(unittest.TestCase):
    def test_head_on_interval_exact(self):
        # A in [0,1]^3 moves +x at speed 1 into B in [3,4]x[0,1]x[0,1]:
        # face contact at t=2, exit (A fully past B) at t=4.
        res = sweep_contact(CUBE_A, CUBE_B, (1.0, 0.0, 0.0), 0.0, 6.0)
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertEqual(res.initial_state, STATUS_SEPARATED)
        self.assertFalse(res.relative_rest)
        self.assertAlmostEqual(res.interval[0], 2.0, delta=1e-9)
        self.assertAlmostEqual(res.interval[1], 4.0, delta=1e-9)
        self.assertEqual(res.first_contact.kind, "impact")
        self.assertEqual(res.last_contact.kind, "separation")
        self.assertAlmostEqual(res.first_contact.time, 2.0, delta=1e-9)
        self.assertAlmostEqual(res.last_contact.time, 4.0, delta=1e-9)
        # impact witnesses coincide at the touching face point (3, y, z)
        p = res.first_contact.point_a
        self.assertAlmostEqual(p[0], 3.0, delta=1e-7)
        self.assertTrue(0.0 - 1e-7 <= p[1] <= 1.0 + 1e-7)
        self.assertTrue(0.0 - 1e-7 <= p[2] <= 1.0 + 1e-7)
        self.assertAlmostEqual(
            math.dist(res.first_contact.point_a,
                      res.first_contact.point_b), 0.0, delta=1e-7)
        # unit normal points from B toward A (the -x face)
        n = res.first_contact.normal
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in n)), 1.0,
                               delta=1e-9)
        self.assertLess(n[0], -0.999)
        self.assertEqual(res.closest_distance, 0.0)

    def test_complete_miss_minimum_at_start(self):
        # transverse +y motion: gap starts at 2 and only grows
        res = sweep_contact(CUBE_A, CUBE_B, (0.0, 2.0, 0.0), 0.0, 6.0)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertIsNone(res.interval)
        self.assertIsNone(res.first_contact)
        self.assertIsNone(res.last_contact)
        self.assertAlmostEqual(res.closest_distance, 2.0, delta=1e-9)
        self.assertAlmostEqual(res.closest_time, 0.0, delta=1e-9)

    def test_grazing_corner_single_instant(self):
        # point p(t) = (t, 0.5, 3 - 0.5 t) meets the box corner
        # (4, .5, 1) exactly at t=4 and nowhere else (hand-derived:
        # d^2 = (t-4)^2 + (2-0.5t)^2 on t in [3,4], unique zero at 4).
        B = box_pts((3.0, 0.0, 0.0), (4.0, 1.0, 1.0))
        res = sweep_contact([(0.0, 0.5, 3.0)], B,
                            (1.0, 0.0, -0.5), 0.0, 8.0)
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertEqual(res.interval[0], res.interval[1])
        self.assertAlmostEqual(res.first_contact.time, 4.0, delta=1e-8)
        p = res.first_contact.point_a
        self.assertAlmostEqual(p[0], 4.0, delta=1e-7)
        self.assertAlmostEqual(p[1], 0.5, delta=1e-7)
        self.assertAlmostEqual(p[2], 1.0, delta=1e-7)
        # normal lies in the approach quadrant (-x, +z) of the corner
        n = res.first_contact.normal
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in n)), 1.0,
                               delta=1e-9)
        self.assertLessEqual(n[0], 1e-6)
        self.assertGreaterEqual(n[2], -1e-6)
        self.assertEqual(res.first_contact.kind, "touch")

    def test_point_through_point_is_touch(self):
        res = sweep_contact([(-5.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)],
                            (1.0, 0.0, 0.0), 0.0, 10.0)
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertAlmostEqual(res.interval[0], 5.0, delta=1e-9)
        self.assertAlmostEqual(res.first_contact.point_a[0], 0.0,
                               delta=1e-7)
        n = res.first_contact.normal
        self.assertAlmostEqual(n[0], -1.0, delta=1e-7)

    def test_point_sweeping_across_box_face(self):
        # point travels in the top face plane z=2 of B whose top is 2:
        # it is on the surface for x in [3,4] -> interval [3,4].
        B = box_pts((3.0, 0.0, 0.0), (4.0, 1.0, 2.0))
        res = sweep_contact([(0.0, 0.5, 2.0)], B, (1.0, 0.0, 0.0),
                            0.0, 6.0)
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertAlmostEqual(res.interval[0], 3.0, delta=1e-8)
        self.assertAlmostEqual(res.interval[1], 4.0, delta=1e-8)


class TestSweepInitialOverlapAndRest(unittest.TestCase):
    def test_initial_overlap_then_exit(self):
        # overlap width in x is 0.5 initially; exit at t = 1.5
        res = sweep_contact(CUBE_A, OVERLAP_B, (1.0, 0.0, 0.0),
                            0.0, 6.0)
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertEqual(res.initial_state, STATUS_INTERSECTING)
        self.assertAlmostEqual(res.interval[0], 0.0, delta=1e-12)
        self.assertAlmostEqual(res.interval[1], 1.5, delta=1e-8)
        self.assertEqual(res.first_contact.kind, "initial_overlap")
        self.assertEqual(res.first_contact.time, 0.0)
        self.assertEqual(res.last_contact.kind, "separation")

    def test_overlap_covers_whole_window(self):
        res = sweep_contact(CUBE_A, OVERLAP_B, (1.0, 0.0, 0.0),
                            0.0, 1.0)
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertEqual(res.interval, (0.0, 1.0))
        self.assertEqual(res.last_contact.kind, "window_end")

    def test_relative_rest_separated(self):
        res = sweep_contact(CUBE_A, CUBE_B, (0.0, 0.0, 0.0),
                            0.0, 6.0)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertTrue(res.relative_rest)
        self.assertAlmostEqual(res.closest_distance, 2.0, delta=1e-9)

    def test_relative_rest_overlapping(self):
        res = sweep_contact(CUBE_A, OVERLAP_B, (0.0, 0.0, 0.0),
                            1.0, 7.0)
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertTrue(res.relative_rest)
        self.assertEqual(res.interval, (1.0, 7.0))
        self.assertEqual(res.first_contact.kind, "initial_overlap")
        self.assertEqual(res.last_contact.kind, "rest_overlap")

    def test_contact_at_window_start(self):
        res = sweep_contact(CUBE_A, CUBE_B, (1.0, 0.0, 0.0),
                            2.0, 6.0)
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertEqual(res.initial_state, STATUS_INTERSECTING)
        self.assertAlmostEqual(res.interval[0], 2.0, delta=1e-10)
        self.assertAlmostEqual(res.interval[1], 4.0, delta=1e-8)

    def test_contact_at_window_end_is_touch(self):
        res = sweep_contact(CUBE_A, CUBE_B, (1.0, 0.0, 0.0),
                            0.0, 2.0)
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertAlmostEqual(res.interval[0], 2.0, delta=1e-9)
        self.assertAlmostEqual(res.interval[1], 2.0, delta=1e-9)

    def test_still_overlapping_at_window_end(self):
        res = sweep_contact(CUBE_A, CUBE_B, (1.0, 0.0, 0.0),
                            0.0, 3.0)
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertAlmostEqual(res.interval[0], 2.0, delta=1e-9)
        self.assertAlmostEqual(res.interval[1], 3.0, delta=1e-12)
        self.assertEqual(res.last_contact.kind, "window_end")

    def test_touch_at_start_moving_apart(self):
        # faces already coincide at t=0 (A max x = B min x = 1) and A
        # moves away (-x): a single contact instant at the window start
        B = box_pts((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        res = sweep_contact(CUBE_A, B, (-1.0, 0.0, 0.0), 0.0, 3.0)
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertEqual(res.initial_state, STATUS_INTERSECTING)
        self.assertAlmostEqual(res.first_contact.time, 0.0, delta=1e-10)


class TestSweepRandomBoxes(unittest.TestCase):
    def test_against_closed_form_box_reference(self):
        rng = random.Random(707)
        checked = 0
        while checked < 60:
            loA = tuple(rng.uniform(-3, 1) for _ in range(3))
            hiA = tuple(loA[k] + rng.uniform(0.3, 2.0)
                        for k in range(3))
            loB = tuple(rng.uniform(-2, 4) for _ in range(3))
            hiB = tuple(loB[k] + rng.uniform(0.3, 2.0)
                        for k in range(3))
            vel = tuple(rng.uniform(-3, 3) for _ in range(3))
            t0 = rng.uniform(-1, 1)
            t1 = t0 + rng.uniform(0.5, 5.0)
            A, B = box_pts(loA, hiA), box_pts(loB, hiB)
            want, wiv = ref_box_sweep_interval(
                loA, hiA, vel, loB, hiB, t0, t1)
            res = sweep_contact(A, B, vel, t0, t1)
            self.assertNotEqual(res.status, STATUS_NOT_CONVERGED)
            checked += 1
            # status must agree (touching/intersecting may swap only at
            # the reference's own epsilon boundary)
            if res.status != want:
                self.assertTrue(
                    {res.status, want} <= {STATUS_TOUCHING,
                                           STATUS_INTERSECTING})
            if wiv is not None:
                self.assertIsNotNone(res.interval)
                self.assertAlmostEqual(res.interval[0], wiv[0],
                                       delta=1e-6)
                self.assertAlmostEqual(res.interval[1], wiv[1],
                                       delta=1e-6)
            else:
                self.assertIsNone(res.interval)
            # contact point lies in both bodies at the entry instant
            if res.first_contact is not None and want != "separated":
                tc = res.first_contact.time
                loAt = tuple(loA[k] + vel[k] * tc for k in range(3))
                hiAt = tuple(hiA[k] + vel[k] * tc for k in range(3))
                p = res.first_contact.point_a
                self.assertTrue(point_in_box(p, loAt, hiAt))
                self.assertTrue(point_in_box(p, loB, hiB))
            # a reported miss must show strictly positive grid distance
            if res.status == STATUS_SEPARATED:
                _, _, _, dg = grid_status(A, B, vel, t0, t1, 120)
                self.assertGreater(dg, 1e-7)

    def test_minimum_distance_matches_grid_on_misses(self):
        rng = random.Random(808)
        accepted = 0
        while accepted < 20:
            loA = tuple(rng.uniform(-2, 1) for _ in range(3))
            hiA = tuple(loA[k] + rng.uniform(0.3, 1.5)
                        for k in range(3))
            loB = tuple(rng.uniform(4, 6) for _ in range(3))
            hiB = tuple(loB[k] + rng.uniform(0.3, 1.5)
                        for k in range(3))
            vel = tuple(rng.uniform(-1.5, 1.5) for _ in range(3))
            t0, t1 = 0.0, 3.0
            A, B = box_pts(loA, hiA), box_pts(loB, hiB)
            res = sweep_contact(A, B, vel, t0, t1)
            if res.status != STATUS_SEPARATED:
                continue
            _, _, _, dg = grid_status(A, B, vel, t0, t1, 200)
            # grid minimum overestimates; analytic value must not be
            # above the grid by more than its discretisation
            self.assertLessEqual(
                res.closest_distance, dg + 5e-3 * max(1.0, dg))
            accepted += 1


class TestSweepRandomGeneric(unittest.TestCase):
    def test_small_clouds_against_grid(self):
        rng = random.Random(909)
        checked = 0
        while checked < 40:
            na = rng.randint(1, 6)
            nb = rng.randint(1, 6)
            A = [tuple(rng.uniform(-3, 3) for _ in range(3))
                 for _ in range(na)]
            B = [tuple(rng.uniform(-3, 3) + 2.5 for _ in range(3))
                 for _ in range(nb)]
            vel = tuple(rng.uniform(-4, 4) for _ in range(3))
            t0, t1 = 0.0, rng.uniform(0.5, 4.0)
            res = sweep_contact(A, B, vel, t0, t1)
            self.assertNotEqual(res.status, STATUS_NOT_CONVERGED)
            ts, ds, tg, dg = grid_status(A, B, vel, t0, t1, 200)
            checked += 1
            if res.status == STATUS_SEPARATED:
                self.assertGreater(dg, 1e-7)
                self.assertLessEqual(
                    res.closest_distance, dg + 5e-3 * max(1.0, dg))
            else:
                self.assertLessEqual(dg, 1e-7)
                first_grid = next(t for t, d in zip(ts, ds)
                                  if d <= 1e-7)
                self.assertLess(
                    abs(first_grid - res.interval[0]),
                    (t1 - t0) / 100 + 1e-6)


class TestSweepSymmetryAndDeterminism(unittest.TestCase):
    def test_moving_B_with_opposite_velocity_matches(self):
        # A(t) = A + v t against fixed B  <=>  fixed A against B - v t,
        # i.e. sweep_contact(B, A, -v, ...) with bodies swapped.
        rng = random.Random(111)
        for _ in range(20):
            A = [tuple(rng.uniform(-2, 2) for _ in range(3))
                 for _ in range(rng.randint(2, 6))]
            B = [tuple(rng.uniform(-2, 2) + 2.5 for _ in range(3))
                 for _ in range(rng.randint(2, 6))]
            vel = tuple(rng.uniform(-3, 3) for _ in range(3))
            r1 = sweep_contact(A, B, vel, 0.0, 4.0)
            r2 = sweep_contact(B, A, tuple(-c for c in vel), 0.0, 4.0)
            self.assertEqual(r1.status, r2.status)
            if r1.interval is not None:
                self.assertAlmostEqual(r1.interval[0], r2.interval[0],
                                       delta=1e-7)
                self.assertAlmostEqual(r1.interval[1], r2.interval[1],
                                       delta=1e-7)

    def test_rotation_covariance(self):
        # rotate both bodies and the velocity by 90 deg about z
        rot = lambda p: (-p[1], p[0], p[2])
        vel = (1.0, 0.0, 0.0)
        r1 = sweep_contact(CUBE_A, CUBE_B, vel, 0.0, 6.0)
        Ar = [rot(p) for p in CUBE_A]
        Br = [rot(p) for p in CUBE_B]
        vr = rot(vel)
        r2 = sweep_contact(Ar, Br, vr, 0.0, 6.0)
        self.assertEqual(r1.status, r2.status)
        self.assertAlmostEqual(r1.interval[0], r2.interval[0],
                               delta=1e-8)
        self.assertAlmostEqual(r1.interval[1], r2.interval[1],
                               delta=1e-8)
        p1, p2 = r1.first_contact.point_a, r2.first_contact.point_a
        for k in range(3):
            self.assertAlmostEqual(rot(p1)[k], p2[k], delta=1e-7)

    def test_deterministic(self):
        args = (CUBE_A, CUBE_B, (1.3, -0.7, 0.4), -1.0, 4.0)
        r1 = sweep_contact(*args)
        r2 = sweep_contact(*args)
        self.assertEqual(r1, r2)


class TestSweepBudget(unittest.TestCase):
    def test_budget_exhaustion_is_reported(self):
        res = sweep_contact(CUBE_A, CUBE_B, (1.0, 0.0, 0.0),
                            0.0, 6.0, max_iter=1)
        self.assertEqual(res.status, STATUS_NOT_CONVERGED)
        self.assertFalse(res.converged)
        self.assertIsNone(res.interval)

    def test_max_iter_out_of_range(self):
        for bad in (0, 129, -1):
            with self.assertRaises(InputError):
                sweep_contact(CUBE_A, CUBE_B, (1, 0, 0), 0, 1,
                              max_iter=bad)


class TestSweepValidation(unittest.TestCase):
    def _good(self):
        return dict(A=[(0.0, 0.0, 0.0)], B=[(1.0, 0.0, 0.0)],
                    velocity=(1.0, 0.0, 0.0), t_start=0.0, t_end=1.0)

    def test_velocity_shape(self):
        kw = self._good()
        kw["velocity"] = (1.0, 0.0)
        with self.assertRaisesRegex(InputError, "velocity: expected"):
            sweep_contact(**kw)
        kw = self._good()
        kw["velocity"] = "nope"
        with self.assertRaisesRegex(InputError, "velocity"):
            sweep_contact(**kw)
        kw = self._good()
        kw["velocity"] = (True, 0.0, 0.0)
        with self.assertRaisesRegex(InputError, r"velocity\[0\]"):
            sweep_contact(**kw)

    def test_velocity_nonfinite(self):
        for j, bad in enumerate((float("nan"), float("inf"),
                                 float("-inf"))):
            kw = self._good()
            kw["velocity"] = tuple(bad if k == j else 0.0
                                   for k in range(3))
            with self.assertRaisesRegex(InputError,
                                        rf"velocity\[{j}\]"):
                sweep_contact(**kw)

    def test_time_validation(self):
        kw = self._good()
        kw["t_end"] = 0.0
        with self.assertRaisesRegex(InputError, "t_start < t_end"):
            sweep_contact(**kw)
        kw = self._good()
        kw["t_start"], kw["t_end"] = 2.0, 1.0
        with self.assertRaisesRegex(InputError, "t_start < t_end"):
            sweep_contact(**kw)
        for name, bad in (("t_start", "x"), ("t_end", True),
                          ("t_start", float("nan")),
                          ("t_end", float("inf"))):
            kw = self._good()
            kw[name] = bad
            with self.assertRaises(InputError):
                sweep_contact(**kw)

    def test_tol_validation(self):
        for bad in (0.0, 1.0, -1e-9, float("nan")):
            with self.assertRaises(InputError):
                sweep_contact(CUBE_A, CUBE_B, (1, 0, 0), 0, 1, tol=bad)

    def test_points_still_validated(self):
        kw = self._good()
        kw["A"] = []
        with self.assertRaisesRegex(InputError, "A: expected 1..64"):
            sweep_contact(**kw)
        kw = self._good()
        kw["B"] = [(0.0, float("inf"), 0.0)]
        with self.assertRaisesRegex(InputError, r"B\[0\]\[1\]"):
            sweep_contact(**kw)

    def test_no_partial_state_on_bad_input(self):
        # pure function: repeated valid call after a rejected one
        with self.assertRaises(InputError):
            sweep_contact(CUBE_A, CUBE_B, (1, 0), 0, 1)
        res = sweep_contact(CUBE_A, CUBE_B, (1.0, 0.0, 0.0), 0, 6)
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertAlmostEqual(res.interval[0], 2.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
