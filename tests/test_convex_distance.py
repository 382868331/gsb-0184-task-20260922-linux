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
    STATUS_SEPARATED,
    STATUS_INTERSECTING,
    STATUS_NOT_CONVERGED,
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


class TestNearContactRegression(unittest.TestCase):
    """Regression tests for two core defects exposed by the sweep work:
    near-touching pairs (~1e-9 * scale) were misreported as intersecting
    through the barycentric feasibility slack, and a support point
    repeating after being dropped from the simplex sent GJK into a
    cycle that burned the whole iteration budget (false not_converged).
    """

    def test_near_touch_is_separated_not_intersecting(self):
        # point 2e-9 above the top face of the unit cube: well above the
        # zero threshold (~1.7e-11) but inside the old barycentric slack
        cube = box_corners((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        res = distance(cube, [(0.5, 1.0 + 2e-9, 0.0)])
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 2e-9, delta=1e-13)

    def test_support_cycle_terminates(self):
        # this exact configuration previously exhausted max_iter=128
        cube = box_corners((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        res = distance(cube, [(0.999999, 1.000001, 0.0)])
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 1.000001e-6, delta=2e-12)
        self.assertLess(res.iterations, 128)


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


if __name__ == "__main__":
    unittest.main()
