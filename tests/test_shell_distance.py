"""Tests for the ellipsoid-shell interface distance_shell.

Independent references (analytic sphere distance, point-ellipsoid
distance via Lagrange-multiplier bisection, point-segment distance,
support-function evaluation over sampled directions) are implemented
here from scratch and never call the module under test.  The sampled
-direction checks only verify support bounds and witness feasibility;
they are not used as a proof of global optimality.
"""

import math
import random
import unittest

from convex_distance import (
    InputError,
    distance,
    distance_shell,
    STATUS_SEPARATED,
    STATUS_INTERSECTING,
    STATUS_NOT_CONVERGED,
)


# ---------------------------------------------------------------------------
# independent helpers (no calls into convex_distance)
# ---------------------------------------------------------------------------

def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _sub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def _norm(u):
    return math.sqrt(_dot(u, u))


def _mat_T_vec(L, v):
    """L^T v for row-major 3x3 L."""
    return (L[0][0] * v[0] + L[1][0] * v[1] + L[2][0] * v[2],
            L[0][1] * v[0] + L[1][1] * v[1] + L[2][1] * v[2],
            L[0][2] * v[0] + L[1][2] * v[1] + L[2][2] * v[2])


def _mat_vec(L, v):
    return (L[0][0] * v[0] + L[0][1] * v[1] + L[0][2] * v[2],
            L[1][0] * v[0] + L[1][1] * v[1] + L[1][2] * v[2],
            L[2][0] * v[0] + L[2][1] * v[1] + L[2][2] * v[2])


def _mat_inv(L):
    """3x3 inverse via cofactors (independent of the module)."""
    a, b, c = L[0]
    d, e, f = L[1]
    g, h, i = L[2]
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    det = a * A + b * B + c * C
    inv = ((A, -(b * i - c * h), b * f - c * e),
           (B, a * i - c * g, -(a * f - c * d)),
           (C, -(a * h - b * g), a * e - b * d))
    return tuple(tuple(x / det for x in row) for row in inv)


def ref_support_dilated(verts, L, u):
    """Support value of conv(verts) (+) L*ball in direction u:
    max_i v_i.u + ||L^T u||  (None shell -> just the max)."""
    h = max(_dot(p, u) for p in verts)
    if L is not None:
        h += _norm(_mat_T_vec(L, u))
    return h


def ref_point_ellipsoid_diag(p, axes):
    """Distance from point p to the solid ellipsoid diag(axes)*ball
    centred at the origin, by bisection on the Lagrange multiplier.

    Minimise ||p - x||^2 s.t. sum((x_i/a_i)^2) <= 1.  For p outside,
    the KKT point is x_i = p_i a_i^2 / (a_i^2 + t) with t >= 0 chosen
    so the constraint is active; g(t) = sum((p_i a_i/(a_i^2+t))^2) is
    strictly decreasing in t, so bisection is exact.
    """
    def g(t):
        return sum((p[k] * axes[k] / (axes[k] ** 2 + t)) ** 2
                   for k in range(3))

    if g(0.0) <= 1.0:
        return 0.0  # p inside the ellipsoid
    lo, hi = 0.0, math.sqrt(sum((p[k] * axes[k]) ** 2 for k in range(3)))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    x = tuple(p[k] * axes[k] ** 2 / (axes[k] ** 2 + t) for k in range(3))
    return _norm(_sub(p, x))


def ref_point_segment(p, a, b):
    ab = _sub(b, a)
    denom = _dot(ab, ab)
    t = 0.0 if denom == 0.0 else max(0.0, min(1.0,
                                              _dot(_sub(p, a), ab) / denom))
    c = (a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2])
    return _norm(_sub(p, c))


def rotation_matrix(axis, angle):
    """Rodrigues' formula for a unit axis."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return ((c + x * x * C, x * y * C - z * s, x * z * C + y * s),
            (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
            (z * x * C - y * s, z * y * C + x * s, c + z * z * C))


def mat_mul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def diag(a, b, c):
    return ((a, 0.0, 0.0), (0.0, b, 0.0), (0.0, 0.0, c))


IDENT = diag(1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# analytic special cases
# ---------------------------------------------------------------------------

class TestSphericalShell(unittest.TestCase):
    """L = r*I must reproduce the analytic sphere distance exactly;
    using a maximal-radius sphere instead of L L^T n/||L^T n|| would
    pass here too, so the anisotropic cases below are the discriminating
    ones -- this class pins the spherical special case."""

    def test_point_point_spheres(self):
        rng = random.Random(11)
        for _ in range(20):
            ca = tuple(rng.uniform(-3, 3) for _ in range(3))
            cb = tuple(rng.uniform(-3, 3) for _ in range(3))
            ra, rb = rng.uniform(0.05, 1.5), rng.uniform(0.05, 1.5)
            La = diag(ra, ra, ra)
            Lb = diag(rb, rb, rb)
            gap = _norm(_sub(ca, cb)) - ra - rb
            res = distance_shell([ca], [cb], shell_a=La, shell_b=Lb)
            if gap > 1e-9:
                self.assertEqual(res.status, STATUS_SEPARATED)
                self.assertAlmostEqual(res.distance, gap, delta=1e-8)
                self.assertLessEqual(res.lower_bound, gap + 1e-9)
                self.assertGreaterEqual(res.upper_bound, gap - 1e-9)
                # witnesses sit on the two sphere surfaces
                self.assertAlmostEqual(_norm(_sub(res.witness_a, ca)), ra,
                                       delta=1e-9)
                self.assertAlmostEqual(_norm(_sub(res.witness_b, cb)), rb,
                                       delta=1e-9)
            else:
                self.assertEqual(res.status, STATUS_INTERSECTING)

    def test_overlapping_spheres_intersect(self):
        res = distance_shell([(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0)],
                             shell_a=IDENT, shell_b=IDENT)
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertLessEqual(res.distance, 1e-9)
        self.assertEqual(res.lower_bound, 0.0)
        self.assertIsNone(res.separating_axis)
        # zero-direction contract: offsets are exactly zero
        self.assertEqual(res.shell_offset_a, (0.0, 0.0, 0.0))
        self.assertEqual(res.shell_offset_b, (0.0, 0.0, 0.0))


class TestPointEllipsoid(unittest.TestCase):
    """Point core + anisotropic ellipsoid shell vs a bare point, checked
    against an independent Lagrange-bisection reference.  A maximal-
    radius sphere approximation would fail these."""

    def test_axis_aligned_ellipsoid(self):
        rng = random.Random(22)
        checked = 0
        while checked < 20:
            axes = (rng.uniform(0.3, 2.0), rng.uniform(0.3, 2.0),
                    rng.uniform(0.3, 2.0))
            c = tuple(rng.uniform(-2, 2) for _ in range(3))
            p = tuple(rng.uniform(-6, 6) for _ in range(3))
            expected = ref_point_ellipsoid_diag(_sub(p, c), axes)
            res = distance_shell([c], [p], shell_a=diag(*axes))
            if expected > 1e-8:
                self.assertEqual(res.status, STATUS_SEPARATED)
                self.assertAlmostEqual(res.distance, expected, delta=1e-7)
                self.assertAlmostEqual(res.upper_bound, expected, delta=1e-7)
                self.assertLessEqual(res.lower_bound, expected + 1e-9)
                checked += 1
            else:
                self.assertEqual(res.status, STATUS_INTERSECTING)
                checked += 1

    def test_ellipsoid_is_not_its_bounding_sphere(self):
        # Long thin ellipsoid along x; point on the y axis just outside
        # the minor radius.  Bounding-sphere (r = 4) would report
        # "intersecting"; the true ellipsoid distance is 0.25.
        L = diag(4.0, 0.5, 0.5)
        res = distance_shell([(0.0, 0.0, 0.0)], [(0.0, 0.75, 0.0)],
                             shell_a=L)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 0.25, delta=1e-9)
        self.assertAlmostEqual(res.witness_a[1], 0.5, delta=1e-9)

    def test_rotated_shell(self):
        rng = random.Random(33)
        for _ in range(15):
            axes = (rng.uniform(0.4, 2.5), rng.uniform(0.4, 2.5),
                    rng.uniform(0.4, 2.5))
            axis = tuple(rng.uniform(-1, 1) for _ in range(3))
            n = _norm(axis)
            axis = tuple(x / n for x in axis)
            angle = rng.uniform(-math.pi, math.pi)
            R = rotation_matrix(axis, angle)
            L = mat_mul(R, diag(*axes))
            c = tuple(rng.uniform(-2, 2) for _ in range(3))
            p = tuple(rng.uniform(-6, 6) for _ in range(3))
            # rotate the problem into the ellipsoid frame: R^T (p - c)
            pl = _mat_T_vec(R, _sub(p, c))
            expected = ref_point_ellipsoid_diag(pl, axes)
            res = distance_shell([c], [p], shell_a=L)
            if expected > 1e-8:
                self.assertEqual(res.status, STATUS_SEPARATED)
                self.assertAlmostEqual(res.distance, expected, delta=1e-7)
            else:
                self.assertEqual(res.status, STATUS_INTERSECTING)


class TestDegenerateCoreWithShell(unittest.TestCase):
    def test_segment_core_sphere_shell_vs_point_sphere(self):
        rng = random.Random(44)
        for _ in range(20):
            a = tuple(rng.uniform(-3, 3) for _ in range(3))
            b = tuple(rng.uniform(-3, 3) for _ in range(3))
            p = tuple(rng.uniform(-3, 3) for _ in range(3))
            ra, rb = rng.uniform(0.05, 1.0), rng.uniform(0.05, 1.0)
            core_gap = ref_point_segment(p, a, b)
            expected = core_gap - ra - rb
            res = distance_shell([a, b], [p],
                                 shell_a=diag(ra, ra, ra),
                                 shell_b=diag(rb, rb, rb))
            if expected > 1e-8:
                self.assertEqual(res.status, STATUS_SEPARATED)
                self.assertAlmostEqual(res.distance, expected, delta=1e-7)
                # core witness is on the segment
                self.assertAlmostEqual(
                    ref_point_segment(res.core_witness_a, a, b), 0.0,
                    delta=1e-7)
            else:
                self.assertEqual(res.status, STATUS_INTERSECTING)

    def test_degenerate_segment_core_ellipsoid_shell(self):
        # segment endpoints coincide -> point core; ellipsoid shell
        L = diag(2.0, 1.0, 0.5)
        res = distance_shell([(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
                             [(5.0, 1.0, 1.0)], shell_a=L)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 2.0, delta=1e-9)


# ---------------------------------------------------------------------------
# witness decomposition and certified bounds
# ---------------------------------------------------------------------------

class TestWitnessAndBounds(unittest.TestCase):
    def _random_case(self, rng):
        na = rng.randint(1, 6)
        nb = rng.randint(1, 6)
        A = [tuple(rng.uniform(-2, 2) for _ in range(3)) for _ in range(na)]
        B = [tuple(rng.uniform(-2, 2) + 3.0 for _ in range(3))
             for _ in range(nb)]
        La = mat_mul(rotation_matrix(
            (0.0, 0.0, 1.0), rng.uniform(-math.pi, math.pi)),
            diag(rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0),
                 rng.uniform(0.2, 1.0)))
        Lb = mat_mul(rotation_matrix(
            (0.0, 1.0, 0.0), rng.uniform(-math.pi, math.pi)),
            diag(rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0),
                 rng.uniform(0.2, 1.0)))
        return A, B, La, Lb

    def test_witness_splits_into_core_and_shell(self):
        rng = random.Random(55)
        for _ in range(15):
            A, B, La, Lb = self._random_case(rng)
            res = distance_shell(A, B, shell_a=La, shell_b=Lb)
            if res.status != STATUS_SEPARATED:
                continue
            # witness = core + offset, exactly as returned
            for k in range(3):
                self.assertAlmostEqual(
                    res.witness_a[k],
                    res.core_witness_a[k] + res.shell_offset_a[k],
                    delta=1e-12)
                self.assertAlmostEqual(
                    res.witness_b[k],
                    res.core_witness_b[k] + res.shell_offset_b[k],
                    delta=1e-12)
            # offsets lie exactly on their ellipsoid boundaries:
            # ||L^{-1} offset|| == 1 (independent inverse via cofactors)
            ua = _mat_vec(_mat_inv(La), res.shell_offset_a)
            ub = _mat_vec(_mat_inv(Lb), res.shell_offset_b)
            self.assertAlmostEqual(_norm(ua), 1.0, delta=1e-9)
            self.assertAlmostEqual(_norm(ub), 1.0, delta=1e-9)
            # core witnesses stay inside the core support bounds:
            # for every sampled direction u, core_witness.u <= max_i a_i.u
            for _ in range(40):
                u = tuple(rng.gauss(0, 1) for _ in range(3))
                u = tuple(x / _norm(u) for x in u)
                self.assertLessEqual(_dot(res.core_witness_a, u),
                                     max(_dot(a, u) for a in A) + 1e-7)
                self.assertLessEqual(_dot(res.core_witness_b, u),
                                     max(_dot(b, u) for b in B) + 1e-7)

    def test_certified_bounds_against_sampled_directions(self):
        rng = random.Random(66)
        checked = 0
        while checked < 15:
            A, B, La, Lb = self._random_case(rng)
            res = distance_shell(A, B, shell_a=La, shell_b=Lb)
            if res.status != STATUS_SEPARATED:
                continue
            # upper bound is exactly the returned witness gap
            self.assertAlmostEqual(
                res.upper_bound, _norm(_sub(res.witness_a, res.witness_b)),
                delta=1e-12)
            self.assertLessEqual(res.lower_bound, res.upper_bound + 1e-12)
            self.assertLessEqual(res.lower_bound, res.distance + 1e-9)
            # converged: the certified interval is tight
            self.assertLessEqual(res.upper_bound - res.lower_bound,
                                 1e-6 * max(1.0, res.distance))
            # independent support bound: for every sampled unit u the
            # projection gap max(0, min_A' a.u - max_B' b.u) is a valid
            # lower bound on the true distance, hence <= upper_bound.
            for _ in range(60):
                u = tuple(rng.gauss(0, 1) for _ in range(3))
                u = tuple(x / _norm(u) for x in u)
                proj = (min(_dot(a, u) for a in A)
                        - _norm(_mat_T_vec(La, u))
                        - max(_dot(b, u) for b in B)
                        - _norm(_mat_T_vec(Lb, u)))
                self.assertLessEqual(max(0.0, proj),
                                     res.upper_bound + 1e-9)
            # the returned axis itself certifies a gap close to distance
            n = res.separating_axis
            gap_n = (min(_dot(a, n) for a in A) - _norm(_mat_T_vec(La, n))
                     - max(_dot(b, n) for b in B)
                     - _norm(_mat_T_vec(Lb, n)))
            self.assertAlmostEqual(gap_n, res.distance, delta=1e-6)
            checked += 1

    def test_no_shell_matches_plain_distance(self):
        rng = random.Random(77)
        for _ in range(10):
            A = [tuple(rng.uniform(-2, 2) for _ in range(3))
                 for _ in range(rng.randint(1, 5))]
            B = [tuple(rng.uniform(-2, 2) + 3.0 for _ in range(3))
                 for _ in range(rng.randint(1, 5))]
            r1 = distance_shell(A, B)
            r2 = distance(A, B)
            self.assertEqual(r1.status, r2.status)
            self.assertAlmostEqual(r1.distance, r2.distance, delta=1e-12)
            for k in range(3):
                self.assertAlmostEqual(r1.witness_a[k], r2.witness_a[k],
                                       delta=1e-9)
                self.assertAlmostEqual(r1.shell_offset_a[k], 0.0, delta=0.0)


# ---------------------------------------------------------------------------
# budget, status and validation
# ---------------------------------------------------------------------------

class TestShellBudgetAndStatus(unittest.TestCase):
    def test_budget_exhausted_keeps_interval(self):
        A = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
             (0.0, 0.0, 1.0)]
        B = [(4.0, 4.0, 4.0), (5.0, 4.0, 4.0), (4.0, 5.0, 4.0),
             (4.0, 4.0, 5.0)]
        res = distance_shell(A, B, shell_a=diag(0.5, 0.5, 0.5),
                             shell_b=diag(0.5, 0.5, 0.5), max_iter=1)
        self.assertEqual(res.status, STATUS_NOT_CONVERGED)
        self.assertFalse(res.converged)
        # the interval is kept; no exact intersection is claimed
        self.assertNotEqual(res.status, STATUS_INTERSECTING)
        self.assertGreaterEqual(res.lower_bound, 0.0)
        self.assertLessEqual(res.lower_bound, res.upper_bound)
        self.assertGreater(res.residual, 0.0)


class TestShellValidation(unittest.TestCase):
    def test_singular_matrix_rejected(self):
        bad = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaisesRegex(InputError, "shell_a: singular"):
            distance_shell([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                           shell_a=bad)
        rank1 = ((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (0.0, 1.0, 0.0))
        with self.assertRaisesRegex(InputError, "shell_b: singular"):
            distance_shell([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                           shell_b=rank1)

    def test_zero_matrix_rejected(self):
        with self.assertRaisesRegex(InputError, "singular"):
            distance_shell([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                           shell_a=((0.0, 0.0, 0.0),) * 3)

    def test_nonfinite_and_bool_entries_rejected(self):
        with self.assertRaisesRegex(InputError, r"shell_a\[1\]\[2\]"):
            distance_shell([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                           shell_a=((1.0, 0.0, 0.0),
                                    (0.0, 1.0, float("nan")),
                                    (0.0, 0.0, 1.0)))
        with self.assertRaisesRegex(InputError, r"shell_b\[0\]\[0\]"):
            distance_shell([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                           shell_b=((True, 0.0, 0.0),
                                    (0.0, 1.0, 0.0),
                                    (0.0, 0.0, 1.0)))
        with self.assertRaisesRegex(InputError, r"shell_a\[2\]\[1\]"):
            distance_shell([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                           shell_a=((1.0, 0.0, 0.0),
                                    (0.0, 1.0, 0.0),
                                    (0.0, float("inf"), 1.0)))

    def test_shape_errors(self):
        with self.assertRaisesRegex(InputError, "shell_a: expected exactly 3 rows"):
            distance_shell([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                           shell_a=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        with self.assertRaisesRegex(InputError, r"shell_a\[0\]: expected exactly 3 columns"):
            distance_shell([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                           shell_a=((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
        with self.assertRaisesRegex(InputError, "shell_a: expected a 3x3 matrix"):
            distance_shell([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                           shell_a="identity")

    def test_fraction_entries_accepted(self):
        from fractions import Fraction as F
        L = ((F(3, 2), 0, 0), (0, F(1), 0), (0, 0, F(1, 4)))
        res = distance_shell([(0, 0, 0)], [(F(5), 0, 0)], shell_a=L)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 3.5, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
