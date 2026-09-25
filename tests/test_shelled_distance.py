"""Tests for ellipsoidal Minkowski shells (distance_shelled).

Analytic special cases are checked against closed-form geometry computed
here independently (point-sphere, point-ellipsoid on a principal axis,
rotated principal axis, capsule of a degenerate segment core).

Global optimality is NOT asserted from sampling: an independent direction
sample only checks (a) the shell support offset against independently
recomputed geometry and sampled ellipsoid points, (b) feasibility of the
returned witnesses, and (c) the reported distance bracket.  The analytic
cases are the only exact-distance checks.
"""

import math
import random
import unittest

from convex_distance import (
    InputError,
    STATUS_SEPARATED,
    STATUS_INTERSECTING,
    STATUS_NOT_CONVERGED,
    distance,
    distance_shelled,
    shell_support_offset,
)

DOT_EPS = 1e-8


# ---------------------------------------------------------------------------
# independent geometry (never imports the module's internals)
# ---------------------------------------------------------------------------

def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def mat_vec(M, x):
    return (M[0][0] * x[0] + M[0][1] * x[1] + M[0][2] * x[2],
            M[1][0] * x[0] + M[1][1] * x[1] + M[1][2] * x[2],
            M[2][0] * x[0] + M[2][1] * x[1] + M[2][2] * x[2])


def transpose(M):
    return ((M[0][0], M[1][0], M[2][0]),
            (M[0][1], M[1][1], M[2][1]),
            (M[0][2], M[1][2], M[2][2]))


def solve3(M, b):
    """Gaussian elimination, partial pivoting; raises if singular."""
    m = [list(M[i]) + [b[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) <= 1e-14:
            raise ValueError("singular")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= f * m[col][c]
    x = [0.0, 0.0, 0.0]
    for r in range(2, -1, -1):
        x[r] = (m[r][3] - sum(m[r][c] * x[c] for c in range(r + 1, 3))) \
            / m[r][r]
    return tuple(x)


def ref_offset(L, n):
    """L L^T n / ||L^T n||, independently."""
    Lt = transpose(L)
    y = mat_vec(Lt, n)
    ylen = math.sqrt(dot(y, y))
    return tuple(c / ylen for c in mat_vec(L, y))


def rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return ((c, -s, 0.0),
            (s, c, 0.0),
            (0.0, 0.0, 1.0))


def mat_mul(A, B):
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
        for i in range(3))


def sample_dirs(k, seed):
    """Deterministic near-uniform unit directions (fibonacci lattice +
    jittered extras), so tests do not depend on the module's own choices."""
    rng = random.Random(seed)
    dirs = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(k):
        z = 1.0 - 2.0 * (i + 0.5) / k
        r = math.sqrt(max(0.0, 1.0 - z * z))
        ang = golden * i
        dirs.append((math.cos(ang) * r, math.sin(ang) * r, z))
    for _ in range(k // 2):
        v = tuple(rng.uniform(-1, 1) for _ in range(3))
        n = math.sqrt(dot(v, v))
        dirs.append(tuple(c / n for c in v))
    return dirs


# ---------------------------------------------------------------------------
# spherical special case
# ---------------------------------------------------------------------------

class TestSphericalShell(unittest.TestCase):
    def test_two_points_with_radii(self):
        # point cores, spherical shells rA, rB: distance D - rA - rB
        for (D, rA, rB) in ((5.0, 0.5, 1.0), (10.0, 2.0, 0.25),
                            (3.0, 0.4, 1.5)):
            kw = {}
            if rA:
                kw["LA"] = ((rA, 0, 0), (0, rA, 0), (0, 0, rA))
            if rB:
                kw["LB"] = ((rB, 0, 0), (0, rB, 0), (0, 0, rB))
            res = distance_shelled(
                [(0.0, 0.0, 0.0)], [(D, 0.0, 0.0)], **kw)
            self.assertEqual(res.status, STATUS_SEPARATED)
            self.assertAlmostEqual(res.distance, D - rA - rB, delta=1e-8)
            self.assertAlmostEqual(res.witness_a[0], rA, delta=1e-8)
            self.assertAlmostEqual(res.witness_b[0], D - rB, delta=1e-8)
            self.assertAlmostEqual(res.offset_a[0], rA, delta=1e-8)
            self.assertAlmostEqual(res.offset_b[0], -rB, delta=1e-8)
            # core witnesses are the original input points
            self.assertAlmostEqual(dot(res.core_a, res.core_a), 0.0)
            self.assertAlmostEqual(res.core_b[0], D, delta=1e-10)

    def test_one_sided_shell_point_core(self):
        # no shell on A (LA omitted), radius shell on B only
        r = 1.5
        res = distance_shelled(
            [(0.0, 0.0, 0.0)], [(3.0, 0.0, 0.0)],
            LB=((r, 0, 0), (0, r, 0), (0, 0, r)))
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 3.0 - r, delta=1e-8)
        self.assertFalse(res.shell_a)
        self.assertTrue(res.shell_b)

    def test_point_inside_sphere_is_intersecting(self):
        L = ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))
        res = distance_shelled([(0.0, 0.0, 0.0)], [(0.3, 0.2, 0.1)],
                               LA=L)
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertEqual(res.lower_bound, 0.0)
        self.assertEqual(res.upper_bound, 0.0)
        self.assertIsNone(res.separating_axis)

    def test_shell_matches_shrunk_gap_in_3d(self):
        # triangle core with unit spherical shell vs a far point:
        # distance = point-to-triangle distance - 1
        tri = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
        p = [(0.5, 0.5, 4.0)]
        plain = distance(tri, p)
        L = ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))
        res = distance_shelled(tri, p, LA=L)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, plain.distance - 1.0,
                               delta=1e-8)

    def test_one_sided_shell_equivalent_to_both_half_radii(self):
        r = 0.8
        Lr = ((r, 0, 0), (0, r, 0), (0, 0, r))
        Lh = ((r / 2, 0, 0), (0, r / 2, 0), (0, 0, r / 2))
        A = [(0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (0.0, -1.0, 1.0)]
        B = [(4.0, -1.0, -1.0), (5.0, 1.0, 0.0)]
        r1 = distance_shelled(A, B, LA=Lr)
        r2 = distance_shelled(A, B, LA=Lh, LB=Lh)
        self.assertEqual(r1.status, STATUS_SEPARATED)
        self.assertAlmostEqual(r1.distance, r2.distance, delta=1e-8)


# ---------------------------------------------------------------------------
# anisotropic ellipsoids on principal / rotated axes (analytic)
# ---------------------------------------------------------------------------

class TestEllipsoidAnalytic(unittest.TestCase):
    def test_point_vs_axis_aligned_ellipsoid_principal_axes(self):
        a, b, c = 1.5, 0.7, 2.25
        L = ((a, 0, 0), (0, b, 0), (0, 0, c))
        # external points pushed out along each principal axis
        for axis, radius, gap in ((0, a, 1.3), (1, b, 0.9), (2, c, 2.0)):
            p = [tuple((radius + gap) if k == axis else 0.0
                       for k in range(3))]
            res = distance_shelled([(0.0, 0.0, 0.0)], p, LA=L)
            self.assertEqual(res.status, STATUS_SEPARATED,
                             msg=f"axis {axis}")
            self.assertAlmostEqual(res.distance, gap, delta=1e-8,
                                   msg=f"axis {axis}")
            # contact point of the ellipsoid is exactly +radius on axis
            contact = [0.0, 0.0, 0]
            contact[axis] = radius
            for k in range(3):
                self.assertAlmostEqual(res.witness_a[k], contact[k],
                                       delta=1e-8)
            # offsets actually on the ellipsoid boundary
            self.assertAlmostEqual(
                dot(solve3(L, res.offset_a), solve3(L, res.offset_a)),
                1.0, delta=1e-9)

    def test_point_inside_ellipsoid_intersecting(self):
        L = ((1.5, 0, 0), (0, 0.7, 0), (0, 0, 2.25))
        # strictly inside: (x/a)^2 + (y/b)^2 + (z/c)^2 < 1
        res = distance_shelled([(0.0, 0.0, 0.0)], [(0.5, 0.2, -0.3)], LA=L)
        self.assertEqual(res.status, STATUS_INTERSECTING)

    def test_two_ellipsoids_on_shared_axis(self):
        # centers distance D apart on x; gap = D - aA - aB
        D = 6.0
        LA = ((1.5, 0, 0), (0, 3.0, 0), (0, 0, 0.5))
        LB = ((0.8, 0, 0), (0, 0.2, 0), (0, 0, 4.0))
        res = distance_shelled([(0.0, 0.0, 0.0)], [(D, 0.0, 0.0)],
                               LA=LA, LB=LB)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, D - 1.5 - 0.8, delta=1e-8)
        self.assertAlmostEqual(res.witness_a[0], 1.5, delta=1e-8)
        self.assertAlmostEqual(res.witness_b[0], D - 0.8, delta=1e-8)

    def test_rotated_ellipsoid_principal_contact(self):
        # Rotate diag(a,b,c): contact in world frame is R (a,0,0), and a
        # point placed at R (a + gap, 0, 0) is exactly ``gap`` away.
        theta = 0.7
        R = rot_z(theta)
        a, b, cc = 2.0, 0.6, 1.1
        D = ((a, 0, 0), (0, b, 0), (0, 0, cc))
        L = mat_mul(R, D)
        gap = 1.25
        p = [mat_vec(R, (a + gap, 0.0, 0.0))]
        res = distance_shelled([(0.0, 0.0, 0.0)], p, LA=L)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, gap, delta=1e-8)
        contact_world = mat_vec(R, (a, 0.0, 0.0))
        for k in range(3):
            self.assertAlmostEqual(res.witness_a[k], contact_world[k],
                                   delta=1e-8)

    def test_rotated_ellipsoid_off_axis_point_inside(self):
        # World point R (0.5 a, 0.5 b, 0) is inside the rotated ellipsoid.
        R = rot_z(1.1)
        D = ((2.0, 0, 0), (0, 0.5, 0), (0, 0, 1.0))
        L = mat_mul(R, D)
        p = [mat_vec(R, (1.0, 0.25, 0.0))]
        res = distance_shelled([(0.0, 0.0, 0.0)], p, LA=L)
        self.assertEqual(res.status, STATUS_INTERSECTING)

    def test_anisotropic_shell_not_collapsed_to_sphere(self):
        # The crucial "no max-radius sphere" check: centers gap D on x,
        # x semi-axis 0.4 but y semi-axis 3.0.  A max-radius-sphere
        # implementation would subtract 3.0; the correct gap is D - 0.4.
        L = ((0.4, 0, 0), (0, 3.0, 0), (0, 0, 2.0))
        D = 5.0
        res = distance_shelled([(0.0, 0.0, 0.0)], [(D, 0.0, 0.0)], LA=L)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, D - 0.4, delta=1e-8)
        self.assertGreater(abs(res.distance - (D - 3.0)), 1.0)


# ---------------------------------------------------------------------------
# degenerate segment core + shell (capsule geometry)
# ---------------------------------------------------------------------------

class TestSegmentCoreShell(unittest.TestCase):
    def test_segment_with_spherical_shell_capsule(self):
        # segment [-c, c] on x plus radius-r sphere = an r-capsule.
        c, r = 2.0, 0.5
        L = ((r, 0, 0), (0, r, 0), (0, 0, r))
        seg = [(-c, 0.0, 0.0), (c, 0.0, 0.0)]

        # point above the middle: straight-line gap h - r
        p1 = [(0.0, 1.75, 0.0)]
        res1 = distance_shelled(seg, p1, LA=L)
        self.assertEqual(res1.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res1.distance, 1.25, delta=1e-8)
        # The shelled witness on the capsule is exactly (0, r, 0).  With a
        # face (cylindrical-side) contact the core witness may slide along
        # the segment under roundoff; what is exact is the shelled point
        # and the feasibility/splitting identity.
        self.assertAlmostEqual(res1.witness_a[0], 0.0, delta=1e-8)
        self.assertAlmostEqual(res1.witness_a[1], r, delta=1e-8)
        self.assertTrue(-c - 1e-7 <= res1.core_a[0] <= c + 1e-7)
        self.assertAlmostEqual(res1.core_a[1], 0.0, delta=1e-9)
        self.assertAlmostEqual(
            res1.offset_a[1],
            r - res1.core_a[1], delta=1e-9)
        for k in range(3):
            self.assertAlmostEqual(
                res1.witness_a[k], res1.core_a[k] + res1.offset_a[k],
                delta=1e-12)

        # point beyond the cap: gap along x = (c + t) - c - r = t - r
        p2 = [(3.25, 0.0, 0.0)]
        res2 = distance_shelled(seg, p2, LA=L)
        self.assertEqual(res2.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res2.distance, 0.75, delta=1e-8)
        self.assertAlmostEqual(res2.core_a[0], c, delta=1e-8)
        self.assertAlmostEqual(res2.offset_a[0], r, delta=1e-8)

    def test_coincident_endpoints_point_core_with_shell(self):
        r = 1.0
        L = ((r, 0, 0), (0, r, 0), (0, 0, r))
        seg = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        res = distance_shelled(seg, [(3.0, 4.0, 0.0)], LA=L)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 5.0 - r, delta=1e-8)

    def test_segment_ellipsoidal_shell_along_slab_direction(self):
        # segment on x [-1,1], shell diag(0.3, 1.0, 1.0).  Point straight
        # above the middle: analytic gap h - 1.0.
        L = ((0.3, 0, 0), (0, 1.0, 0), (0, 0, 1.0))
        seg = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        res = distance_shelled(seg, [(0.0, 3.5, 0.0)], LA=L)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 2.5, delta=1e-8)


# ---------------------------------------------------------------------------
# independent sampling: support geometry, feasibility, bracket
# (explicitly NOT a global-optimality proof)
# ---------------------------------------------------------------------------

class TestSampledSupportAndBracket(unittest.TestCase):
    def setUp(self):
        rng = random.Random(909)
        self.A = [(rng.uniform(-2, 2), rng.uniform(-2, 2),
                   rng.uniform(-2, 2)) for _ in range(8)]
        self.B = [(rng.uniform(-2, 2) + 4.0, rng.uniform(-1, 1),
                   rng.uniform(-1, 1)) for _ in range(6)]
        self.LA = mat_mul(rot_z(0.6),
                          ((0.9, 0, 0), (0, 0.35, 0), (0, 0, 0.6)))
        self.LB = ((0.25, 0.1, 0.0),
                   (0.0, 0.4, 0.2),
                   (0.1, 0.0, 0.5))
        self.res = distance_shelled(self.A, self.B,
                                    LA=self.LA, LB=self.LB)
        self.assertEqual(self.res.status, STATUS_SEPARATED)

    def test_offset_formula_on_sampled_directions(self):
        LAt, LBt = transpose(self.LA), transpose(self.LB)
        for n in sample_dirs(120, 111):
            # body A shell offset in +n, body B shell offset in -n
            oa = shell_support_offset(self.LA, n)
            ob = shell_support_offset(self.LB, tuple(-c for c in n))
            self._check_offset(self.LA, LAt, n, oa)
            self._check_offset(self.LB, LBt, tuple(-c for c in n), ob)

    def _check_offset(self, L, Lt, n, off):
        # identity with the independent closed form
        ref = ref_offset(L, n)
        for k in range(3):
            self.assertAlmostEqual(off[k], ref[k], delta=1e-9)
        # off.n == ||L^T n||
        self.assertAlmostEqual(dot(off, n),
                               math.sqrt(dot(mat_vec(Lt, n),
                                             mat_vec(Lt, n))),
                               delta=1e-9)
        # boundary point: ||L^{-1} off|| == 1
        z = solve3(L, off)
        self.assertAlmostEqual(math.sqrt(dot(z, z)), 1.0, delta=1e-9)
        # support dominance vs 60 independently sampled ellipsoid points
        for u in sample_dirs(60, abs(hash((n[0], n[1]))) % 2**31):
            q = mat_vec(L, u)  # point on L S2
            self.assertLessEqual(dot(q, n), dot(off, n) + 1e-9)

    def test_returned_witnesses_are_feasible(self):
        res = self.res
        # shell offsets are feasible points of each ellipsoid (|L^-1 o|<=1)
        za = solve3(self.LA, res.offset_a)
        self.assertLessEqual(math.sqrt(dot(za, za)), 1.0 + 1e-8)
        zb = solve3(self.LB, res.offset_b)
        self.assertLessEqual(math.sqrt(dot(zb, zb)), 1.0 + 1e-8)
        # core witnesses inside each core hull: sampled support directions
        # must not cut them off (sampling check, not an exhaustive proof)
        for d in sample_dirs(150, 222):
            self.assertLessEqual(dot(res.core_a, d),
                                 max(dot(a, d) for a in self.A) + 1e-8)
            self.assertGreaterEqual(dot(res.core_b, d),
                                    min(dot(b, d) for b in self.B) - 1e-8)
        # witness == core + offset exactly
        for k in range(3):
            self.assertAlmostEqual(
                res.witness_a[k], res.core_a[k] + res.offset_a[k],
                delta=1e-12)
            self.assertAlmostEqual(
                res.witness_b[k], res.core_b[k] + res.offset_b[k],
                delta=1e-12)

    def test_bracket_and_certificate(self):
        res = self.res
        witness_dist = math.sqrt(
            dot(tuple(res.witness_a[k] - res.witness_b[k]
                      for k in range(3)),
                tuple(res.witness_a[k] - res.witness_b[k]
                      for k in range(3))))
        # feasible upper bound
        self.assertAlmostEqual(witness_dist, res.upper_bound, delta=1e-9)
        self.assertLessEqual(res.lower_bound, res.distance + 1e-12)
        self.assertLessEqual(res.distance, res.upper_bound + 1e-12)
        self.assertLess(res.lower_bound, res.upper_bound + 1e-12)
        # certificate along the reported axis n (B -> A): recompute the
        # *near* supporting faces independently -- the lower support of
        # A' in direction n and the upper support of B' -- and check that
        # their gap brackets the reported distance.
        n = res.separating_axis
        self.assertAlmostEqual(dot(n, n), 1.0, delta=1e-9)
        near_a = min(dot(a, n) for a in self.A) \
            - dot(shell_support_offset(self.LA, n), n)
        near_b = max(dot(b, n) for b in self.B) \
            + dot(shell_support_offset(self.LB, n), n)
        self.assertAlmostEqual(near_a - near_b, res.distance, delta=1e-7)
        self.assertGreaterEqual(near_a - near_b + 1e-9, res.lower_bound)


class TestNotConvergedKeepsBracket(unittest.TestCase):
    def test_budget_exhausted_reports_interval_not_intersection(self):
        L = ((0.5, 0.1, 0.0), (0.0, 0.8, 0.2), (0.1, 0.0, 0.3))
        A = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        B = [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0), (5.0, 6.0, 5.0)]
        res = distance_shelled(A, B, LA=L, LB=L, max_iter=1)
        self.assertEqual(res.status, STATUS_NOT_CONVERGED)
        self.assertNotEqual(res.status, STATUS_INTERSECTING)
        self.assertGreater(res.residual, 0.0)
        # a usable bracket survives: 0 <= lower <= distance <= upper
        self.assertGreaterEqual(res.lower_bound, 0.0)
        self.assertLessEqual(res.lower_bound, res.distance + 1e-12)
        self.assertLessEqual(res.distance, res.upper_bound + 1e-12)
        self.assertLess(res.lower_bound, res.upper_bound)
        # axis remains the unverified best direction
        self.assertIsNotNone(res.separating_axis)

    def test_converged_bracket_is_tight(self):
        L = ((1.0, 0, 0), (0, 2.0, 0), (0, 0, 0.5))
        res = distance_shelled([(0.0, 0.0, 0.0)], [(6.0, 0.0, 0.0)], LA=L)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.lower_bound, res.upper_bound,
                               delta=res.upper_bound * 1e-8)


# ---------------------------------------------------------------------------
# shell flags, no-shell equivalence, validation
# ---------------------------------------------------------------------------

class TestInterfaceAndValidation(unittest.TestCase):
    def test_shell_flags(self):
        L = ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))
        r0 = distance_shelled([(0, 0, 0)], [(3, 0, 0)])
        self.assertFalse(r0.shell_a)
        self.assertFalse(r0.shell_b)
        r1 = distance_shelled([(0, 0, 0)], [(3, 0, 0)], LA=L)
        self.assertTrue(r1.shell_a)
        self.assertFalse(r1.shell_b)
        r2 = distance_shelled([(0, 0, 0)], [(3, 0, 0)], LA=L, LB=L)
        self.assertTrue(r2.shell_a)
        self.assertTrue(r2.shell_b)

    def test_no_shell_matches_legacy_distance(self):
        rng = random.Random(707)
        A = [tuple(rng.uniform(-2, 2) for _ in range(3)) for _ in range(7)]
        B = [tuple(rng.uniform(-2, 2) + 3.0 for _ in range(3))
             for _ in range(5)]
        r1 = distance(A, B)
        r2 = distance_shelled(A, B)
        self.assertEqual(r1.status, r2.status)
        self.assertAlmostEqual(r1.distance, r2.distance, delta=1e-12)
        for k in range(3):
            self.assertAlmostEqual(r1.witness_a[k], r2.witness_a[k],
                                   delta=1e-12)
            self.assertAlmostEqual(r1.witness_b[k], r2.witness_b[k],
                                   delta=1e-12)
            self.assertAlmostEqual(r2.core_a[k], r2.witness_a[k],
                                   delta=1e-12)
            self.assertEqual(r2.offset_a[k], 0.0)
            self.assertEqual(r2.offset_b[k], 0.0)

    def test_zero_direction_rejected(self):
        L = ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))
        with self.assertRaisesRegex(InputError, "zero support direction"):
            shell_support_offset(L, (0.0, 0.0, 0.0))
        with self.assertRaisesRegex(InputError, r"n\[1\]"):
            shell_support_offset(L, (1.0, float("nan"), 0.0))

    def test_singular_matrices_rejected(self):
        singular = [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((1.0, 2.0, 0.0), (2.0, 4.0, 0.0), (0.0, 0.0, 1.0)),
            ((1.0, 0.0, 0.0), (0.0, 1e-14, 0.0), (0.0, 0.0, 1.0)),
        ]
        for L in singular:
            with self.assertRaisesRegex(InputError, r"LA: matrix is singular"):
                distance_shelled([(0, 0, 0)], [(3, 0, 0)], LA=L)
        with self.assertRaisesRegex(InputError, r"LB: matrix is singular"):
            distance_shelled([(0, 0, 0)], [(3, 0, 0)],
                             LB=((1.0, 0.0, 0.0),) * 3)

    def test_matrix_shape_and_value_errors(self):
        with self.assertRaisesRegex(InputError, r"LA: expected a 3x3"):
            distance_shelled([(0, 0, 0)], [(3, 0, 0)],
                             LA=((1.0, 0.0, 0.0),) * 2)
        with self.assertRaisesRegex(InputError, r"LA\[2\]: expected exactly 3"):
            distance_shelled([(0, 0, 0)], [(3, 0, 0)],
                             LA=((1, 0, 0), (1, 0, 0), (1, 0)))
        with self.assertRaisesRegex(InputError, r"LA\[1\]\[0\]"):
            distance_shelled([(0, 0, 0)], [(3, 0, 0)],
                             LA=((1, 0, 0), ("x", 0, 0), (0, 0, 1)))
        with self.assertRaisesRegex(InputError, r"LB\[0\]\[2\]"):
            distance_shelled([(0, 0, 0)], [(3, 0, 0)],
                             LB=((1, 0, float("inf")), (0, 1, 0),
                                 (0, 0, 1)))
        with self.assertRaisesRegex(InputError, "LA"):
            distance_shelled([(0, 0, 0)], [(3, 0, 0)], LA=True)

    def test_legacy_interface_unchanged(self):
        # old contract still returns the old result type with no shell
        res = distance([(0, 0, 0)], [(3, 4, 0)])
        self.assertAlmostEqual(res.distance, 5.0, delta=1e-12)
        self.assertFalse(hasattr(res, "core_a"))


if __name__ == "__main__":
    unittest.main()
