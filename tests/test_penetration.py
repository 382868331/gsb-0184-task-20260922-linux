"""Tests for penetration(): EPA depth bounds, direction and contact points.

Independent references (SAT axis enumeration over face normals and
edge-edge cross products, Caratheodory barycentric membership) are
implemented here from scratch and never call the module under test.
"""

import itertools
import math
import unittest

from convex_distance import (
    InputError,
    distance,
    penetration,
    STATUS_SEPARATED,
    STATUS_INTERSECTING,
    STATUS_NOT_CONVERGED,
    STATUS_PENETRATING,
    STATUS_TOUCHING,
)


# ---------------------------------------------------------------------------
# independent references
# ---------------------------------------------------------------------------

def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _sub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def _norm(u):
    return math.sqrt(_dot(u, u))


def _rot(axis, angle):
    """Rodrigues rotation matrix (3x3, row-major tuples)."""
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return (
        (x * x * C + c, x * y * C - z * s, x * z * C + y * s),
        (y * x * C + z * s, y * y * C + c, y * z * C - x * s),
        (z * x * C - y * s, z * y * C + x * s, z * z * C + c),
    )


def _apply(mat, p):
    return tuple(sum(mat[i][j] * p[j] for j in range(3)) for i in range(3))


def candidate_axes(verts_a, verts_b):
    """SAT candidate axes: face normals of both hulls + edge-edge crosses.

    Computed purely from the vertex clouds (all triples / all pairs), so
    degenerate (flat) hulls still contribute their plane normal.
    """
    axes = []
    for verts in (verts_a, verts_b):
        for i, j, k in itertools.combinations(range(len(verts)), 3):
            n = _cross(_sub(verts[j], verts[i]), _sub(verts[k], verts[i]))
            if _norm(n) > 1e-12:
                axes.append(n)
    for i, j in itertools.combinations(range(len(verts_a)), 2):
        ea = _sub(verts_a[j], verts_a[i])
        for k, l in itertools.combinations(range(len(verts_b)), 2):
            n = _cross(ea, _sub(verts_b[l], verts_b[k]))
            if _norm(n) > 1e-12:
                axes.append(n)
    out = []
    for n in axes:
        nn = _norm(n)
        out.append((n[0] / nn, n[1] / nn, n[2] / nn))
    return out


def sat_intervals(verts_a, verts_b, axis):
    lo_a = min(_dot(p, axis) for p in verts_a)
    hi_a = max(_dot(p, axis) for p in verts_a)
    lo_b = min(_dot(p, axis) for p in verts_b)
    hi_b = max(_dot(p, axis) for p in verts_b)
    return lo_a, hi_a, lo_b, hi_b


def sat_depth(verts_a, verts_b):
    """Independent min-translation depth via SAT axis enumeration.

    Returns None when a separating axis exists (hulls do not overlap),
    else the minimum over all candidate axes of the one-axis escape
    distance min(hiA - loB, hiB - loA) (the exact amount one hull must
    move along that axis to clear the other; correct also for nested or
    degenerate intervals, where the naive overlap length is wrong).
    """
    depth = math.inf
    for ax in candidate_axes(verts_a, verts_b):
        lo_a, hi_a, lo_b, hi_b = sat_intervals(verts_a, verts_b, ax)
        if hi_a < lo_b - 1e-9 or hi_b < lo_a - 1e-9:
            return None
        depth = min(depth, hi_a - lo_b, hi_b - lo_a)
    return depth


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


def in_convex_hull(p, verts, tol=1e-7):
    """Independent Caratheodory membership test (subsets of <= 4 points)."""
    for k in range(1, 5):
        for idx in itertools.combinations(range(len(verts)), k):
            # barycentric coords of p's projection onto aff(subset):
            # solve the KKT system [G 1; 1^T 0] [lam; mu] = [g; 1]
            vs = [verts[i] for i in idx]
            mat = [[_dot(a, b) for b in vs] + [1.0] for a in vs]
            mat.append([1.0] * k + [0.0])
            rhs = [_dot(p, v) for v in vs] + [1.0]
            lam = _solve_linear(mat, rhs)
            if lam is None:
                continue
            lam = lam[:k]
            proj = tuple(sum(l * v[c] for l, v in zip(lam, vs))
                         for c in range(3))
            if _norm(_sub(p, proj)) <= tol and \
                    all(l >= -tol for l in lam):
                return True
    return False


def cube(lo, hi):
    return [(float(x), float(y), float(z))
            for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

UNIT = cube((0, 0, 0), (1, 1, 1))

# Oblique penetration: unit cube rotated 40 deg about (1, 2, 3) and shifted
# so the overlap direction is far from axis-aligned.
ROT = _rot((1.0, 2.0, 3.0), math.radians(40.0))
OBLIQUE_B = [tuple(_apply(ROT, p)[i] + off for i, off in
                   enumerate((0.55, 0.35, 0.45)))
             for p in cube((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))]

TETRA = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
OCTA = [(0.9, 0.2, 0.2), (-0.5, 0.2, 0.2), (0.2, 0.9, 0.2),
        (0.2, -0.5, 0.2), (0.2, 0.2, 0.9), (0.2, 0.2, -0.5)]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestPenetrationDepth(unittest.TestCase):
    def test_box_axis_aligned(self):
        b = cube((0.7, 0.0, 0.0), (1.7, 1.0, 1.0))
        res = penetration(UNIT, b)
        self.assertEqual(res.status, STATUS_PENETRATING)
        self.assertAlmostEqual(res.depth, 0.3, delta=1e-9)
        self.assertLessEqual(res.depth_lower, 0.3 + 1e-9)
        self.assertGreaterEqual(res.depth_upper, 0.3 - 1e-9)
        self.assertAlmostEqual(abs(_dot(res.direction, (1, 0, 0))), 1.0,
                               delta=1e-9)

    def test_box_oblique_matches_sat(self):
        res = penetration(UNIT, OBLIQUE_B)
        self.assertEqual(res.status, STATUS_PENETRATING)
        ref = sat_depth(UNIT, OBLIQUE_B)
        self.assertIsNotNone(ref)
        self.assertGreater(ref, 0.0)
        # verifiable bounds bracket the independent SAT reference
        self.assertLessEqual(res.depth_lower, ref + 1e-7)
        self.assertGreaterEqual(res.depth_upper, ref - 1e-7)
        self.assertAlmostEqual(res.depth, ref, delta=1e-6)

    def test_small_polytopes_match_sat(self):
        res = penetration(TETRA, OCTA)
        self.assertEqual(res.status, STATUS_PENETRATING)
        ref = sat_depth(TETRA, OCTA)
        self.assertIsNotNone(ref)
        self.assertGreater(ref, 0.0)
        self.assertLessEqual(res.depth_lower, ref + 1e-7)
        self.assertGreaterEqual(res.depth_upper, ref - 1e-7)
        self.assertAlmostEqual(res.depth, ref, delta=1e-6)

    def test_translation_by_upper_separates(self):
        for a, b in ((UNIT, OBLIQUE_B), (TETRA, OCTA),
                     (UNIT, cube((0.7, 0.2, 0.1), (1.7, 1.2, 1.1)))):
            res = penetration(a, b)
            self.assertEqual(res.status, STATUS_PENETRATING)
            n = res.direction
            t = res.depth_upper + 1e-6
            moved = [tuple(p[i] + t * n[i] for i in range(3)) for p in b]
            self.assertIsNone(sat_depth(a, moved),
                              "translating B by depth_upper*direction "
                              "must separate (independent SAT check)")

    def test_translation_below_lower_still_intersects(self):
        res = penetration(UNIT, OBLIQUE_B)
        n = res.direction
        t = res.depth_lower * 0.5
        moved = [tuple(p[i] + t * n[i] for i in range(3)) for p in OBLIQUE_B]
        self.assertIsNotNone(sat_depth(UNIT, moved),
                             "translating B by half the lower bound must "
                             "not separate (independent SAT check)")

    def test_witnesses_are_convex_combinations(self):
        for a, b in ((UNIT, OBLIQUE_B), (TETRA, OCTA)):
            res = penetration(a, b)
            self.assertEqual(res.status, STATUS_PENETRATING)
            self.assertTrue(in_convex_hull(res.witness_a, a),
                            "witness_a must lie in conv(A)")
            self.assertTrue(in_convex_hull(res.witness_b, b),
                            "witness_b must lie in conv(B)")
            gap = _sub(res.witness_a, res.witness_b)
            self.assertAlmostEqual(_norm(gap), res.depth_lower, delta=1e-7)
            self.assertAlmostEqual(
                _dot(gap, res.direction), res.depth_lower, delta=1e-7,
                msg="witness gap must align with direction")

    def test_duplicate_points_ignored(self):
        dup = UNIT + UNIT  # every vertex twice
        b = cube((0.7, 0.0, 0.0), (1.7, 1.0, 1.0))
        res = penetration(dup, b)
        self.assertEqual(res.status, STATUS_PENETRATING)
        self.assertAlmostEqual(res.depth, 0.3, delta=1e-9)

    def test_flat_hull_penetrating(self):
        # zero-volume square slicing through the unit cube
        square = [(0.2, 0.2, -0.5), (0.8, 0.2, -0.5),
                  (0.8, 0.8, -0.5), (0.2, 0.8, -0.5)]
        square = [(x, y, z + 0.75) for x, y, z in square]  # plane z = 0.25
        res = penetration(square, UNIT)
        self.assertEqual(res.status, STATUS_PENETRATING)
        ref = sat_depth(square, UNIT)
        self.assertIsNotNone(ref)
        self.assertLessEqual(res.depth_lower, ref + 1e-7)
        self.assertGreaterEqual(res.depth_upper, ref - 1e-7)

    def test_budget_exhausted_keeps_bounds(self):
        res = penetration(UNIT, OBLIQUE_B, max_iter=4)
        self.assertEqual(res.status, STATUS_NOT_CONVERGED)
        ref = sat_depth(UNIT, OBLIQUE_B)
        self.assertIsNotNone(res.direction)
        self.assertLessEqual(res.depth_lower, ref + 1e-7)
        self.assertGreaterEqual(res.depth_upper, ref - 1e-7)
        self.assertTrue(math.isfinite(res.depth_upper))


class TestTouchingAndSeparated(unittest.TestCase):
    def test_face_contact(self):
        b = cube((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        res = penetration(UNIT, b)
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertEqual(res.depth, 0.0)
        self.assertEqual(res.depth_lower, 0.0)
        self.assertEqual(res.depth_upper, 0.0)
        self.assertIsNone(res.direction)
        self.assertAlmostEqual(_norm(_sub(res.witness_a, res.witness_b)),
                               0.0, delta=1e-9)

    def test_edge_contact(self):
        b = cube((1.0, 1.0, 0.0), (2.0, 2.0, 1.0))
        res = penetration(UNIT, b)
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertEqual(res.depth, 0.0)

    def test_point_contact(self):
        b = cube((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
        res = penetration(UNIT, b)
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertEqual(res.depth, 0.0)

    def test_coplanar_flat_contact(self):
        # flat square lying exactly on the cube's top face z = 1
        square = [(0.2, 0.2, 1.0), (0.8, 0.2, 1.0),
                  (0.8, 0.8, 1.0), (0.2, 0.8, 1.0)]
        res = penetration(square, UNIT)
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertEqual(res.depth, 0.0)
        self.assertIsNone(res.direction)

    def test_separated(self):
        b = cube((2.0, 0.0, 0.0), (3.0, 1.0, 1.0))
        res = penetration(UNIT, b)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertEqual(res.depth, 0.0)
        self.assertIsNone(res.direction)


class TestPenetrationContract(unittest.TestCase):
    def test_input_errors(self):
        with self.assertRaisesRegex(InputError, "non-finite"):
            penetration([(0.0, 0.0, float("nan"))], UNIT)
        with self.assertRaises(InputError):
            penetration(UNIT, UNIT, tol=True)
        with self.assertRaises(InputError):
            penetration(UNIT, UNIT, max_iter=True)
        with self.assertRaises(InputError):
            penetration(UNIT, UNIT, max_iter=0)

    def test_distance_interface_unchanged(self):
        # overlapping hulls still report intersecting with ~zero distance
        res = distance(UNIT, cube((0.7, 0.0, 0.0), (1.7, 1.0, 1.0)))
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertLessEqual(res.distance, 1e-9)
        self.assertIsNone(res.separating_axis)
        # separated hulls still report a verified axis
        res = distance(UNIT, cube((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)))
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 1.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
