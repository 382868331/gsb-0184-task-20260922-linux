"""Tests for convex_distance.penetration (EPA minimum translation).

Independent references -- SAT enumeration over candidate separating axes
(face normals and edge-cross-edge directions derived from the raw vertex
sets) and brute-force translation checks via the public distance() --
are implemented here from scratch and never call the EPA code.
"""

import math
import unittest

from convex_distance import (
    InputError,
    distance,
    penetration,
    STATUS_SEPARATED,
    STATUS_PENETRATING,
    STATUS_TOUCHING,
    STATUS_NOT_CONVERGED,
)


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


def _translate(P, t):
    return [(p[0] + t[0], p[1] + t[1], p[2] + t[2]) for p in P]


def ref_mtv_depth(A, B):
    """Independent penetration depth by SAT axis enumeration.

    For intersecting convex polytopes the minimum translation length is
    attained on a face normal of either hull or on an edge-cross-edge
    direction.  Without face topology we use all pairwise vertex
    differences and their cross products, a superset of those axes.
    Along a unit axis n the needed translation is
    min(max_A - min_B, max_B - min_A) of the projected intervals (this
    handles one interval containing the other).  Returns
    (depth, unit_axis) or None when an axis separates the hulls.
    """
    diffs_a = [_sub(p, q) for i, p in enumerate(A) for q in A[i + 1:]]
    diffs_b = [_sub(p, q) for i, p in enumerate(B) for q in B[i + 1:]]
    axes = []
    for u in diffs_a + diffs_b:
        for v in diffs_a + diffs_b:
            n = _cross(u, v)
            if _norm(n) > 1e-9:
                axes.append(n)
    best = None
    for n in axes:
        ln = _norm(n)
        n = (n[0] / ln, n[1] / ln, n[2] / ln)
        min_a = min(_dot(a, n) for a in A)
        max_a = max(_dot(a, n) for a in A)
        min_b = min(_dot(b, n) for b in B)
        max_b = max(_dot(b, n) for b in B)
        if min(max_a, max_b) - max(min_a, min_b) < -1e-9:
            return None  # separated
        overlap = min(max_a - min_b, max_b - min_a)
        if best is None or overlap < best[0]:
            best = (overlap, n)
    return best


def solve_convex_weights(P, x):
    """Barycentric weights of x w.r.t. P by least squares on a subset of
    at most 4 affinely independent points (Caratheodory); returns the
    weights vector over all of P, or None when x is outside conv(P).
    Independent of the module under test.
    """
    import itertools

    def solve(mat, rhs):
        n = len(mat)
        m = [row[:] + [rhs[i]] for i, row in enumerate(mat)]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(m[r][col]))
            if abs(m[piv][col]) <= 1e-12:
                return None
            m[col], m[piv] = m[piv], m[col]
            inv = 1.0 / m[col][col]
            for r in range(col + 1, n):
                f = m[r][col] * inv
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
        out = [0.0] * n
        for r in range(n - 1, -1, -1):
            out[r] = (m[r][n] - sum(m[r][c] * out[c]
                                    for c in range(r + 1, n))) / m[r][r]
        return out

    for k in range(1, min(4, len(P)) + 1):
        for idx in itertools.combinations(range(len(P)), k):
            basis = [P[i] for i in idx]
            # normal equations for min ||sum w_i b_i - x|| s.t. sum w = 1
            n = k + 1
            mat = [[0.0] * n for _ in range(n)]
            rhs = [0.0] * n
            for i in range(k):
                for j in range(k):
                    mat[i][j] = _dot(basis[i], basis[j])
                mat[i][k] = 1.0
                mat[k][i] = 1.0
                rhs[i] = _dot(basis[i], x)
            rhs[k] = 1.0
            w = solve(mat, rhs)
            if w is None:
                continue
            w = w[:k]
            if min(w) < -1e-7:
                continue
            px = tuple(sum(wi * b[c] for wi, b in zip(w, basis))
                       for c in range(3))
            if _norm(_sub(px, x)) <= 1e-6 * max(1.0, _norm(x)):
                full = [0.0] * len(P)
                for i, wi in zip(idx, w):
                    full[i] = wi
                return full
    return None


def box(cx, cy, cz, hx, hy, hz):
    return [(cx + sx * hx, cy + sy * hy, cz + sz * hz)
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)]


def rotz(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])


def roty(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    return (c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2])


class TestPenetrationBoxes(unittest.TestCase):
    def test_axis_aligned_box_depth(self):
        a = box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = box(1.5, 0.0, 0.0, 1.0, 1.0, 1.0)
        res = penetration(a, b)
        self.assertEqual(res.status, STATUS_PENETRATING)
        self.assertAlmostEqual(res.depth_lower, 0.5, places=9)
        self.assertAlmostEqual(res.depth_upper, 0.5, places=9)
        self.assertLessEqual(res.residual, 1e-9)
        # direction must be the unit x axis (either sense is a valid MTV
        # only if consistent with the contact points; here +x)
        self.assertAlmostEqual(_norm(res.direction), 1.0, places=12)
        self.assertAlmostEqual(abs(res.direction[0]), 1.0, places=9)

    def test_oblique_box_matches_sat_reference(self):
        # Rotated box vs axis box: the answer is NOT axis-aligned-box
        # trivial; cross-check against the independent SAT enumeration.
        a = box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = [rotz(p, math.pi / 4.0) for p in box(1.2, 0.3, 0.0, 1.0, 1.0, 0.5)]
        ref = ref_mtv_depth(a, b)
        self.assertIsNotNone(ref)
        ref_depth, _ = ref
        res = penetration(a, b)
        self.assertEqual(res.status, STATUS_PENETRATING)
        self.assertLessEqual(res.depth_lower, ref_depth + 1e-9)
        self.assertGreaterEqual(res.depth_upper, ref_depth - 1e-9)
        self.assertAlmostEqual(res.depth_lower, ref_depth, places=7)
        self.assertAlmostEqual(res.depth_upper, ref_depth, places=7)

    def test_tilted_small_polytopes_match_sat_reference(self):
        # Two small irregular convex polytopes, oblique penetration.
        a = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0),
             (0.0, 0.0, 2.0), (1.0, 1.0, 1.0)]
        b = [roty(rotz(p, 0.5), 0.3) for p in
             box(0.0, 0.0, 0.0, 0.6, 0.4, 0.7)]
        b = _translate(b, (0.8, 0.5, 0.4))
        ref = ref_mtv_depth(a, b)
        self.assertIsNotNone(ref)
        ref_depth, _ = ref
        res = penetration(a, b)
        self.assertEqual(res.status, STATUS_PENETRATING)
        self.assertLessEqual(res.depth_lower, ref_depth + 1e-9)
        self.assertGreaterEqual(res.depth_upper, ref_depth - 1e-9)
        self.assertAlmostEqual(0.5 * (res.depth_lower + res.depth_upper),
                               ref_depth, places=6)

    def test_translation_by_depth_separates(self):
        a = box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = [rotz(p, math.pi / 4.0) for p in box(1.2, 0.3, 0.0, 1.0, 1.0, 0.5)]
        res = penetration(a, b)
        self.assertEqual(res.status, STATUS_PENETRATING)
        n = res.direction
        # Pushing B by slightly more than the upper bound must separate.
        t = tuple(1.01 * res.depth_upper * c for c in n)
        self.assertEqual(distance(a, _translate(b, t)).status,
                         STATUS_SEPARATED)
        # Pushing B by slightly less than the lower bound must not.
        t = tuple(0.99 * res.depth_lower * c for c in n)
        self.assertNotEqual(distance(a, _translate(b, t)).status,
                            STATUS_SEPARATED)

    def test_contact_points_are_convex_combinations(self):
        a = box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = [rotz(p, math.pi / 4.0) for p in box(1.2, 0.3, 0.0, 1.0, 1.0, 0.5)]
        res = penetration(a, b)
        self.assertEqual(res.status, STATUS_PENETRATING)
        # Each contact point lies in its own hull (weights found by an
        # independent solver).
        wa = solve_convex_weights(a, res.contact_a)
        wb = solve_convex_weights(b, res.contact_b)
        self.assertIsNotNone(wa)
        self.assertIsNotNone(wb)
        self.assertAlmostEqual(sum(wa), 1.0, places=9)
        self.assertAlmostEqual(sum(wb), 1.0, places=9)
        # contact_a - contact_b == depth * direction (the MTV itself).
        mtv = _sub(res.contact_a, res.contact_b)
        depth = _norm(mtv)
        self.assertGreaterEqual(res.depth_upper + 1e-9, depth)
        self.assertLessEqual(res.depth_lower - 1e-9, depth)
        for c in range(3):
            self.assertAlmostEqual(mtv[c], depth * res.direction[c],
                                   places=7)


class TestPenetrationStates(unittest.TestCase):
    def test_face_contact_is_explicit_zero_depth_touching(self):
        a = box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = box(2.0, 0.0, 0.0, 1.0, 1.0, 1.0)  # coplanar face contact
        res = penetration(a, b)
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertEqual(res.depth_lower, 0.0)
        self.assertEqual(res.depth_upper, 0.0)
        self.assertIsNone(res.direction)

    def test_edge_contact_touching(self):
        a = box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = box(2.0, 2.0, 0.0, 1.0, 1.0, 1.0)  # edge-to-edge contact
        res = penetration(a, b)
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertEqual(res.depth_lower, 0.0)
        self.assertEqual(res.depth_upper, 0.0)

    def test_separated(self):
        a = box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = box(4.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        res = penetration(a, b)
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertEqual(res.depth_lower, 0.0)
        self.assertEqual(res.depth_upper, 0.0)
        self.assertIsNone(res.direction)

    def test_budget_exhaustion_keeps_verifiable_bounds(self):
        a = box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = [rotz(p, math.pi / 4.0) for p in box(1.2, 0.3, 0.0, 1.0, 1.0, 0.5)]
        full = penetration(a, b)
        starved = penetration(a, b, max_iter=4)
        self.assertEqual(starved.status, STATUS_NOT_CONVERGED)
        # Bounds remain ordered and bracket the converged answer.
        self.assertLessEqual(starved.depth_lower, starved.depth_upper)
        self.assertLessEqual(starved.depth_lower,
                             full.depth_upper + 1e-9)
        self.assertGreaterEqual(starved.depth_upper,
                                full.depth_lower - 1e-9)
        # The reported direction is the best known one, still a unit
        # vector, but the status flags it as not converged.
        self.assertAlmostEqual(_norm(starved.direction), 1.0, places=9)

    def test_duplicate_points_do_not_break_epa(self):
        a = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
        b = box(0.4, 0.2, 0.2, 0.3, 0.3, 0.3)
        res = penetration(a, b)
        self.assertEqual(res.status, STATUS_PENETRATING)
        self.assertLessEqual(res.depth_lower, res.depth_upper)

    def test_distance_interface_unchanged(self):
        a = box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = box(1.5, 0.0, 0.0, 1.0, 1.0, 1.0)
        res = distance(a, b)
        self.assertEqual(res.status, "intersecting")
        self.assertIsNone(res.separating_axis)

    def test_validation_shared_with_distance(self):
        a = box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        with self.assertRaises(InputError):
            penetration(a, [(0.0, 0.0, float("nan"))])
        with self.assertRaises(InputError):
            penetration(a, a, max_iter=0)
        with self.assertRaises(InputError):
            penetration(a, a, tol=True)


if __name__ == "__main__":
    unittest.main()
