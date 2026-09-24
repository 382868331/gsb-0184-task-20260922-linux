"""Tests for the translational sweep (first contact / contact interval).

Expected values come from hand-computed analytic cases and from an
independent per-axis analytic reference for axis-aligned boxes
(ref_aabb_sweep below); neither calls the module under test.
"""

import math
import random
import unittest

from convex_distance import (
    InputError,
    distance,
    sweep,
    STATUS_SEPARATED,
    STATUS_INTERSECTING,
    STATUS_MISS,
    STATUS_TOUCHING,
)


def box_corners(lo, hi):
    return [(x, y, z)
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])]


def ref_aabb_sweep(lo1, hi1, lo2, hi2, v, t0, t1):
    """Independent analytic contact interval for two axis-aligned boxes.

    Box 1 static, box 2 moving with velocity v.  Per axis the overlap
    condition is linear in t, so the contact set is the intersection of
    three t-intervals.  Returns (t_enter, t_exit) clipped to [t0, t1],
    or None when there is no contact in the window.
    """
    enter = -math.inf
    exit_ = math.inf
    for k in range(3):
        lo = lo2[k]
        hi = hi2[k]
        vk = v[k]
        # overlap on axis k: lo + t*vk <= hi1[k] and lo1[k] <= hi + t*vk
        if vk > 0.0:
            e = (lo1[k] - hi) / vk
            x = (hi1[k] - lo) / vk
        elif vk < 0.0:
            e = (hi1[k] - lo) / vk
            x = (lo1[k] - hi) / vk
        else:
            if lo > hi1[k] or lo1[k] > hi:
                return None
            e, x = -math.inf, math.inf
        enter = max(enter, e)
        exit_ = min(exit_, x)
    if enter > exit_:
        return None
    en = max(enter, t0)
    ex = min(exit_, t1)
    if en > ex:
        return None
    return (en, ex)


def ref_aabb_distance(lo1, hi1, lo2, hi2):
    s = 0.0
    for k in range(3):
        gap = max(0.0, lo1[k] - hi2[k], lo2[k] - hi1[k])
        s += gap * gap
    return math.sqrt(s)


class TestHandComputed(unittest.TestCase):
    def test_point_point_touching(self):
        # point B starts at (5,0,0), moves at (-1,0,0); meets A at t = 5
        res = sweep([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                    (-1.0, 0.0, 0.0), (0.0, 10.0))
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertAlmostEqual(res.t_enter, 5.0, delta=1e-6)
        self.assertAlmostEqual(res.t_exit, 5.0, delta=1e-6)
        self.assertFalse(res.initially_overlapping)
        self.assertFalse(res.stationary)
        self.assertEqual(res.contact_interval, (res.t_enter, res.t_exit))
        for k in range(3):
            self.assertAlmostEqual(res.witness_a[k], 0.0, delta=1e-6)
            self.assertAlmostEqual(res.witness_b[k], 0.0, delta=1e-6)

    def test_cube_cube_interval(self):
        # A = [-1,1]^3; B = [4,6]x[-1,1]^2 moving at (-1,0,0).
        # faces meet at t = 3, B passes through, separates again at t = 7.
        a = box_corners((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        b = box_corners((4.0, -1.0, -1.0), (6.0, 1.0, 1.0))
        res = sweep(a, b, (-1.0, 0.0, 0.0), (0.0, 10.0))
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertAlmostEqual(res.t_enter, 3.0, delta=1e-6)
        self.assertAlmostEqual(res.t_exit, 7.0, delta=1e-6)
        self.assertFalse(res.initially_overlapping)

    def test_initial_overlap_then_separation(self):
        # B overlaps A at t = 0 and slides out; last contact at t = 0.5.
        a = box_corners((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        b = box_corners((0.5, -1.0, -1.0), (2.5, 1.0, 1.0))
        res = sweep(a, b, (1.0, 0.0, 0.0), (0.0, 10.0))
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertTrue(res.initially_overlapping)
        self.assertAlmostEqual(res.t_enter, 0.0, delta=1e-9)
        self.assertAlmostEqual(res.t_exit, 0.5, delta=1e-6)

    def test_stationary_overlapping(self):
        a = box_corners((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        b = box_corners((0.5, -1.0, -1.0), (2.5, 1.0, 1.0))
        res = sweep(a, b, (0.0, 0.0, 0.0), (2.0, 9.0))
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertTrue(res.stationary)
        self.assertTrue(res.initially_overlapping)
        self.assertEqual(res.t_enter, 2.0)
        self.assertEqual(res.t_exit, 9.0)

    def test_stationary_separated(self):
        a = box_corners((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        b = box_corners((4.0, -1.0, -1.0), (6.0, 1.0, 1.0))
        res = sweep(a, b, (0.0, 0.0, 0.0), (0.0, 10.0))
        self.assertEqual(res.status, STATUS_MISS)
        self.assertTrue(res.stationary)
        self.assertIsNone(res.t_enter)
        self.assertIsNone(res.contact_interval)
        self.assertAlmostEqual(res.min_distance, 3.0, delta=1e-8)

    def test_miss_perpendicular_motion(self):
        # moving parallel to the gap: distance stays 3 the whole window
        a = box_corners((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        b = box_corners((4.0, -1.0, -1.0), (6.0, 1.0, 1.0))
        res = sweep(a, b, (0.0, 1.0, 0.0), (0.0, 10.0))
        self.assertEqual(res.status, STATUS_MISS)
        self.assertAlmostEqual(res.min_distance, 3.0, delta=1e-8)

    def test_miss_with_closest_approach(self):
        # point passes the origin at distance 3 at t = 5
        res = sweep([(0.0, 0.0, 0.0)], [(5.0, 3.0, 0.0)],
                    (-1.0, 0.0, 0.0), (0.0, 10.0))
        self.assertEqual(res.status, STATUS_MISS)
        self.assertAlmostEqual(res.t_closest, 5.0, delta=1e-6)
        self.assertAlmostEqual(res.min_distance, 3.0, delta=1e-8)
        # the separating axis at closest approach certifies the gap
        # (convention: n points from B towards A, i.e. (0,-1,0) here)
        n = res.separating_axis
        self.assertAlmostEqual(n[0] * n[0] + n[1] * n[1] + n[2] * n[2],
                               1.0, delta=1e-9)
        self.assertAlmostEqual(n[1], -1.0, delta=1e-6)

    def test_touching_graze(self):
        # point (t, 2 - t, 0) grazes the cube corner edge at (1, 1, 0)
        # exactly at t = 1; distance is |1 - t| to the left and
        # sqrt(2)*(t - 1) to the right, so the contact is a single instant.
        cube = box_corners((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        res = sweep(cube, [(0.0, 2.0, 0.0)], (1.0, -1.0, 0.0), (0.0, 4.0))
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertAlmostEqual(res.t_enter, 1.0, delta=1e-6)
        self.assertAlmostEqual(res.t_exit, 1.0, delta=1e-6)
        for k, want in enumerate((1.0, 1.0, 0.0)):
            self.assertAlmostEqual(res.witness_a[k], want, delta=1e-5)
            self.assertAlmostEqual(res.witness_b[k], want, delta=1e-5)

    def test_contact_at_window_end(self):
        # contact exactly at t1 counts as (touching) contact in the window
        res = sweep([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                    (-1.0, 0.0, 0.0), (0.0, 5.0))
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertAlmostEqual(res.t_enter, 5.0, delta=1e-6)

    def test_contact_at_window_start(self):
        res = sweep([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                    (-1.0, 0.0, 0.0), (5.0, 10.0))
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertTrue(res.initially_overlapping)
        self.assertAlmostEqual(res.t_enter, 5.0, delta=1e-9)

    def test_contact_just_outside_window_is_miss(self):
        # contact at t = 5, window ends at 4
        res = sweep([(0.0, 0.0, 0.0)], [(5.0, 0.0, 0.0)],
                    (-1.0, 0.0, 0.0), (0.0, 4.0))
        self.assertEqual(res.status, STATUS_MISS)
        self.assertAlmostEqual(res.min_distance, 1.0, delta=1e-8)

    def test_overlap_throughout_window(self):
        a = box_corners((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        b = box_corners((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
        res = sweep(a, b, (0.1, 0.0, 0.0), (0.0, 1.0))
        self.assertEqual(res.status, STATUS_INTERSECTING)
        self.assertEqual(res.t_enter, 0.0)
        self.assertEqual(res.t_exit, 1.0)


class TestAabbReference(unittest.TestCase):
    def test_random_boxes_against_analytic(self):
        rng = random.Random(909)
        checked = 0
        attempts = 0
        while checked < 40 and attempts < 4000:
            attempts += 1
            lo1 = tuple(rng.uniform(-3, 3) for _ in range(3))
            hi1 = tuple(lo1[k] + rng.uniform(0.2, 2.0) for k in range(3))
            lo2 = tuple(rng.uniform(-3, 3) for _ in range(3))
            hi2 = tuple(lo2[k] + rng.uniform(0.2, 2.0) for k in range(3))
            v = tuple(rng.uniform(-2, 2) for _ in range(3))
            if math.sqrt(sum(c * c for c in v)) < 0.2:
                continue
            t0, t1 = 0.0, rng.uniform(2.0, 6.0)
            ref = ref_aabb_sweep(lo1, hi1, lo2, hi2, v, t0, t1)
            # skip cases whose interval is too close to the documented
            # touching/intersecting resolution to classify unambiguously
            if ref is not None and 1e-8 < ref[1] - ref[0] < 1e-5:
                continue
            res = sweep(box_corners(lo1, hi1), box_corners(lo2, hi2),
                        v, (t0, t1))
            if ref is None:
                self.assertEqual(res.status, STATUS_MISS,
                                 msg=f"case {attempts}: expected miss")
                # min distance over the window must not exceed the
                # endpoint distances (convexity sanity check)
                d0 = ref_aabb_distance(
                    lo1, hi1, lo2, hi2)
                d1 = ref_aabb_distance(
                    lo1, hi1,
                    tuple(lo2[k] + t1 * v[k] for k in range(3)),
                    tuple(hi2[k] + t1 * v[k] for k in range(3)))
                self.assertLessEqual(res.min_distance,
                                     min(d0, d1) + 1e-6)
            else:
                en, ex = ref
                want_status = (STATUS_TOUCHING if ex - en <= 1e-8
                               else STATUS_INTERSECTING)
                self.assertEqual(res.status, want_status,
                                 msg=f"case {attempts}: ref={ref}")
                self.assertAlmostEqual(res.t_enter, en, delta=1e-5,
                                       msg=f"case {attempts}")
                self.assertAlmostEqual(res.t_exit, ex, delta=1e-5,
                                       msg=f"case {attempts}")
                self.assertEqual(res.initially_overlapping, en == t0)
                # witnesses coincide at contact
                gap = math.dist(res.witness_a, res.witness_b)
                self.assertLessEqual(gap, 1e-5)
            checked += 1
        self.assertGreaterEqual(checked, 40)

    def test_miss_certificate_verifies(self):
        # for a miss, the separating axis at t_closest must separate
        # A from B shifted by t_closest * v, vertex by vertex
        rng = random.Random(919)
        checked = 0
        while checked < 10:
            lo1 = tuple(rng.uniform(-2, 2) for _ in range(3))
            hi1 = tuple(lo1[k] + rng.uniform(0.5, 1.5) for k in range(3))
            lo2 = tuple(rng.uniform(-2, 2) for _ in range(3))
            hi2 = tuple(lo2[k] + rng.uniform(0.5, 1.5) for k in range(3))
            v = tuple(rng.uniform(-1.5, 1.5) for _ in range(3))
            t1 = rng.uniform(1.0, 4.0)
            if ref_aabb_sweep(lo1, hi1, lo2, hi2, v, 0.0, t1) is not None:
                continue
            A = box_corners(lo1, hi1)
            B = box_corners(lo2, hi2)
            res = sweep(A, B, v, (0.0, t1))
            self.assertEqual(res.status, STATUS_MISS)
            n = res.separating_axis
            self.assertIsNotNone(n)
            self.assertAlmostEqual(sum(c * c for c in n), 1.0, delta=1e-9)
            wa_n = sum(res.witness_a[k] * n[k] for k in range(3))
            wb_n = sum(res.witness_b[k] * n[k] for k in range(3))
            for a in A:
                self.assertGreaterEqual(sum(a[k] * n[k] for k in range(3)),
                                        wa_n - 1e-6)
            for b in B:
                bs = tuple(b[k] + res.t_closest * v[k] for k in range(3))
                self.assertLessEqual(sum(bs[k] * n[k] for k in range(3)),
                                     wb_n + 1e-6)
            checked += 1


class TestSweepValidation(unittest.TestCase):
    def setUp(self):
        self.a = [(0.0, 0.0, 0.0)]
        self.b = [(1.0, 0.0, 0.0)]

    def test_velocity_shape_and_type(self):
        with self.assertRaisesRegex(InputError, "velocity: expected exactly 3"):
            sweep(self.a, self.b, (1.0, 0.0), (0.0, 1.0))
        with self.assertRaisesRegex(InputError, r"velocity\[1\]"):
            sweep(self.a, self.b, (1.0, "x", 0.0), (0.0, 1.0))
        with self.assertRaisesRegex(InputError, r"velocity\[0\]"):
            sweep(self.a, self.b, (True, 0.0, 0.0), (0.0, 1.0))
        with self.assertRaisesRegex(InputError, "velocity: expected"):
            sweep(self.a, self.b, "fast", (0.0, 1.0))

    def test_velocity_non_finite(self):
        with self.assertRaisesRegex(InputError, r"velocity\[2\]"):
            sweep(self.a, self.b, (1.0, 0.0, float("nan")), (0.0, 1.0))
        with self.assertRaisesRegex(InputError, r"velocity\[0\]"):
            sweep(self.a, self.b, (float("inf"), 0.0, 0.0), (0.0, 1.0))

    def test_window_errors(self):
        with self.assertRaisesRegex(InputError, "t_window: expected t0 < t1"):
            sweep(self.a, self.b, (1.0, 0.0, 0.0), (2.0, 1.0))
        with self.assertRaisesRegex(InputError, "t_window: expected t0 < t1"):
            sweep(self.a, self.b, (1.0, 0.0, 0.0), (1.0, 1.0))
        with self.assertRaisesRegex(InputError, r"t_window\[1\]"):
            sweep(self.a, self.b, (1.0, 0.0, 0.0), (0.0, float("nan")))
        with self.assertRaisesRegex(InputError, r"t_window\[0\]"):
            sweep(self.a, self.b, (1.0, 0.0, 0.0), (False, 1.0))
        with self.assertRaisesRegex(InputError,
                                    "t_window: expected exactly 2"):
            sweep(self.a, self.b, (1.0, 0.0, 0.0), (0.0, 1.0, 2.0))

    def test_point_and_option_validation_still_applies(self):
        with self.assertRaisesRegex(InputError, r"A\[0\]\[2\]"):
            sweep([(0.0, 0.0, float("nan"))], self.b,
                  (1.0, 0.0, 0.0), (0.0, 1.0))
        with self.assertRaisesRegex(InputError, "B: expected 1..64"):
            sweep(self.a, [], (1.0, 0.0, 0.0), (0.0, 1.0))
        with self.assertRaisesRegex(InputError, "tol"):
            sweep(self.a, self.b, (1.0, 0.0, 0.0), (0.0, 1.0), tol=0.0)
        with self.assertRaisesRegex(InputError, "max_iter"):
            sweep(self.a, self.b, (1.0, 0.0, 0.0), (0.0, 1.0), max_iter=0)

    def test_invalid_input_leaves_no_partial_state(self):
        # a failing call must not affect subsequent valid calls
        with self.assertRaises(InputError):
            sweep(self.a, self.b, (1.0, 0.0, 0.0), (1.0, 0.0))
        res = sweep(self.a, self.b, (-1.0, 0.0, 0.0), (0.0, 2.0))
        self.assertEqual(res.status, STATUS_TOUCHING)
        self.assertAlmostEqual(res.t_enter, 1.0, delta=1e-6)


class TestSweepDeterminism(unittest.TestCase):
    def test_repeated_calls_identical(self):
        a = box_corners((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        b = box_corners((4.0, -0.5, -0.5), (5.0, 0.5, 0.5))
        r1 = sweep(a, b, (-0.7, 0.1, -0.2), (0.0, 8.0))
        r2 = sweep(a, b, (-0.7, 0.1, -0.2), (0.0, 8.0))
        self.assertEqual(r1, r2)

    def test_old_entry_point_unaffected(self):
        # distance() keeps its original contract after the refactor
        res = distance([(0.0, 0.0, 0.0)], [(3.0, 4.0, 0.0)])
        self.assertEqual(res.status, STATUS_SEPARATED)
        self.assertAlmostEqual(res.distance, 5.0, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
