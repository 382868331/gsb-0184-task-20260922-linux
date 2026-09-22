"""输入契约与错误语义测试。"""

import math
import unittest

from convex_distance import (
    ConvexBody,
    InvalidBodyError,
    InvalidParameterError,
    MAX_ITERATIONS,
    MAX_POINTS,
    box,
    distance,
)


class TestPointValidation(unittest.TestCase):
    def test_reject_nan(self):
        with self.assertRaises(InvalidBodyError) as cm:
            distance([(0.0, float("nan"), 0.0)], [(1.0, 0.0, 0.0)])
        self.assertIn("A[0][1]", str(cm.exception))

    def test_reject_infinity(self):
        for val in (float("inf"), float("-inf")):
            with self.subTest(val=val):
                with self.assertRaises(InvalidBodyError) as cm:
                    ConvexBody([(val, 0.0, 0.0)], name="B")
                self.assertIn("B[0][0]", str(cm.exception))

    def test_reject_bool_coordinate(self):
        # bool 是 int 的子类型，但契约要求显式拒绝
        with self.assertRaises(TypeError) as cm:
            ConvexBody([(True, 0.0, 0.0)])
        self.assertIn("bool", str(cm.exception))

    def test_reject_non_numeric(self):
        with self.assertRaises(TypeError) as cm:
            ConvexBody([("1.0", 0.0, 0.0)], name="A")
        self.assertIn("A[0][0]", str(cm.exception))

    def test_reject_wrong_dimension(self):
        with self.assertRaises(InvalidBodyError) as cm:
            ConvexBody([(0.0, 0.0)], name="A")
        self.assertIn("A[0]", str(cm.exception))

    def test_reject_non_sequence_point(self):
        with self.assertRaises(TypeError):
            ConvexBody([3])

    def test_reject_empty(self):
        with self.assertRaises(InvalidBodyError) as cm:
            ConvexBody([])
        self.assertIn("不能为空", str(cm.exception))

    def test_reject_too_many_points(self):
        pts = [(float(i), 0.0, 0.0) for i in range(MAX_POINTS + 1)]
        with self.assertRaises(InvalidBodyError) as cm:
            ConvexBody(pts)
        self.assertIn(str(MAX_POINTS + 1), str(cm.exception))
        self.assertIn(str(MAX_POINTS), str(cm.exception))

    def test_accepts_bound_64_points(self):
        pts = [(float(i), 1.0, 0.0) for i in range(MAX_POINTS)]
        body = ConvexBody(pts)
        self.assertEqual(len(body), MAX_POINTS)

    def test_points_must_be_sequence(self):
        with self.assertRaises(TypeError):
            ConvexBody({(0.0, 0.0, 0.0)})

    def test_duplicate_points_deduped_deterministically(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        body = ConvexBody(pts)
        self.assertEqual(len(body), 2)

    def test_integer_points_accepted(self):
        r = distance([(0, 0, 0), (2, 0, 0)], [(5, 0, 0)])
        self.assertAlmostEqual(r.distance, 3.0)

    def test_box_rejects_bool_and_nan(self):
        with self.assertRaises(TypeError):
            box((True, 0, 0), (1, 1, 1))
        with self.assertRaises(InvalidBodyError):
            box((0, 0, float("nan")), (1, 1, 1))


class TestParameterValidation(unittest.TestCase):
    A = [(0.0, 0.0, 0.0)]
    B = [(3.0, 0.0, 0.0)]

    def test_max_iterations_bool_rejected(self):
        with self.assertRaises(TypeError) as cm:
            distance(self.A, self.B, max_iterations=True)
        self.assertIn("max_iterations", str(cm.exception))

    def test_max_iterations_range(self):
        distance(self.A, self.B, max_iterations=1)  # 合法边界
        distance(self.A, self.B, max_iterations=MAX_ITERATIONS)
        for bad in (0, -1, MAX_ITERATIONS + 1):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidParameterError):
                    distance(self.A, self.B, max_iterations=bad)

    def test_max_iterations_type(self):
        with self.assertRaises(TypeError):
            distance(self.A, self.B, max_iterations=4.0)

    def test_tolerance_validation(self):
        with self.assertRaises(InvalidParameterError):
            distance(self.A, self.B, tolerance_rel=-1e-12)
        with self.assertRaises(InvalidParameterError):
            distance(self.A, self.B, tolerance_abs=float("nan"))
        with self.assertRaises(TypeError):
            distance(self.A, self.B, tolerance_rel=True)


class TestStatelessness(unittest.TestCase):
    def test_invalid_call_leaves_no_partial_effect(self):
        body = ConvexBody([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        before = body.points
        with self.assertRaises(InvalidBodyError):
            distance(body, [(float("inf"), 0.0, 0.0)])
        self.assertEqual(body.points, before)

    def test_deterministic_no_hidden_state(self):
        a = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0)]
        b = [(3.0, 1.0, 1.0), (4.0, 2.0, 2.0)]
        r1 = distance(a, b)
        for _ in range(5):
            r = distance(a, b)
            self.assertEqual(r.distance, r1.distance)
            self.assertEqual(r.witness_a, r1.witness_a)
            self.assertEqual(r.iterations, r1.iterations)
            self.assertEqual(r.separation_direction, r1.separation_direction)


if __name__ == "__main__":
    unittest.main()
