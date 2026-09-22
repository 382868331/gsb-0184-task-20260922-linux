"""解析参考与独立枚举参考对拍：点-线段、轴对齐盒、小凸组合穷举。"""

import math
import random
import unittest

from convex_distance import INTERSECTING, NOT_CONVERGED, SEPARATED, box, distance

from reference import (
    aabb_info,
    enumerative_distance,
    in_convex_hull,
    point_segment_distance,
)


def _vnorm(v):
    return math.sqrt(sum(c * c for c in v))


class TestPointSegment(unittest.TestCase):
    """点-线段：与解析公式逐例对拍。"""

    CASES = [
        # 点投影在线段中部
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        # 点超出端点
        ((5.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((-5.0, -2.0, 3.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        # 三维斜线段上的投影
        ((2.0, 2.0, 2.0), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
    ]

    ON_SEGMENT_CASES = [
        # 点恰好在线段内部/端点：距离 0，intersecting
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
        ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ]

    def test_cases_against_analytic(self):
        for p, a, b in self.CASES:
            with self.subTest(p=p, a=a, b=b):
                d_ref, q_ref = point_segment_distance(p, a, b)
                r = distance([p], [a, b])
                self.assertEqual(r.status, SEPARATED)
                self.assertAlmostEqual(r.distance, d_ref, places=12)
                # 线段侧见证应等于解析最近点
                for k in range(3):
                    self.assertAlmostEqual(r.witness_b[k], q_ref[k], places=11)
                # 点侧见证就是点本身
                for k in range(3):
                    self.assertAlmostEqual(r.witness_a[k], p[k], places=12)

    def test_on_segment_cases_intersect(self):
        for p, a, b in self.ON_SEGMENT_CASES:
            with self.subTest(p=p):
                d_ref, q_ref = point_segment_distance(p, a, b)
                self.assertAlmostEqual(d_ref, 0.0, places=12)
                r = distance([p], [a, b])
                self.assertEqual(r.status, INTERSECTING)
                self.assertEqual(r.distance, 0.0)

    def test_degenerate_segment_duplicated_endpoint(self):
        # 两端点重复：线段退化为点，不做抖动也应正确
        d_ref, _ = point_segment_distance(
            (0.0, 0.0, 2.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        r = distance([(0.0, 0.0, 2.0)],
                     [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        self.assertEqual(r.status, SEPARATED)
        self.assertAlmostEqual(r.distance, d_ref, places=12)
        self.assertEqual(r.witness_b, (1.0, 0.0, 0.0))


class TestAxisAlignedBoxes(unittest.TestCase):
    def test_axis_gaps(self):
        specs = [
            ((0, 0, 0), (1, 1, 1), (3, 0, 0), (4, 1, 1), 2.0),
            ((0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), math.sqrt(3)),
            ((-2, -3, -4), (1, 0, 2), (1, -1, -1), (5, 2, 4), 0.0),
        ]
        for lo1, hi1, lo2, hi2, gap in specs:
            with self.subTest(lo2=lo2):
                if gap > 0.0:
                    r = distance(box(lo1, hi1, name="A"),
                                 box(lo2, hi2, name="B"))
                    d_ref, qa, qb = aabb_info(lo1, hi1, lo2, hi2)
                    self.assertEqual(r.status, SEPARATED)
                    self.assertAlmostEqual(r.distance, d_ref, places=11)
                    self.assertAlmostEqual(r.distance, gap, places=12)
                    # 最近集是整张面（凸集，最近点不唯一）：GJK 的见证
                    # 只需是其中任意一点。独立核验见证落在各自最近面上：
                    for k in range(3):
                        if hi1[k] < lo2[k]:
                            self.assertAlmostEqual(r.witness_a[k], float(hi1[k]))
                            self.assertAlmostEqual(r.witness_b[k], float(lo2[k]))
                        elif hi2[k] < lo1[k]:
                            self.assertAlmostEqual(r.witness_a[k], float(lo1[k]))
                            self.assertAlmostEqual(r.witness_b[k], float(hi2[k]))
                        else:
                            self.assertGreaterEqual(
                                r.witness_a[k],
                                min(lo1[k], lo2[k]) - 1e-9)
                            self.assertLessEqual(
                                r.witness_a[k],
                                max(hi1[k], hi2[k]) + 1e-9)
                    # 解析最近点本身也是合法最近对，距离一致即可
                    self.assertAlmostEqual(
                        sum((r.witness_a[k] - r.witness_b[k]) ** 2
                            for k in range(3)) ** 0.5,
                        d_ref, places=11)

    def test_face_contact_reports_intersecting(self):
        r = distance(box((0, 0, 0), (1, 1, 1)), box((1, 0, 0), (2, 1, 1)))
        self.assertEqual(r.status, INTERSECTING)
        self.assertEqual(r.distance, 0.0)

    def test_random_boxes_against_analytic(self):
        rng = random.Random(1234)
        for _ in range(100):
            c1 = [rng.randint(-5, 5) for _ in range(3)]
            e1 = [rng.randint(0, 3) for _ in range(3)]
            c2 = [rng.randint(-5, 5) for _ in range(3)]
            e2 = [rng.randint(0, 3) for _ in range(3)]
            lo1 = [c1[k] - e1[k] for k in range(3)]
            hi1 = [c1[k] + e1[k] for k in range(3)]
            lo2 = [c2[k] - e2[k] for k in range(3)]
            hi2 = [c2[k] + e2[k] for k in range(3)]
            d_ref, qa, qb = aabb_info(lo1, hi1, lo2, hi2)
            r = distance(box(lo1, hi1), box(lo2, hi2))
            if d_ref > 1e-9:
                self.assertEqual(r.status, SEPARATED)
                self.assertAlmostEqual(r.distance, d_ref, places=9)
                # 证书自洽：witness_a - witness_b = distance * n
                n = r.separation_direction
                self.assertAlmostEqual(abs(_vnorm(n) - 1.0), 0.0, places=12)
                for k in range(3):
                    self.assertAlmostEqual(
                        r.witness_a[k] - r.witness_b[k],
                        r.distance * n[k], places=9,
                    )
                # support 证书：间隔沿 n 成立
                min_a = min(sum(n[k] * p[k] for k in range(3))
                            for p in box(lo1, hi1).points)
                max_b = max(sum(n[k] * p[k] for k in range(3))
                            for p in box(lo2, hi2).points)
                self.assertAlmostEqual(min_a - max_b, r.distance, places=8)
            else:
                self.assertEqual(r.status, INTERSECTING)


class TestEnumerativeReference(unittest.TestCase):
    """小点云：与独立 KKT 枚举参考对拍 + 见证落在凸包内。"""

    def test_random_small_clouds(self):
        rng = random.Random(20260922)
        checked = 0
        for _ in range(180):
            m, n = rng.randint(1, 5), rng.randint(1, 5)
            P = [(float(rng.randint(-3, 3)), float(rng.randint(-3, 3)),
                  float(rng.randint(-3, 3))) for _ in range(m)]
            Q0 = [(float(rng.randint(-3, 3)), float(rng.randint(-3, 3)),
                   float(rng.randint(-3, 3))) for _ in range(n)]
            if rng.random() < 0.6:
                t = [rng.choice([-1, 1]) * rng.randint(4, 9) for _ in range(3)]
                Q = [(q[0] + t[0], q[1] + t[1], q[2] + t[2]) for q in Q0]
            else:
                Q = Q0
            r = distance(P, Q)
            d_ref, lam, mu = enumerative_distance(P, Q)
            scale = max([1.0] + [_vnorm(p) for p in P + Q])
            if d_ref > 1e-8 * scale:
                self.assertEqual(r.status, SEPARATED, msg=(P, Q))
                self.assertLessEqual(abs(r.distance - d_ref), 1e-8 * scale + 1e-10)
                tol = 1e-7 * scale
                self.assertTrue(in_convex_hull(P, r.witness_a, tol),
                                msg=("wa", P, r.witness_a))
                self.assertTrue(in_convex_hull(Q, r.witness_b, tol),
                                msg=("wb", Q, r.witness_b))
                checked += 1
            else:
                self.assertEqual(r.status, INTERSECTING, msg=(P, Q, d_ref))
        self.assertGreater(checked, 80)  # 确认确实覆盖了大量分离用例


class TestDegeneracy(unittest.TestCase):
    def test_collinear_points(self):
        line = [(float(k), 0.0, 0.0) for k in range(-3, 4)]
        # 重复注入
        line.append((2.0, 0.0, 0.0))
        r = distance(line, [(0.0, 2.0, 0.0)])
        d_ref, q = point_segment_distance((0.0, 2.0, 0.0),
                                          (-3.0, 0.0, 0.0), (3.0, 0.0, 0.0))
        self.assertEqual(r.status, SEPARATED)
        self.assertAlmostEqual(r.distance, d_ref, places=12)

    def test_coplanar_squares(self):
        sq1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
               (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        sq2 = [(0.2, 0.2, 3.0), (0.8, 0.2, 3.0),
               (0.8, 0.8, 3.0), (0.2, 0.8, 3.0)]
        r = distance(sq1, sq2)
        self.assertEqual(r.status, SEPARATED)
        self.assertAlmostEqual(r.distance, 3.0, places=11)
        # 见证是面内点（凸组合），不是顶点
        self.assertAlmostEqual(r.witness_a[2], 0.0, places=11)
        self.assertAlmostEqual(r.witness_b[2], 3.0, places=11)

    def test_parallel_coplanar_segments(self):
        r = distance([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
                     [(2.0, 3.0, 0.0), (8.0, 3.0, 0.0)])
        self.assertEqual(r.status, SEPARATED)
        self.assertAlmostEqual(r.distance, 3.0, places=12)

    def test_skew_segments_3d(self):
        r = distance([(-2.0, 0.0, 1.0), (2.0, 0.0, 1.0)],
                     [(0.0, -2.0, -1.0), (0.0, 2.0, -1.0)])
        self.assertEqual(r.status, SEPARATED)
        self.assertAlmostEqual(r.distance, 2.0, places=12)
        for k, v in enumerate((0.0, 0.0, 1.0)):
            self.assertAlmostEqual(r.witness_a[k], v, places=11)
        for k, v in enumerate((0.0, 0.0, -1.0)):
            self.assertAlmostEqual(r.witness_b[k], v, places=11)

    def test_degenerate_difference_no_jitter_distance_exact(self):
        # 平行线段：Minkowski 差是退化的共面平行四边形（零体积）。
        # 即使主循环没有持有 4 顶点（无奇异子集被枚举），降维也必须
        # 无抖动地给出精确距离；奇异子集计数在子算法单测中直接覆盖。
        r = distance([(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)],
                     [(1.0, 3.0, 0.0), (3.0, 3.0, 0.0)])
        self.assertEqual(r.status, SEPARATED)
        self.assertGreaterEqual(r.singular_skipped, 0)
        self.assertAlmostEqual(r.distance, 3.0, places=12)


class TestContactAndTranslation(unittest.TestCase):
    def test_exact_face_contact(self):
        r = distance(box((0, 0, 0), (1, 1, 1)), box((1, 0, 0), (2, 1, 1)))
        self.assertEqual(r.status, INTERSECTING)
        self.assertEqual(r.distance, 0.0)

    def test_edge_contact_slanted(self):
        r = distance([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
                     [(1.0, 1.0, 0.0), (3.0, 3.0, 0.0)])
        self.assertEqual(r.status, INTERSECTING)
        for k, v in enumerate((1.0, 1.0, 0.0)):
            self.assertAlmostEqual(r.witness_a[k], v, places=11)
            self.assertAlmostEqual(r.witness_b[k], v, places=11)

    def test_point_vertex_contact(self):
        r = distance([(1.0, 1.0, 1.0)],
                     [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
        self.assertEqual(r.status, INTERSECTING)

    def test_translation_invariance(self):
        tet = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
               (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        t2 = [(x + 2.0, y + 0.3, z + 0.3) for x, y, z in tet]
        r1 = distance(tet, t2)
        shift = (12.7, -3.1, 8.9)
        tet_s = [(x + shift[0], y + shift[1], z + shift[2]) for x, y, z in tet]
        t2_s = [(x + shift[0], y + shift[1], z + shift[2]) for x, y, z in t2]
        r2 = distance(tet_s, t2_s)
        self.assertEqual(r1.status, r2.status)
        self.assertAlmostEqual(r1.distance, r2.distance, places=12)
        self.assertEqual(r1.iterations, r2.iterations)
        for k in range(3):
            self.assertAlmostEqual(
                r2.witness_a[k] - r1.witness_a[k], shift[k], places=9
            )

    def test_symmetry(self):
        a = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
        b = [(3.0, 0.2, 0.0), (4.0, 1.0, 0.0)]
        r1 = distance(a, b)
        r2 = distance(b, a)
        self.assertAlmostEqual(r1.distance, r2.distance, places=12)
        for k in range(3):
            self.assertAlmostEqual(r1.witness_a[k], r2.witness_b[k], places=11)


class TestBudget(unittest.TestCase):
    def test_budget_exhaustion_distinct_from_intersection(self):
        r = distance(box((0, 0, 0), (1, 1, 1)),
                     box((3, 0.5, 0.5), (4, 1.5, 1.5)),
                     max_iterations=1)
        self.assertFalse(r.converged)
        self.assertEqual(r.status, NOT_CONVERGED)
        self.assertEqual(r.reason, "budget_exhausted")
        self.assertIsNone(r.separation_direction)
        # 估计值仍然合理（真实距离 2.0）
        self.assertGreater(r.distance, 1.5)

    def test_max_iterations_128_cap_enforced(self):
        r = distance(box((0, 0, 0), (1, 1, 1)),
                     box((30, 0, 0), (31, 1, 1)))
        self.assertLessEqual(r.iterations, 128)
        self.assertEqual(r.status, SEPARATED)


if __name__ == "__main__":
    unittest.main()
