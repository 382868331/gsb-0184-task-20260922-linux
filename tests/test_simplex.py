"""最近点子算法（Johnson 子集枚举）的直接单元测试。"""

import math
import unittest

from convex_distance.simplex import (
    MinkowskiVertex,
    closest_on_simplex,
)


def mv(p, ai=0, bi=0):
    return MinkowskiVertex(p=p, ai=ai, bi=bi)


def norm(v):
    return math.sqrt(sum(c * c for c in v))


class TestClosestOnSimplex(unittest.TestCase):
    def test_single_vertex(self):
        s = closest_on_simplex([mv((3.0, 4.0, 0.0))])
        self.assertAlmostEqual(s.norm, 5.0)
        self.assertEqual(s.weights, (1.0,))
        self.assertEqual(s.mask, 0b1)

    def test_segment_origin_between_endpoints(self):
        # 线段穿过原点附近：棱向量 Gram 不应奇异（接触友好）
        s = closest_on_simplex([mv((-2.0, 0.0, 0.0)), mv((3.0, 0.0, 0.0))])
        self.assertAlmostEqual(s.norm, 0.0)
        self.assertAlmostEqual(s.weights[0], 0.6, places=12)
        self.assertAlmostEqual(s.weights[1], 0.4, places=12)
        self.assertEqual(s.subset_size, 2)

    def test_segment_origin_beyond_endpoint(self):
        s = closest_on_simplex([mv((2.0, 0.0, 0.0)), mv((5.0, 0.0, 0.0))])
        self.assertAlmostEqual(s.point[0], 2.0)
        self.assertEqual(s.subset_size, 1)

    def test_triangle_closest_at_edge(self):
        verts = [
            mv((0.0, 1.0, 0.0)),
            mv((-1.0, 3.0, 0.0)),
            mv((1.0, 3.0, 0.0)),
        ]
        s = closest_on_simplex(verts)
        self.assertAlmostEqual(s.norm, 1.0)
        self.assertEqual(s.subset_size, 1)

    def test_triangle_interior_closest(self):
        # 三角形在 z=2 平面，原点投影在三角形内部
        verts = [
            mv((-3.0, -3.0, 2.0)),
            mv((3.0, -3.0, 2.0)),
            mv((0.0, 3.0, 2.0)),
        ]
        s = closest_on_simplex(verts)
        self.assertAlmostEqual(s.norm, 2.0, places=11)
        self.assertEqual(s.subset_size, 3)
        self.assertAlmostEqual(sum(s.weights), 1.0, places=12)

    def test_degenerate_collinear_triangle_falls_back_to_edge(self):
        # 三点共线：三角形子集奇异被跳过，自动降维到线段（无抖动）
        verts = [
            mv((-2.0, 0.0, 0.0)),
            mv((1.0, 0.0, 0.0)),
            mv((4.0, 0.0, 0.0)),
        ]
        s = closest_on_simplex(verts)
        self.assertAlmostEqual(s.norm, 0.0)
        self.assertEqual(s.subset_size, 2)  # 不是 3
        self.assertGreater(s.singular_skipped, 0)
        # 权重仅在获胜边上非零（原点在 -2..1 之间，权为 1/3、2/3）
        active = [i for i, w in enumerate(s.weights) if w > 1e-12]
        self.assertEqual(len(active), 2
                         )
        self.assertAlmostEqual(s.point[0], 0.0, places=12)

    def test_degenerate_coplanar_tetra_falls_back(self):
        verts = [
            mv((0.0, 0.0, 1.0)),
            mv((1.0, 0.0, 1.0)),
            mv((0.0, 1.0, 1.0)),
            mv((1.0, 1.0, 1.0)),
        ]
        s = closest_on_simplex(verts)
        self.assertAlmostEqual(s.norm, 1.0, places=11)
        self.assertLessEqual(s.subset_size, 3)  # 四面体全共面 -> 降维

    def test_duplicate_vertices_skip_singular(self):
        verts = [
            mv((1.0, 0.0, 0.0), ai=0, bi=0),
            mv((1.0, 0.0, 0.0), ai=1, bi=0),  # 不同来源但同点
            mv((0.0, 1.0, 0.0)),
        ]
        s = closest_on_simplex(verts)
        self.assertAlmostEqual(s.norm, 1.0 / math.sqrt(2.0), places=11)

    def test_weights_are_convex(self):
        verts = [
            mv((2.0, -1.0, 0.5)),
            mv((-1.0, 2.0, 0.5)),
            mv((1.0, 1.0, -2.0)),
            mv((-2.0, -2.0, -1.0)),
        ]
        s = closest_on_simplex(verts)
        for w in s.weights:
            self.assertGreaterEqual(w, -1e-15)
        self.assertAlmostEqual(sum(s.weights), 1.0, places=12)
        # point 与权重组合一致
        combo = [0.0, 0.0, 0.0]
        for w, v in zip(s.weights, verts):
            for d in range(3):
                combo[d] += w * v.p[d]
        for d in range(3):
            self.assertAlmostEqual(combo[d], s.point[d], places=10)


if __name__ == "__main__":
    unittest.main()
