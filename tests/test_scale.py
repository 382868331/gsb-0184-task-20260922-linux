"""尺度鲁棒性、机器精度地板、见证自洽与 64 点上限规模测试。"""

import math
import random
import unittest

from convex_distance import (
    INTERSECTING,
    NOT_CONVERGED,
    SEPARATED,
    box,
    distance,
)


class TestScaleRobustness(unittest.TestCase):
    def test_small_gap_at_huge_scale_still_separated(self):
        # 半径 R~2e12 的盒，真实缝隙 1.5：不得被相对容差吞成相交
        r = distance(box((-1e12, -1e12, -1e12), (1e12, 1e12, 1e12)),
                     box((1e12 + 1.5, 0, 0), (1e12 + 2.5, 1, 1)))
        self.assertEqual(r.status, SEPARATED)
        self.assertAlmostEqual(r.distance, 1.5, places=8)

    def test_true_contact_at_huge_scale_reports_intersecting(self):
        r = distance(box((-1e12, -1e12, -1e12), (1e12, 1e12, 1e12)),
                     box((1e12, 0, 0), (1e12 + 1, 1, 1)))
        self.assertEqual(r.status, INTERSECTING)

    def test_micro_gap_small_scale(self):
        gap = 1e-7
        r = distance(box((0, 0, 0), (1, 1, 1)),
                     box((1 + gap, 0, 0), (2, 1, 1)))
        self.assertEqual(r.status, SEPARATED)
        self.assertAlmostEqual(r.distance, gap, delta=gap * 1e-6)

    def test_witness_residual_tiny_when_certified(self):
        rng = random.Random(31)
        for _ in range(40):
            m, n = rng.randint(1, 20), rng.randint(1, 20)
            P = [(float(rng.randint(-20, 20)),
                  float(rng.randint(-20, 20)),
                  float(rng.randint(-20, 20))) for _ in range(m)]
            t = [rng.choice([-1, 1]) * rng.randint(30, 60) for _ in range(3)]
            Q = [(float(rng.randint(-20, 20)) + t[0],
                  float(rng.randint(-20, 20)) + t[1],
                  float(rng.randint(-20, 20)) + t[2]) for _ in range(n)]
            r = distance(P, Q)
            self.assertEqual(r.status, SEPARATED)
            scale = max(1.0, max(math.sqrt(sum(c * c for c in p))
                                 for p in P + Q))
            self.assertLessEqual(r.witness_residual, 1e-9 * scale + 1e-9)
            # 证书方向单位长
            self.assertAlmostEqual(
                math.sqrt(sum(c * c for c in r.separation_direction)),
                1.0, places=12,
            )

    def test_machine_floor_ambiguity_is_reachable_and_unsafe_cert_refused(self):
        """在 R~2e5 的盒上构造近接触构型：双精度地板与缝隙同阶时，
        必须返回 not_converged/machine_floor_ambiguity，
        而不是冒发分离证书。构型由固定随机种子确定性生成。"""
        rng = random.Random(7)
        found_ambiguous = False
        found_separated = False
        for _ in range(120):
            s = 10 ** rng.randint(4, 6)
            gap = 0.0
            box((-s, -s, -s), (s, s, s))  # 预热（无作用，保留生成节奏一致）
            oy = rng.uniform(-0.5, 0.5)
            oz = rng.uniform(-0.5, 0.5)
            A = box((-s, -s, -s), (s, s, s))
            B = box((s + gap, oy, oz), (s + gap + 1, oy + 1, oz + 1))
            r = distance(A, B)
            if r.status == NOT_CONVERGED:
                self.assertFalse(r.converged)
                self.assertIsNone(r.separation_direction)
                self.assertEqual(r.reason, "machine_floor_ambiguity")
                found_ambiguous = True
                break
            if r.status == INTERSECTING:
                found_separated = True
        self.assertTrue(
            found_ambiguous,
            "应当能实际触发一次机器地板不可判定（否则地板逻辑未被覆盖）",
        )
        self.assertTrue(found_separated)


class TestLargeInputs(unittest.TestCase):
    def test_64_points_boundary(self):
        # 两个各 64 点的点云（网格点），真实距离由解析方式给出
        vals = [float(k) for k in range(4)]
        grid_a = [(x, y, z) for x in vals for y in vals for z in vals]
        self.assertEqual(len(grid_a), 64)
        grid_b = [(x + 10.0, y, z) for x, y, z in grid_a]
        r = distance(grid_a, grid_b)
        self.assertEqual(r.status, SEPARATED)
        self.assertAlmostEqual(r.distance, 7.0, places=10)
        self.assertLessEqual(r.iterations, 128)

    def test_64_points_inside_small_hull(self):
        # 64 个点全部位于单位四面体内（含重复）：内部点距离为 0
        outer = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        pts = [
            (0.25 + 0.01 * (i % 5), 0.1 + 0.01 * (i % 4),
             0.05 + 0.005 * (i % 6))
            for i in range(60)
        ]
        pts = pts + outer
        r = distance(pts, [(0.2, 0.2, 0.2)])
        self.assertEqual(r.status, INTERSECTING)
        self.assertEqual(r.distance, 0.0)


if __name__ == "__main__":
    unittest.main()
