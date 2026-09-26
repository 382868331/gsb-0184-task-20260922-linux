# convex_distance — 凸多面体最近点、分离证书与穿透深度

离线计算几何模块：给定两个凸包（各由 ≤ 64 个三维点表示），用 GJK 在
Minkowski 差上迭代求原点最近点，输出距离、两物体上的重心组合见证点，
以及可独立验证的分离方向证书；确认相交后用 EPA（扩展多面体）给出
最小平移穿透深度的上下界、分离方向与两侧接触点。

- **运行环境**：Python 3.14.7，Linux，仅标准库（`math` / `numbers` /
  `dataclasses`），安装 Python 后无需任何额外依赖。
- **核心算法为从零实现**，未调用任何第三方计算几何库。
- 当前版本：`1.1.0`（新增 `penetration()`；`distance()` 接口不变）。

## 准备与运行

```bash
python3.14 -m venv .venv        # 可选；模块本身零依赖
source .venv/bin/activate
python -m unittest discover -s tests -v   # 全部测试（含独立对照）
python demo.py                             # 距离/分离证书演示
python demo_iteration.py                   # 穿透深度演示（正常+边界）
```

## 接口

```python
from convex_distance import distance, DistanceResult, InputError

res = distance(A, B, *, tol=1e-9, max_iter=128)
```

- `A`, `B`：各 1..64 个点的序列；每个点是恰好 3 个有限实数
  （`int` / `float` / `Fraction`）的序列。输入即其凸包，**不构造面**，
  只通过 support 函数访问。
- `tol`：相对收敛容差，有限实数，`0 < tol < 1`。
- `max_iter`：GJK 迭代预算，整数，`1..128`（题面契约上限即 128）。

`DistanceResult`（frozen dataclass）字段：

| 字段 | 含义 |
|---|---|
| `status` | `"separated"` / `"intersecting"` / `"not_converged"` |
| `distance` | 最近点距离；`intersecting` 时为 ≤ 零阈值的残差（**不是穿透深度**） |
| `witness_a` / `witness_b` | 两物体上的最近点（输入点的重心组合） |
| `separating_axis` | 单位向量 `n`，满足 `min_A a·n ≥ wa·n`、`max_B b·n ≤ wb·n`；`intersecting` 时为 `None`；`not_converged` 时为未验证的最优方向 |
| `iterations` | 实际迭代次数 |
| `residual` | 最终相对间隙 `(‖v‖² − w·v)/‖v‖²`；`separated` 时 ≤ `tol` |

`n` 的方向约定为从 B 指向 A（`n = (wa − wb)/‖wa − wb‖`）。

## 穿透查询（EPA）

接口：`res = penetration(A, B, *, tol=1e-9, max_iter=128)`，返回
`PenetrationResult`（frozen dataclass）。GJK 确认原点落入 Minkowski
差 C = A − B 后，EPA 在 C 内部围绕原点扩展一个包围多面体，输出最
小平移向量的可验证上下界：

| 字段 | 含义 |
|---|---|
| `status` | penetrating / touching / separated / not_converged |
| `depth` | 深度最优估计（上下界中点）；touching/separated 时为 0 |
| `depth_lower` | 可验证下界：原点到内部多面体 P ⊆ C 最近三角面的距离，真实深度 ≥ 此值 |
| `depth_upper` | 可验证上界：所有已查询单位法向 n 上 support_C(n)·n 的最小值，真实深度 ≤ 此值（初始四面体即提供 4 次查询，恒有限） |
| `direction` | 单位向量 n（从 B 指向 A）：把 B 平移 t·n（任意 t > depth_upper）即可分离，depth·n 即（近似）最小平移向量；touching/separated 时为 None |
| `witness_a` / `witness_b` | 接触点，分别为 A、B 输入点的重心组合；witness_a − witness_b = depth_lower · direction |
| `iterations` | GJK 播种 + EPA 扩展的实际迭代数（两阶段预算各为 max_iter） |
| `residual` | (depth_upper − depth_lower) / depth_upper；收敛时 ≈ ≤ tol |

**零深度明确状态**：共面贴合、共享棱/顶点等“相交但无正体积穿透”
的情形返回 status=touching、depth=0、direction=None —— 不虚构任
何法向。判定依据是 EPA 下界收敛到零阈值以下，而非 GJK 单纯形的偶
然形状。

**退化与预算**：退化面（零面积）在构面时直接跳过；重复支持点或迭
代预算耗尽时返回 not_converged，但 depth_lower/depth_upper 仍是可
独立验证的界，direction 标注为当前最优而非已认证的唯一答案。

## 错误语义

所有非法输入抛出 `InputError`（`ValueError` 子类），消息可定位到具体元素：

- 坐标为 NaN / ±Infinity → `A[0][2]: non-finite coordinate ...`
- 坐标或 `tol` 为 `bool`、非实数 → 指明位置与类型
- `max_iter` 为 `bool` 或非整数、超出 `1..128` → 指明期望值
- 点数为 0 或 > 64、点维度 ≠ 3、输入为字符串等 → 指明位置

不做任何静默纠正。`distance()` 为纯函数：输入校验全部完成后才开始计算，
失败不会留下部分变更的状态。

`"not_converged"`（搜索预算耗尽）与 `"intersecting"`（已证明原点落在
Minkowski 差内）严格区分，绝不互相冒充。`penetration()` 的输入校验与
`distance()` 完全一致。

## 误差口径（浮点）

- 全部计算在 IEEE 754 双精度中进行；有理数输入（`int`/`Fraction`）先转为
  float64，每个坐标引入 ≤ 0.5 ulp 的舍入。
- 收敛判据为相对间隙 `g = (‖v‖² − w·v)/‖v‖² ≤ tol`（默认 1e-9）。收敛时
  真实距离 `d*` 满足 `‖v‖·(1 − g) ≤ d* ≤ ‖v‖`，即报告距离的相对误差
  不超过 `residual ≤ tol`。
- 相交判定使用独立的零阈值 `‖v‖ ≤ 1e-11 · max(1, 输入坐标最大模长)`，
  与 `tol` 分开，不用宽容差掩盖边界：刚好接触（距离恰为 0）会落入
  `intersecting`，且 `distance ≤` 零阈值。
- 测试中的断言裕量（1e-7 量级）只是对 float64 累积舍入的留白，边界用例
  （接触、共面、重复点）均以精确期望值或零阈值判定。

## 设计取舍

- **Support 函数 + GJK**：不显式构造凸包面/拓扑，每次迭代只需
  `argmax a·d` / `argmin b·d`，O(n) 每次迭代。
- **降维处理退化**：单纯形最近点用 Johnson 式子集枚举（≤ 4 点 → ≤ 15 个
  子集），仿射相关的子集（共线、共面、重复点）在 KKT 求解中判奇异后跳过，
  由更小维子集覆盖 —— 不使用随机抖动。
- **见证点**：每个单纯形点同时记录 A、B 两侧的支撑点，最终用重心权重
  组合出 `witness_a` / `witness_b`。
- **确定性**：支撑函数平局取最小下标，初始方向取质心差，结果可复现。
- **EPA 穿透**：GJK 单纯形播种四面体（退化时沿正交方向补支持点），
  逐次取离原点最近面、沿其法向取支持点扩展；下界=原点到内部多面体最近
  三角面距离，上界=已查法向的最小支持值，两者都可独立验证。

## 已知限制

- `distance()` 不提供穿透深度（用 `penetration()`）；相交/接触统一报 `intersecting`。
- `penetration()` 的 `not_converged` 只保证上下界有效，`direction` 是当前最优
  而非已认证的唯一方向；`touching` 时不给方向。
- `not_converged` 时的 `separating_axis` 是未验证的最优方向，不能当证书用。
- 只实现本题契约的数学子集（三维、凸包、距离与分离证书），不宣称兼容
  任何工业标准接口。
- 坐标量级极端（≳ 1e150 或 ≲ 1e-150）时 float64 上溢/下溢不在保障范围内。

## 运行

```bash
python -m unittest discover -s tests -v   # 23 个测试
python demo.py                            # < 1 秒，打印分离证书与拒绝示例
```

测试中的参考实现（点-线段解析式、轴对齐盒解析式、小凸组合穷举的
active-set 枚举，以及穿透测试里的 SAT 候选轴枚举与 Carathéodory 凸
组合判定）全部独立写在 `tests/` 内，不调用被测核心。
