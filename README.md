# convex_distance — 凸多面体最近点与分离证书（含椭球 Minkowski 外壳）

离线计算几何模块：给定两个凸包（各由 ≤ 64 个三维点表示），用 GJK 在
Minkowski 差上迭代求原点最近点，输出距离、两物体上的重心组合见证点，
以及可独立验证的分离方向证书。新入口 `distance_shell` 支持每个物体再
叠加一个由可逆 3×3 矩阵 `L` 给出的椭球外壳 `L·单位球`（方向厚度各向
异性），并输出可检查的分离距离上下界。

- **运行环境**：Python 3.14.7，Linux，仅标准库（`math` / `numbers` /
  `dataclasses`），安装 Python 后无需任何额外依赖。
- **核心算法为从零实现**，未调用任何第三方计算几何库。

## 接口

```python
from convex_distance import distance, distance_shell, InputError

res  = distance(A, B, *, tol=1e-9, max_iter=128)                 # 原接口（无外壳）
res2 = distance_shell(A, B, *, shell_a=L1, shell_b=L2,           # 椭球外壳接口
                      tol=1e-9, max_iter=128)
```

- `A`, `B`：各 1..64 个点的序列；每个点是恰好 3 个有限实数
  （`int` / `float` / `Fraction`）的序列。输入即其凸包，**不构造面**，
  只通过 support 函数访问。
- `shell_a` / `shell_b`：`None`（无外壳）或行主序 3×3 可逆矩阵 `L`
  （3 行 × 3 个有限实数）。物体形状为 Minkowski 和
  `conv(顶点) (+) L·单位球`。方向 `n` 上的外壳支撑偏移为

  ```
  off(n) = L Lᵀ n / ‖Lᵀ n‖
  ```

  这是真正的椭球支撑，**不会**被简化成最大半径球（否则会改变各向
  异性距离）。`L` 奇异、含 NaN/±Inf/bool、形状不符都会抛
  `InputError`；零方向（仅在外壳相交、无方向可言时出现）按契约返回
  零偏移。
- `tol`：相对收敛容差，有限实数，`0 < tol < 1`。
- `max_iter`：GJK 迭代预算，整数，`1..128`（题面契约上限即 128）。

`DistanceResult`（`distance`，frozen dataclass）字段：

| 字段 | 含义 |
|---|---|
| `status` | `"separated"` / `"intersecting"` / `"not_converged"` |
| `distance` | 最近点距离；`intersecting` 时为 ≤ 零阈值的残差（**不是穿透深度**） |
| `witness_a` / `witness_b` | 两物体上的最近点（输入点的重心组合） |
| `separating_axis` | 单位向量 `n`，满足 `min_A a·n ≥ wa·n`、`max_B b·n ≤ wb·n`；`intersecting` 时为 `None`；`not_converged` 时为未验证的最优方向 |
| `iterations` | 实际迭代次数 |
| `residual` | 最终相对间隙 `(‖v‖² − w·v)/‖v‖²`；`separated` 时 ≤ `tol` |

`ShellDistanceResult`（`distance_shell`）在此基础上把见证拆成
**核心见证 + 外壳偏移**（原输入顶点的重心组合不再完整表示膨胀面）：

| 字段 | 含义 |
|---|---|
| `core_witness_a` / `core_witness_b` | 核心最近点，输入顶点的重心组合 |
| `shell_offset_a` / `shell_offset_b` | 最终方向上的椭球偏移；`intersecting` 时为零向量 |
| `witness_a` / `witness_b` | 两膨胀外壳上的实际最近点，`witness = core_witness + shell_offset` |
| `lower_bound` | 真实距离的可证下界：各迭代 `w·v/‖v‖` 的最大值（截断到 ≥ 0），可用支撑函数独立复核 |
| `upper_bound` | `‖witness_a − witness_b‖`，由可行见证实现的真实距离上界 |

恒有 `lower_bound ≤ d_true ≤ upper_bound`；收敛分离时区间宽度在收敛
容差量级。`not_converged` 时 `[lower_bound, upper_bound]` 就是诚实的
输出——**不伪造精确相交结论**；`intersecting` 时 `lower_bound = 0`，
`distance` 为 ≤ 零阈值的残差。

`n` 的方向约定为从 B 指向 A（`n = (wa − wb)/‖wa − wb‖`）。

## 错误语义

所有非法输入抛出 `InputError`（`ValueError` 子类），消息可定位到具体元素：

- 坐标为 NaN / ±Infinity → `A[0][2]: non-finite coordinate ...`
- 坐标或 `tol` 为 `bool`、非实数 → 指明位置与类型
- `max_iter` 为 `bool` 或非整数、超出 `1..128` → 指明期望值
- 点数为 0 或 > 64、点维度 ≠ 3、输入为字符串等 → 指明位置
- 外壳矩阵行/列数不符 → `shell_a: expected exactly 3 rows` 等
- 外壳矩阵含 NaN/±Inf/bool → `shell_a[1][2]: ...`
- 外壳矩阵奇异（不可逆）→ `shell_a: singular 3x3 matrix ...`

不做任何静默纠正。两个入口均为纯函数：输入校验全部完成后才开始计算，
失败不会留下部分变更的状态。

`"not_converged"`（搜索预算耗尽）与 `"intersecting"`（已证明原点落在
Minkowski 差内）严格区分，绝不互相冒充。

## 误差口径（浮点）

- 全部计算在 IEEE 754 双精度中进行；有理数输入（`int`/`Fraction`）先转为
  float64，每个坐标引入 ≤ 0.5 ulp 的舍入。
- 收敛判据为相对间隙 `g = (‖v‖² − w·v)/‖v‖² ≤ tol`（默认 1e-9）。收敛时
  真实距离 `d*` 满足 `‖v‖·(1 − g) ≤ d* ≤ ‖v‖`，即报告距离的相对误差
  不超过 `residual ≤ tol`。
- 外壳接口的上下界独立于该判据可复核：`upper_bound` 由返回的两个可行
  见证点直接算出；`lower_bound` 由支撑函数求值 `w·v/‖v‖` 给出，测试中用
  独立方向采样复核（采样只验证支撑界与见证可行性，不作为全局最优证明）。
- 相交判定使用独立的零阈值 `‖v‖ ≤ 1e-11 · max(1, 输入尺度)`（外壳接口的
  输入尺度含外壳矩阵的 Frobenius 范数），与 `tol` 分开，不用宽容差掩盖
  边界：刚好接触（距离恰为 0）会落入 `intersecting`。
- 测试中的断言裕量（1e-7 量级）只是对 float64 累积舍入的留白，边界用例
  （接触、共面、重复点）均以精确期望值或零阈值判定。

## 设计取舍

- **Support 函数 + GJK**：不显式构造凸包面/拓扑。外壳物体的支撑为
  核心支撑加椭球偏移 `L Lᵀ d/‖Lᵀ d‖`，Minkowski 差两侧偏移同号相加。
- **降维处理退化**：单纯形最近点用 Johnson 式子集枚举（≤ 4 点 → ≤ 15 个
  子集），仿射相关的子集（共线、共面、重复点）在 KKT 求解中判奇异后跳过，
  由更小维子集覆盖 —— 不使用随机抖动。
- **见证拆分**：单纯形记录只保存核心顶点；最终核心见证取重心组合，
  外壳偏移在最终分离方向上重新计算，使 `witness = core + offset` 精确
  落在膨胀面上（偏移即椭球边界点 `L·u`，`‖u‖ = 1`）。
- **确定性**：支撑函数平局取最小下标，初始方向取质心差，结果可复现。

## 已知限制

- 不提供穿透深度（EPA 超出本题范围）；相交/接触统一报 `intersecting`。
- `not_converged` 时的 `separating_axis` 是未验证的最优方向，不能当证书用；
  此时以 `[lower_bound, upper_bound]` 区间为准。
- 外壳必须是可逆矩阵给出的完整椭球；零厚度/半无穷外壳不在契约内。
- 不含连续碰撞检测（CCD），不接入真实几何内核；只实现本题契约的数学
  子集（三维、凸包 + 椭球外壳、距离与分离证书），不宣称兼容任何工业
  标准接口。
- 坐标量级极端（≳ 1e150 或 ≲ 1e-150）时 float64 上溢/下溢不在保障范围内。

## 运行

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 39 个测试
./.venv/bin/python demo.py                            # < 1 秒，打印分离证书、外壳见证拆分与拒绝示例
```

测试中的参考实现（点-线段解析式、轴对齐盒解析式、小凸组合穷举的
active-set 枚举、球距离解析式、点-椭球 Lagrange 乘子二分、独立方向
采样的支撑界复核）全部独立写在 `tests/` 内，不调用被测核心。旧接口
`distance` 的回归测试（`tests/test_convex_distance.py`）原样保留。
