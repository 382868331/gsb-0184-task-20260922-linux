# convex_distance — 凸体距离、椭球 Minkowski 外壳与分离证书

离线计算几何模块：给定两个凸包（各由 ≤ 64 个三维点表示），用 GJK 在
Minkowski 差上迭代求原点最近点，输出距离、两物体上的见证点与可独立
验证的分离证书。第二版支持给任一物体（或两者）附加**由可逆 3×3 矩阵
`L` 给出的椭球 Minkowski 外壳**，外壳集合为 `conv(points) + L·S2`
（`S2` 为三维闭单位球），外壳在方向 `n` 上的厚度为 `‖L^T n‖`，各方向
可以不同——**不会被简化成一个最大半径的球而改变距离**。

- **运行环境**：Python 3.14.7，Linux，仅标准库（`math` / `numbers` /
  `dataclasses`），无需任何额外依赖；核心完全离线。
- **核心算法为从零实现**，未调用任何第三方计算几何库，不含连续碰撞
  检测（CCD/TOI）。

## 接口

### 无外壳（旧接口，契约不变）

```python
from convex_distance import distance, DistanceResult, InputError

res = distance(A, B, *, tol=1e-9, max_iter=128)
```

### 椭球外壳（新入口）

```python
from convex_distance import distance_shelled

res = distance_shelled(A, B, LA=None, LB=None, *, tol=1e-9, max_iter=128)
```

- `A`, `B`：各 1..64 个点的序列；每个点是恰好 3 个有限实数
  （`int` / `float` / `Fraction`，`bool` 拒绝）。输入即其凸包，
  **不构造面**，只通过 support 函数访问。
- `LA` / `LB`：`None`（该侧无外壳）或 **3×3 可逆矩阵**（9 个有限
  实数，`bool` 拒绝）。非 `None` 时，被查询集合为
  `conv(points) + L·S2`。`L` 的列向量是椭球半轴；例如
  `diag(0.4, 3, 2)` 表示 x 向厚 0.4、y 向厚 3、z 向厚 2。
- `tol`：相对收敛容差，有限实数，`0 < tol < 1`。
- `max_iter`：GJK 迭代预算，整数，`1..128`。

### 椭球外壳的 support 偏移

单位方向 `n` 上椭球 `L·S2` 的支撑点（唯一暴露点）为

```
h_off(L, n) = L·L^T·n / ‖L^T n‖ ,      h_off(L, n)·n = ‖L^T n‖
```

该式对方向正齐次（次数 0），所以任意非零方向向量都可直接用，无需
先归一化；**零方向没有唯一支撑点，明确报错**。被膨胀物体的支撑点：

```
A 侧（沿 +d 最大化）: argmax_A a·d + h_off(LA,  d)
B 侧（沿 +d 最小化）: argmin_B b·d + h_off(LB, -d)
```

GJK 的每次 support 都把**外壳支撑点与原凸包支撑点一起带入见证**。

## 返回结构

`DistanceResult`（旧，无外壳）字段：

| 字段 | 含义 |
|---|---|
| `status` | `"separated"` / `"intersecting"` / `"not_converged"` |
| `distance` | 最近点距离；`intersecting` 时为 ≤ 零阈值的残差（**不是穿透深度**） |
| `witness_a` / `witness_b` | 两物体上的最近点（输入点的重心组合） |
| `separating_axis` | 单位向量 `n`（B → A）；分离时为已验证证书；相交时 `None`；未收敛时为未验证的最优方向 |
| `iterations` / `residual` | 实际迭代次数 / 最终相对间隙 |

`ShelledDistanceResult`（新）字段：

| 字段 | 含义 |
|---|---|
| `status`, `distance`, `separating_axis`, `iterations`, `residual` | 同上 |
| `witness_a` / `witness_b` | 两物体**实际外壳上**的最近点 |
| `core_a` / `core_b` | 原凸包上的核心见证（输入顶点的重心组合） |
| `offset_a` / `offset_b` | 外壳偏移，满足 `witness = core + offset`，且 `‖L⁻¹ offset‖ ≤ 1` |
| `lower_bound` / `upper_bound` | 真实外壳距离的可检查区间：`lower ≤ d* ≤ upper`；相交时两者为 0 |
| `shell_a` / `shell_b` | 该侧是否带外壳 |

**核心见证与外壳偏移必须拆开**：原输入顶点的重心组合只落在核心凸包
面上，不再完整表示膨胀后的表面（例如线段核心加球壳是胶囊体，最近点
可能落在圆柱侧面上，对应一个内部核心点加径向偏移）。最终外壳见证的
偏移是单纯形各支撑偏移的同组重心组合，是椭球内的可行点（分离时接触
侧恰在边界 `‖L⁻¹ offset‖ = 1`）。

分离上下界来自支撑间隙证书：`upper = ‖v‖`（可行见证距离），
`lower = ‖v‖·(1 − residual)`（support 下界），区间相对宽度即
`residual`。

## 错误语义

所有非法输入抛出 `InputError`（`ValueError` 子类），消息可定位到
具体元素（`A[3][1]` / `LA[1][2]` 等），不做任何静默纠正：

- 坐标 / 矩阵元素为 NaN、±Infinity → 指明位置；
- 坐标、矩阵元素或 `tol` 为 `bool`、非实数 → 指明类型；
- **矩阵奇异**（零矩阵、行列式为 0、或最小主元 ≤ `1e-12·max|元素|`
  的近奇异矩阵）→ `LA/LB: matrix is singular ...`，要求可逆；
- **零方向**调用外壳支撑函数 → `n: zero support direction ...`；
- 矩阵不是 3×3、点数为 0 或 > 64、点维度 ≠ 3、输入为字符串等。

`distance_shelled()` 为纯函数：全部输入校验通过后才开始计算。

`"not_converged"`（搜索预算耗尽）与 `"intersecting"`（已证明原点在
Minkowski 差内）严格区分：**未收敛绝不伪造精确相交结论**，仍返回
`[lower_bound, upper_bound]` 区间与未验证的最优方向。刚好接触（距离
恰为 0）报 `intersecting`。

## 误差口径（浮点）

- 全部计算在 IEEE 754 双精度中进行；有理数输入先转 float64。
- 收敛判据为相对间隙 `g = (‖v‖² − w·v)/‖v‖² ≤ tol`（默认 1e-9）。
  收敛时真实距离满足 `‖v‖·(1−g) ≤ d* ≤ ‖v‖`；椭球外壳完全沿用该
  证书（支撑点取在椭球面上，支撑间隙不等式对膨胀集合同样成立）。
- 相交判定使用独立零阈值 `‖v‖ ≤ 1e-11·max(1, 坐标与矩阵元素最大模长)`。
- 面接触（如胶囊圆柱侧）时核心见证可能沿低维核心有 1e-8 量级滑动，
  但外壳见证 `witness = core + offset` 与距离仍精确；测试据此断言。

## 设计取舍

- **Support 函数 + GJK**：不构造凸包面/拓扑；外壳只增加 O(1) 的
  `L·L^T·n/‖L^T n‖ 计算，每次迭代仍为 O(n)。
- **降维处理退化**：单纯形最近点用 Johnson 式子集枚举（≤ 4 点），
  仿射相关子集（共线、共面、重复点）判奇异后跳过，不用随机抖动。
  退化线段核心 + 外壳即胶囊体，由同一引擎统一处理。
- **见证拆分**：每条单纯形记录同时保存核心顶点、两侧外壳偏移与
  外壳支撑点，最终用同一组重心权重组合出核心/偏移/外壳见证。
- **确定性**：support 平局取最小下标，初始方向取质心差。

## 已知限制

- 不提供穿透深度（EPA 超出范围）；相交/接触统一报 `intersecting`，
  此时 `distance` 是残差而非穿透深度。
- 不做连续碰撞 / 飞行时间（CCD/TOI），不调用任何真实几何库。
- `not_converged` 时的 `separating_axis` 是未验证方向，区间可用但
  不能当严格分离证书。
- 矩阵要求可逆：半厚（半轴为 0 的扁椭球）需用零厚度侧 + 低维核心
  表达，不接受奇异矩阵。
- 坐标/矩阵量级极端（≳ 1e150 或 ≲ 1e-150）时的上溢/下溢不在保障内。

## 运行

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 48 个测试
./.venv/bin/python demo.py                            # 实测 < 1 秒
```

测试组织：

- `tests/test_convex_distance.py`（23 个，旧回归全保留）：点-线段
  解析式、轴对齐盒解析式、小凸组合穷举 active-set 参考、证书、
  预算耗尽状态、输入校验；参考实现全部独立，不调用被测核心。
- `tests/test_shelled_distance.py`（25 个，新增）：
  - 球形特例核对解析球距离（点核 + 双/单侧半径、三角形核心减半径）；
  - 不同主轴的点-椭球（轴对齐三轴、共享轴双椭球、旋转外壳的主轴
    接触点、椭球内部判相交）；
  - **方向性反例**：x 半轴 0.4、y 半轴 3 时沿 x 的间隙是 `D−0.4`，
    明确断言不等于"最大半径球"的 `D−3`；
  - 退化线段核心 + 球壳（胶囊的柱面/球帽两侧）与椭球壳；
  - **独立方向采样**（fibonacci 球面 + 随机方向）只核对 support 界
    （`off·n = ‖L^T n‖`、`‖L⁻¹ off‖ = 1`、采样椭球点不越界）与
    返回见证的可行性（核心在凸包内、偏移在椭球内、区间有序），
    **不把采样当作全局最优证明**；
  - 未收敛保留区间、无外壳时与旧接口逐项一致、零方向/奇异/非有限/
    形状错误全部拒绝。
