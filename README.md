# convex_distance — 凸多面体最近点、分离证书与平移扫掠接触

离线计算几何模块，两个公开入口：

- `distance(A, B)`：给定两个凸包（各由 ≤ 64 个三维点表示），用 GJK 在
  Minkowski 差上迭代求原点最近点，输出距离、两物体上的重心组合见证点，
  以及可独立验证的分离方向证书。
- `sweep(A, B, velocity, t_window)`：B 相对 A 做一段固定速度平移
  （无旋转），求时间窗口内的首次接触时刻与相交区间，覆盖初始相交、
  相切、相对静止、完全错过四种情形，输出可复查的接触状态。

- **运行环境**：Python 3.14.7，Linux，仅标准库（`math` / `numbers` /
  `dataclasses`），安装 Python 后无需任何额外依赖。
- **核心算法为从零实现**，未调用任何第三方计算几何库。

## 接口

```python
from convex_distance import (
    distance, DistanceResult,
    sweep, SweepResult,
    InputError,
)

res = distance(A, B, *, tol=1e-9, max_iter=128)
sw  = sweep(A, B, velocity, (t0, t1), *, tol=1e-9, max_iter=128)
```

- `A`, `B`：各 1..64 个点的序列；每个点是恰好 3 个有限实数
  （`int` / `float` / `Fraction`）的序列。输入即其凸包，**不构造面**，
  只通过 support 函数访问。
- `velocity`：恰好 3 个有限实数，B 相对 A 的平移速度（A 静止、
  `B(t) = B + t·velocity`；只取决于相对运动，因此任意双方的匀速
  平移都可用其相对速度表示）。
- `t_window`：`(t0, t1)` 两个有限实数，要求 `t0 < t1`。
- `tol`：相对收敛容差，有限实数，`0 < tol < 1`。
- `max_iter`：单次 GJK 迭代预算，整数，`1..128`（题面契约上限即 128）。

`DistanceResult`（frozen dataclass）字段：

| 字段 | 含义 |
|---|---|
| `status` | `"separated"` / `"intersecting"` / `"not_converged"` |
| `distance` | 最近点距离；`intersecting` 时为 ≤ 零阈值的残差（**不是穿透深度**） |
| `witness_a` / `witness_b` | 两物体上的最近点（输入点的重心组合） |
| `separating_axis` | 单位向量 `n`，满足 `min_A a·n ≥ wa·n`、`max_B b·n ≤ wb·n`；`intersecting` 时为 `None`；`not_converged` 时为未验证的最优方向 |
| `iterations` | 实际迭代次数 |
| `residual` | 最终相对间隙 `(‖v‖² − w·v)/‖v‖²`；`separated` 时 ≤ `tol`（唯一的例外见下文"误差口径"：float64 噪声层触停时如实报告，可能超过 `tol`） |

`n` 的方向约定为从 B 指向 A（`n = (wa − wb)/‖wa − wb‖`）。

`SweepResult`（frozen dataclass）字段：

| 字段 | 含义 |
|---|---|
| `status` | `"miss"`（窗口内无接触）/ `"touching"`（接触区间退化为单一时刻，相切）/ `"intersecting"`（正长度相交区间，含初始相交） |
| `t_enter` / `t_exit` | 窗口内首次 / 末次接触时刻；`miss` 时为 `None` |
| `initially_overlapping` | `t0` 时刻已接触或相交 |
| `stationary` | 相对速度恰为 `(0, 0, 0)`（此时接触状态在整个窗口不变） |
| `relative_velocity` / `window` | 实际使用的速度与时间窗口 |
| `t_closest` / `min_distance` | 窗口内最近距离时刻及其距离（接触时 ≤ 零阈值） |
| `witness_a` / `witness_b` | 接触时取 `t_enter`、错过时取 `t_closest` 时刻的世界坐标最近点 |
| `separating_axis` | 错过时 `t_closest` 处的分离方向证书；接触时为 `None` |
| `evaluations` | 本次扫掠实际调用的 GJK 距离求值次数（复查用） |
| `zero_tolerance` | 判定接触所用的距离零阈值 |

便捷属性：`sw.contact`（是否接触）、`sw.contact_interval`
（`(t_enter, t_exit)` 或 `None`）。

## 错误语义

所有非法输入抛出 `InputError`（`ValueError` 子类），消息可定位到具体元素：

- 坐标为 NaN / ±Infinity → `A[0][2]: non-finite coordinate ...`
- 坐标、`velocity`、`t_window` 端点或 `tol` 为 `bool`、非实数 → 指明位置与类型
- `velocity` 分量数 ≠ 3、`t_window` 端点数 ≠ 2、`t0 >= t1` → 指明期望值
- `max_iter` 为 `bool` 或非整数、超出 `1..128` → 指明期望值
- 点数为 0 或 > 64、点维度 ≠ 3、输入为字符串等 → 指明位置

不做任何静默纠正。`distance()` 与 `sweep()` 均为纯函数：输入校验全部
完成后才开始计算，失败不会留下部分变更的状态。

`"not_converged"`（搜索预算耗尽）与 `"intersecting"`（已证明原点落在
Minkowski 差内）严格区分，绝不互相冒充。

## 误差口径（浮点）

- 全部计算在 IEEE 754 双精度中进行；有理数输入（`int`/`Fraction`）先转为
  float64，每个坐标引入 ≤ 0.5 ulp 的舍入。
- 收敛判据为相对间隙 `g = (‖v‖² − w·v)/‖v‖² ≤ tol`（默认 1e-9）。收敛时
  真实距离 `d*` 满足 `‖v‖·(1 − g) ≤ d* ≤ ‖v‖`，即报告距离的相对误差
  不超过 `residual ≤ tol`。
- 唯一的例外：当支撑点重复出现（确定性迭代在 float64 噪声层上打转，
  `‖v‖²` 的可能改进量已低于机器分辨率）时，迭代以 `separated` 终止并
  **如实报告**残差，此时 `residual` 可能超过 `tol`；`distance` 本身已达
  float64 最优。预算耗尽仍然报 `not_converged`，二者不会混淆。
- 相交判定使用独立的零阈值 `‖v‖ ≤ 1e-11 · max(1, 输入坐标最大模长)`，
  与 `tol` 分开，不用宽容差掩盖边界：刚好接触（距离恰为 0）会落入
  `intersecting`，且 `distance ≤` 零阈值。重心坐标可行性裕量取 1e-12
  （无量纲），保证"仿射投影被误当凸组合"造成的距离低估（约
  `裕量 × 单纯形直径`）低于该零阈值。
- 扫掠的接触判定复用同一零阈值（对整个扫掠路径取固定尺度，不随 t 漂移）。
  时间定位分辨率 `t_tol ≈ 1e-3 · 零阈值 / 速度`（另有 ~1000 ulp 的绝对
  下限），接触时刻误差由此界定。
- `touching` 与 `intersecting` 的区分：多面体的距离剖面关于 t 分段线性，
  因此相切（零长度接触）与正长度区间在零阈值处清晰可分；分类阈值为
  `max(64·t_tol, 1e-9·max(1, |t0|, |t1|))`，短于该值的相交区间会被归
  为 `touching`（这是分类分辨率极限，见"已知限制"）。
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
- **扫掠归约为一维凸优化**：`d(t) = dist(A, B+t·v)` 是直线与凸 Minkowski
  差的距离，关于 t 是凸函数，其零集是单区间。先用黄金分割在窗口上求
  最小值点 `t*`：`d(t*) > 零阈值` 即完全错过；否则在 `[t0, t*]`、`[t*, t1]`
  上分别二分求进入 / 离开时刻。相对静止（速度恰为零向量）时距离为常数，
  一次求值即可判定。
- **确定性**：支撑函数平局取最小下标，初始方向取质心差，黄金分割与二分
  均为固定格式，结果可复现。

## 已知限制

- 不提供穿透深度（EPA 超出本题范围）；相交/接触统一报 `intersecting`。
- `not_converged` 时的 `separating_axis` 是未验证的最优方向，不能当证书用。
- 扫掠只支持**单段固定速度平移**（无旋转、无加速、无多段轨迹拼接）；
  运动拓扑事件证书不在本次交付范围。
- 长度短于分类阈值（约 `1e-9 · max(1, |t|)` 量级）的相交区间会被归为
  `touching`；速度恰为零才判定 `stationary`，近零速度按普通运动处理。
- 只实现本题契约的数学子集（三维、凸包、距离与分离证书、单段平移
  扫掠），不宣称兼容任何工业标准接口。
- 坐标量级极端（≳ 1e150 或 ≲ 1e-150）时 float64 上溢/下溢不在保障范围内。

## 运行

```bash
python -m unittest discover -s tests -v   # 46 个测试
python demo.py                            # < 1 秒，打印分离证书、扫掠接触区间与拒绝示例
```

测试中的参考实现（点-线段解析式、轴对齐盒解析式、小凸组合穷举的
active-set 枚举、轴对齐盒扫掠的逐轴解析接触区间）全部独立写在
`tests/` 内，不调用被测核心。
