# convex_distance — 凸体最近点、分离证书与定速平移扫描

离线计算几何模块，包含两个公开入口：

1. **静态最近点** `distance(A, B)`：给定两个凸包（各由 ≤ 64 个三维点
   表示），用 GJK 在 Minkowski 差上迭代求原点最近点，输出距离、两物体
   上的重心组合见证点，以及可独立验证的分离方向证书。
2. **定速平移扫描** `sweep_contact(A, B, velocity, t_start, t_end)`：
   A 以固定速度平移（B 静止；只取决于相对运动），求时间窗内**首次接触
   时刻**与**完整相交区间**，支持初始相交、瞬时相切、相对静止和完全
   错过，并给出可复查的接触点与接触法向。

- **运行环境**：Python 3.14.7，Linux，仅标准库（`math` / `numbers` /
  `dataclasses`），安装 Python 后无需任何额外依赖。
- **核心算法为从零实现**，未调用任何第三方计算几何库。

## 接口

```python
from convex_distance import (
    distance, sweep_contact,
    DistanceResult, SweepResult, ContactInfo, InputError,
)

res  = distance(A, B, *, tol=1e-9, max_iter=128)
sw   = sweep_contact(A, B, velocity, t_start, t_end, *,
                     tol=1e-9, max_iter=128)
```

- `A`, `B`：各 1..64 个点的序列；每个点是恰好 3 个有限实数
  （`int` / `float` / `Fraction`，`bool` 拒绝）的序列。输入即其凸包，
  **不构造面**，只通过 support 函数访问。
- `tol`：相对收敛容差，有限实数，`0 < tol < 1`。
- `max_iter`：**每次** GJK 求解的迭代预算，整数 `1..128`；扫描内部
  只发起有界次数的这类求解。
- `velocity`：恰好 3 个有限实数的向量（A 在时刻 t 的位置为
  `A + velocity * t`）。
- `t_start`, `t_end`：有限实数时间窗，要求 `t_start < t_end`。

### DistanceResult（frozen dataclass）

| 字段 | 含义 |
|---|---|
| `status` | `"separated"` / `"intersecting"` / `"not_converged"` |
| `distance` | 最近点距离；`intersecting` 时为 ≤ 零阈值的残差（**不是穿透深度**） |
| `witness_a` / `witness_b` | 两物体上的最近点（输入点的重心组合） |
| `separating_axis` | 单位向量 `n`（从 B 指向 A），分离时为可验证证书；相交时 `None`；未收敛时为未验证方向 |
| `iterations` | 实际迭代次数 |
| `residual` | 最终相对间隙；`separated` 时 ≤ `tol` |

### SweepResult（frozen dataclass）

| 字段 | 含义 |
|---|---|
| `status` | `"separated"`（永不接触）/ `"intersecting"`（存在正时长相交区间）/ `"touching"`（仅在一个瞬时接触）/ `"not_converged"` |
| `first_contact` | 进入接触时的 `ContactInfo`；未接触为 `None`；t_start 已相交时时刻为 t_start |
| `last_contact` | 退出接触时的 `ContactInfo`；未接触为 `None`；相切时与 `first_contact` 相同；窗末仍相交时时刻为 t_end |
| `interval` | 截断到窗口的相交区间 `(t_enter, t_exit)`；未接触为 `None`；相切时两端相等 |
| `initial_state` | t_start 处的 `"separated"` / `"intersecting"` |
| `relative_rest` | 相对速度数值上为零时为 `True` |
| `closest_time` / `closest_distance` | 整窗内最小间距及时刻（接触期间为 0） |
| `iterations` | 全部内部 GJK 求解的迭代次数合计 |
| `residual` | 最后一次 GJK 求解的相对间隙 |

`ContactInfo` 字段：`time`、`point_a`、`point_b`（该时刻世界坐标下的
两侧见证点，接触时在数值误差内重合）、`normal`（从 B 指向 A 的单位
法向；法向不唯一/深穿透时为 `None`）、`kind`，取值：

- `"impact"`：从分离侧进入相交；
- `"touch"`：仅一个瞬时的接触（含擦边相切、接触恰好落在窗端）；
- `"separation"`：从相交侧退出分离；
- `"initial_overlap"`：窗起点已相交；
- `"window_end"`：窗末仍相交（区间截断记录）；
- `"rest_overlap"`：相对静止且整窗相交。

## 扫描的数学口径

- Minkowski 差 C(t) = C0 + velocity·t 随时间仿射平移，到原点的距离
  平方 d(t)² 是时间的**凸函数**，故相交区间至多一个，且 d′ 不减。
- 在 GJK 活动单纯形保持不变的区间内，其 KKT 矩阵（活动顶点的 Gram
  阵）固定，重心权重随时间线性变化，最近点服从仿射律
  x(s) = x0 + q·s，q = velocity + Σ h_l·w_l。算法用该二次型预测驻点，
  **以独立 GJK 求解验证预测**，活动特征提前切换时将步长减半；一旦验证
  到 d′ ≥ 0，凸性即证明之后不可能再接触。
- 接触时刻用括号化 GJK 二分求根；精确相切瞬间静态 GJK 可能在退化单
  纯形上空转，此时用两侧抖动探针判定状态跨越。根再用分离侧距离与法向
  接近速率做线性精化。
- 接触时刻的可分辨精度 = max(float ulp, 零阈值距离 / 接近速率)；短于
  该精度的“相交区间”在 double 精度下本就不可表示，如实报 `touching`，
  不虚构正区间。

## 错误语义

所有非法输入抛出 `InputError`（`ValueError` 子类），消息可定位到具体
元素，例如 `A[3][1]: ...`、`velocity[2]: ...`、`t_start < t_end ...`：

- 坐标 / 速度分量为 NaN、±Infinity；
- 坐标 / 速度分量 / `tol` / 时间为 `bool` 或非实数；
- `max_iter` 为 `bool` 或非 1..128 整数；
- 点数 0 或 > 64、点维度 ≠ 3、速度维度 ≠ 3、时间窗反序或非有限；
- 输入为字符串等。

不做任何静默纠正。两个入口均为纯函数：输入校验全部完成后才开始计算，
失败不会留下部分变更的状态。

搜索预算耗尽（`"not_converged"`）与已证明接触严格区分，绝不互相
冒充；扫描中任一次求解预算耗尽都会整体报 `not_converged`。

## 误差口径（浮点）

- 全部计算在 IEEE 754 双精度中进行；有理数输入（`int`/`Fraction`）先
  转为 float64。
- 静态相交判定使用独立零阈值 `‖v‖ ≤ 1e-11 · max(1, 输入坐标最大模长)`，
  与 `tol` 分开；刚好接触（距离恰为 0）在静态入口落入
  `intersecting`。
- 扫描报告的接触时刻在正常接近速率下误差为 float ulp 量级；接触点为
  二分括号两侧见证点的平均，误差不超过零阈值距离。瞬时相切与正时长
  相交按“可分辨接触时间精度”区分（见上节）。
- 测试断言裕量（1e-6 量级）是对 float64 累积舍入与网格离散的留白，
  手算边界例（t=2、t=4、角点 (4, .5, 1) 等）均以精确期望值判定。

## 设计取舍

- **Support 函数 + GJK**：不显式构造凸包面/拓扑；扫描通过平移并热启动
  活动单纯形复用静态求解器。
- **KKT 系统无量纲化**：最近点重心坐标的 Gram 阵先除以其最大表项，再
  判奇异。否则长支撑向量下合法线段的约束主元会系统性偏小，被误判为仿射
  相关而使 GJK 在大坐标上永久循环（该问题在首轮静态实现中即存在，本次
  一并修复）。
- **降维处理退化**：单纯形最近点用 Johnson 式子集枚举，仿射相关子集
  判奇异后跳过，不使用随机抖动；相切瞬间的求解空转用确定性抖动探针
  解决。
- **凸性证书**：扫描不在时间上盲目密采样；驻点预测均经独立求解验证，
  最近距离为解析二次型在已验证片段上的最小值。
- **确定性**：支撑函数平局取最小下标，初始方向取质心差，无随机数，
  结果可复现。

## 已知限制

- 不提供穿透深度（EPA 超出范围）；静态入口的相交/接触统一报
  `intersecting`。
- 仅支持**单段定速平移**，不做轨迹拼接、旋转扫描或运动拓扑事件证书。
- `not_converged` 时扫描结果不可用（`interval=None`，`closest_distance
  =NaN`）。1900+ 组随机小凸体（含 1e-3..100 坐标量级与负时间窗）差分
  验证中未出现未收敛；极端几何体仍可能触发，此时结果如实标记而非
  猜测。
- 初始已深穿透时，t_start 处接触法向不唯一，记为 `None`；窗末仍相交
  时 `last_contact.normal` 同样为 `None`。
- 坐标量级极端（≳ 1e150 或 ≲ 1e-150）时 float64 上溢/下溢不在保障
  范围内。
- 只实现本题契约的数学子集（三维、凸包、≤ 64 点），不宣称兼容任何
  工业标准接口。

## 运行

```bash
python -m unittest discover -s tests -v   # 50 个测试（静态 23 + 扫描 27）
python demo.py                            # < 1 秒，正常/边界/拒绝示例
```

测试中的参考实现（点-线段解析式、轴对齐盒解析式、两盒扫描的逐轴解析
区间、小凸组合穷举 active-set 枚举、基于静态 distance 的稠密时间
网格）全部独立写在 `tests/` 内，不调用被测扫描核心生成期望值。
