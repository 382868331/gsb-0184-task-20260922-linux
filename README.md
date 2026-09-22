# convex_distance：凸多面体的最近点与分离证书（GJK）

纯 Python 标准库实现：给定两个由至多 64 个三维点表示的凸包
`A = conv(points_A)`、`B = conv(points_B)`，用 **GJK 算法**沿 support
函数迭代求 Minkowski 差 `A-B` 到原点的最近点，输出最近距离、两物体上的
**重心组合见证点**、单位**分离方向**证书，或相交/接触结论；搜索预算耗尽
与机器精度不可判定都会被明确标记，绝不与“证明相交”混淆。

## 运行环境

- **Python 3.14.7**（Linux 原生实测；代码只用 3.10+ 通用语法）
- **零第三方依赖**，仅标准库（`math`、`dataclasses`、`typing`、`unittest`）
- 安装 Python 后直接运行，无需网络、账号、密钥、数据库、中间件或 Docker

## 准确命令

```bash
# 必要测试（54 个用例）
python -m unittest discover -s tests -v

# 固定输入演示（约 0.04 秒，上限 8 秒内；真实计算，无 sleep/预录）
python demo.py
```

## 公开接口

```python
from convex_distance import distance, ConvexBody, box
from convex_distance import SEPARATED, INTERSECTING, NOT_CONVERGED
from convex_distance import (
    ConvexDistanceError, InvalidBodyError, InvalidParameterError,
    MAX_POINTS, MAX_ITERATIONS,   # 64, 128
)

result = distance(A, B, *, max_iterations=128,
                  tolerance_rel=1e-12, tolerance_abs=1e-12)
```

- `A`、`B`：`ConvexBody`，或长度为 3 的点的 `list/tuple`
  （每个物体 **1..64** 个点；坐标接受 `int`/`float`）。
- `max_iterations`：`1..128` 的 `int`。
- `tolerance_rel` / `tolerance_abs`：非负有限的收敛容差。

### DistanceResult 字段（不可变）

| 字段 | 含义 |
|---|---|
| `status` | `"separated"` / `"intersecting"` / `"not_converged"` |
| `converged` | 是否取得证书（前两者为 `True`） |
| `reason` | `"certified"` / `"budget_exhausted"` / `"machine_floor_ambiguity"` / `"numerical_contradiction"` |
| `distance` | 最近距离；相交为 `0.0`；未收敛为当前最佳**估计** |
| `witness_a`, `witness_b` | 两物体上的最近点（凸组合），满足 `wa - wb = distance·n` |
| `separation_direction` | 单位向量 `n`（由 B 指向 A）；不发证书时为 `None` |
| `residual` | support 证书残差 `r = |v| + d·w ≥ 0` |
| `witness_residual` | 证书自洽残差 `|wa - wb - v|` |
| `iterations` / `max_iterations` | 实际/预算迭代次数 |
| `singular_skipped` | 因仿射相关被拒绝的退化子集数（降维发生于此，无抖动） |
| `message` | 中文状态说明 |

便捷属性：`result.separated`、`result.intersecting`。

## 证书语义（分离时可独立核验）

`separated` 时，对返回的单位方向 `n`：

- `distance = min_{a∈A} n·a − max_{b∈B} n·b`（支撑面间隔）；
- `witness_a` 取到 A 侧最小值、`witness_b` 取到 B 侧最大值；
- `witness_a − witness_b = distance·n`。

调用方可以只凭输入点和 `n` 自行复核这三个等式（`residual`/
`witness_residual` 是模块报告的实测违反量）。

## 错误语义（非法输入不静默纠正）

- 坐标为 `NaN`/`±Infinity` → `InvalidBodyError`；
- 坐标或整数参数为 `bool`（`bool` 虽是 `int` 子类型）→ `TypeError`；
- 点不是长度 3 的序列、点集为空或超过 64 点、`max_iterations` 越界等
  → `TypeError` / `InvalidBodyError` / `InvalidParameterError`；
- 错误信息带可定位路径，如 `A[3][1]: 坐标必须有限，拒绝 NaN/Infinity`；
- 完全重复的点**确定性去重**（保留首次出现），其余顺序不变；
- 模块无状态：`ConvexBody` 在全部校验通过后才赋值，失败不留部分结果。

## 算法与设计取舍

1. **只用 support，不先构面**。`ConvexBody.support(d)` 返回
   `argmax p·d`（并列取最小下标，确定性）。GJK 主循环每轮在当前
   Minkowski 单纯形（≤4 顶点）上求最近点 `v`，再取
   `w = h_A(−v/|v|) − h_B(v/|v|)`，残差 `r = |v| + d·w` 同时给出
   距离上界与支撑下界，`r≈0` 即为可核验证书。
2. **Johnson 风格全子集枚举**。枚举单纯形的全部非空子集
   （≤15 个），对每个子集解“原点到仿射包的最近点”，权全非负者可行，
   取范数最小。
3. **棱向量 Gram + 接触友好**。以子集第一个点 `y0` 为基点，解
   `(e_i·e_j) t = −e·y0`（`e_i = y_i − y0`）。即使仿射包**穿过原点**
   （两物体刚好接触），该矩阵仍良态；直接用环境坐标 Gram（`YᵀY`）
   在此情形会奇异。
4. **退化降维，不随机抖动**。棱向量 Gram 奇异 ⇔ 子集仿射相关
   （重复点、共线、共面）。跳过该子集即可——同一凸包必由其某个
   仿射独立真子集表示，而真子集也在枚举范围内。于是退化线段自动
   退到端点、退化三角形自动退到边/点；坐标从不被扰动。
5. **见证维护**。每个 Minkowski 顶点记录来源下标 `(ai, bi)`，
   最近点子集的凸权重同时给出两物体上的重心组合见证，无需二次搜索。
6. **停滞判据只看当前单纯形**。新 support 顶点若已在当前单纯形中
   才是真停滞；用“全局历史集合”判停滞是错误的——曾被裁剪的顶点
   可在新方向下重新成为极值点并带来实质进展（开发期 fuzz 抓到并修复）。

## 误差口径（浮点，不用宽容差掩盖边界错误）

主算法为 IEEE-754 双精度。设构型半径
`R = max |Minkowski 单纯形顶点|`（与两物体坐标尺度同阶）：

- **距离数值**：适度尺度（如 `R ≤ 1e4`）下，对拍独立 KKT 枚举参考，
  距离误差经验上 `< 1e-9·max(1,R)`；`residual`/`witness_residual`
  报告真实残差，容差不会改写坐标或顶点。
- **相交/接触判定**：`|v| ≤ 64·ε·R`（`ε = 2.22e-16`）才判
  `intersecting`。该判据**刻意不使用** `tolerance_rel·R`——否则在
  `R = 1e12` 的构型上，`1e-12·R = 1` 会把真实的 1.5 缝隙吞成“相交”。
  已验证：`R = 2e12` 盒上 1.5 的真实缝隙仍精确报 `separated`，
  而真实面接触在 `R = 2e12` 仍正确报 `intersecting`。
- **分离证书残差**：活动迭代地板 `64·ε·R`；support 停滞时方向由
  可能病态的 Gram 解给出，残差地板放大为
  `32·ε·R·max(1, R/|v|)`。放大地板**仅在严格小于 `0.25·|v|`**
  （即结构条件 `d·w < −0.75|v|`，支撑面确实越过原点）时才接受证书。
- **机器地板不可判定**：当噪声地板与当前距离同阶（`≥ 0.25·|v|`），
  双精度已无法区分“接触”与“微小缝隙”，此时返回
  `status="not_converged"`、`reason="machine_floor_ambiguity"`、
  `separation_direction=None`。这与 `budget_exhausted`（128 次预算
  用完）和 `intersecting`（原点在凸包内的肯定证书）三者严格分开。
  典型触发：`R ≈ 2e5` 的盒在面上接触时，约 `6e-7` 以下的缝隙不可分辨；
  `demo.py` 第 3b 节用固定种子的真实构型演示。
- 常数 64/32/0.25 由尺度扫描（`R=1..1e12`、缝隙 0..10）经验标定，
  不是数学上界；极端病态构型可能进入 `numerical_contradiction`
  （停滞且残差超出地板），同样按未收敛处理，不发假证书。
- 输入 `int` 经 `float()` 转换：绝对值超过 `2^53` 的整数会有
  表示误差，请在该尺度内使用。

## 验证方式（独立参考不调用被测核心）

- `tests/reference.py`：**完全独立**的参考（不导入 `convex_distance`）：
  点-线段解析公式、轴对齐盒解析最近点、以及在重心变量上枚举 KKT
  活动集解小线性系统的凸包距离穷举、`in_convex_hull` 仿射枚举。
  期望值只由它产生；“往返一致”类断言只作补充。
- 54 个 `unittest` 用例覆盖：点-线段/盒解析对拍、180 组随机小点云
  枚举穷举对拍 + 见证落在凸包内、共面/重复/共线降维、刚好接触
  （面/边/顶点）、平移不变、对称性、输入错误语义、64 点上限、
  尺度鲁棒性、预算耗尽与机器地板不可判定的实际触发。
- 开发期另用 650 组随机/对抗 fuzz（已从交付物移除）对拍独立参考，
  含 64 点、退化线/面、`R` 至 `1e10` 的近接触扫描，交付前 0 失败。

## 已知限制（本题数学子集，非行业标准完整实现）

- 仅实现两静止凸包的欧氏最近距离/分离证书；**不**实现 EPA 穿透深度、
  连续碰撞检测、射线检测、有符号内部距离（相交只报 0）、非凸输入。
- 上限：每物体 64 点、3 维、128 GJK 迭代；不做 GPU/空间加速，
  support 为 O(n) 线性扫描（n≤64）。
- 双精度浮点，不是精确有理运算；误差口径见上一节，
  `machine_floor_ambiguity` 是被显式报告而非隐藏的精度失败。
- `convex_distance.box()` 便捷构造器不校验 `lo ≤ hi`
  （两角点的凸包与顺序无关，交换不改变几何）。

## 文件

```
convex_distance/
  __init__.py    公开接口
  errors.py      输入/参数校验与异常（可定位错误信息）
  vectors.py     三维向量与容差
  bodies.py      ConvexBody / box：support 函数访问，不构面
  simplex.py     Johnson 子集枚举、棱向量 Gram、退化降维
  gjk.py         GJK 主循环、见证、证书与精度裁决
tests/
  reference.py   独立参考（解析 + KKT 枚举穷举，不导入被测包）
  test_inputs.py / test_simplex.py / test_geometry.py / test_scale.py
demo.py          固定输入演示
```
